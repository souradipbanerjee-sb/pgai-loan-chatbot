"""
Conversation engine for the two-wheeler loan chatbot.

Design
  - A deterministic slot-filling state machine owns the flow, validation and the decision.
  - The Groq LLM is used for two things only:
      1. extraction: turn free text ("1.2 lakh, scooty, no EMIs") into structured fields
      2. phrasing: write the next friendly question
    No name or contact details are collected; any identifiers typed anyway are masked before
    the LLM or the log sees them. The LLM never produces the decision text.
  - If Groq is unavailable, a local rule-based parser and template questions take over.
  - The decision comes only from loan_classifier.classify() (XGBoost probability vs 0.50).

Dynamic sequence
  - Several fields can be captured from one message; only missing fields are asked.
  - The customer can give a down payment instead of a loan amount (loan = price - down payment).
  - LTV is computed (loan / on-road price) - customers are never asked for it.
  - Salary that looks yearly triggers a monthly/yearly check; values are re-validated on correction.
  - Age outside the model's range ends the flow with a referral instead of a model score.
  - A decline with a feasible lower loan amount gets a counter-offer that can be accepted in chat.
"""
import json, os, re, time, uuid

import requests

import loan_classifier as clf
from loan_repository import Repository, _now as now_utc
from pii import mask_text

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_FALLBACK_MODEL = "openai/gpt-oss-120b"

PRODUCTS = {"MC": "motorcycle", "SC": "scooter", "MO": "moped", "EB": "electric two-wheeler"}
FIELD_STEPS = ["age", "product", "vehicle_price", "loan_amount", "net_salary", "past_loans"]
QUESTIONS = {
    "age": "How old are you?",
    "product": "Which type of two-wheeler are you buying: a motorcycle, scooter, moped or electric?",
    "vehicle_price": "What is the on-road price of the vehicle (in Rs)?",
    "loan_amount": "How much would you like to borrow? You can also just tell me your down payment.",
    "net_salary": "What is your take-home (net) monthly income?",
    "past_loans": "Do you currently have any other loans or EMIs running? (yes / no)",
    "salary_check": "Just to check, is Rs {amount:,} your monthly or yearly income?",
}


HINTS = {"age": "(e.g. 32) ", "product": "(e.g. scooter) ", "vehicle_price": "(e.g. 95000 or 1.2 lakh) ",
         "loan_amount": "(e.g. loan 80000, or down payment 20000) ", "net_salary": "(e.g. 30000) ",
         "past_loans": "(yes or no) ", "salary_check": "(monthly or yearly) "}


def _err(e):
    """Short error description for the log (status code + API error code, never the request)."""
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            code = resp.json().get("error", {}).get("code", "")
        except Exception:
            code = ""
        return f"HTTP {resp.status_code} {code}".strip()
    return type(e).__name__


def rs(x):
    return f"Rs {int(round(x)):,}"


# ---------------------------------------------------------------- LLM client
class GroqClient:
    def __init__(self, api_key, model=GROQ_MODEL, timeout=8):
        self.api_key, self.model, self.timeout = api_key, model, timeout

    def chat(self, messages, json_mode=False, max_tokens=400, temperature=0.2, _retry=True):
        # gpt-oss models reason before answering: low effort + room for reasoning tokens
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "reasoning_effort": "low"}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(GROQ_URL, json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        if r.status_code == 404 and self.model != GROQ_FALLBACK_MODEL:   # model retired -> fallback
            self.model = GROQ_FALLBACK_MODEL
            return self.chat(messages, json_mode, max_tokens, temperature)
        if r.status_code == 429 and _retry:                              # rate limited -> one short retry
            time.sleep(min(float(r.headers.get("retry-after", 1) or 1), 2.0))
            return self.chat(messages, json_mode, max_tokens, temperature, _retry=False)
        if r.status_code == 400 and json_mode and "json_validate_failed" in r.text:
            return "{}"                                                  # no fields (e.g. a question)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


EXTRACT_PROMPT = """You extract loan-application fields from a customer's chat message for a two-wheeler loan in India.
Return ONLY a JSON object. Allowed keys:
  age (integer years)
  product ("MC" motorcycle/bike, "SC" scooter/scooty, "MO" moped e.g. TVS XL, "EB" electric two-wheeler)
  vehicle_price (on-road price in INR, number)
  loan_amount (amount to borrow in INR, number)
  down_payment (INR the customer pays upfront, number)
  net_salary (take-home income in INR, number)
  salary_period ("monthly" or "annual", only if the customer says so)
  past_loans_active (true if the customer currently has any other active loan or EMI, false if none)
  confirm (true if the customer agrees / says yes / correct, false if they disagree or want a change)
Convert words to numbers: 1 lakh = 100000, 50k = 50000, 1 crore = 10000000.
Interpret a bare number or yes/no as the answer to the question that was asked.
Include only values the customer actually stated. Return {} if none.
Question that was asked: "%s"
"""

REPLY_PROMPT = """You are the PGAI Loan Assistant, a warm, concise chatbot helping customers apply for a two-wheeler loan in India.
Write the next chat message. Rules:
- At most 3 short sentences, plain text, no markdown, no emojis.
- If values were captured this turn, acknowledge them briefly (do not repeat every number).
- If there are issues, explain them kindly.
- If the customer asked a question, answer it in one sentence with general information only.
- End by asking exactly the NEXT QUESTION given (you may rephrase it naturally).
- Never say or hint whether the loan will be approved or declined. Never ask for ID numbers,
  bank details, passwords or OTPs. Never invent numbers."""


# ---------------------------------------------------------------- local parsing
_UNITS = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5, "k": 1e3, "thousand": 1e3,
          "crore": 1e7, "cr": 1e7}


_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lakhs": 100000, "lac": 100000, "crore": 10000000}


def words_to_digits(text):
    """'seventy two thousand' -> '72000', 'thirty one' -> '31' (leaves other words alone)."""
    out, total, cur, active = [], 0, 0, False
    for tok in re.findall(r"[a-z]+|[^a-z]+", text.lower()):
        w = tok.strip()
        if w in _ONES or w in _TENS:
            cur += _ONES.get(w, 0) + _TENS.get(w, 0)
            active = True
        elif w in _SCALES and active:
            if w == "hundred":
                cur = max(cur, 1) * 100
            else:
                total, cur = total + max(cur, 1) * _SCALES[w], 0
        elif active and (tok.isspace() or w in ("and", "-")):
            continue
        else:
            if active:
                out.append(f" {total + cur} ")
                total, cur, active = 0, 0, False
            out.append(tok)
    if active:
        out.append(f" {total + cur}")
    return "".join(out)


def parse_amounts(text):
    t = words_to_digits(text).lower().replace(",", "")
    out = []
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crore|cr|thousand|k|l)?\b", t):
        out.append(float(num) * _UNITS.get(unit, 1))
    return out


def parse_yes_no(text):
    t = text.lower()
    if re.search(r"\b(no|nope|none|nah|not|don't|dont|never|zero)\b", t):
        return False
    if re.search(r"\b(yes|yeah|yep|yup|sure|ok|okay|correct|right|agree|i do|have|haan|ha)\b", t):
        return True
    return None


_PRODUCT_WORDS = {"EB": ["electric", "ev", "e-bike", "iqube"], "MO": ["moped", "xl"],
                  "SC": ["scooter", "scooty", "jupiter", "ntorq", "zest"],
                  "MC": ["motorcycle", "bike", "apache", "raider", "radeon", "sport", "ronin"]}
_KEYWORDS = [("down_payment", r"down|upfront"), ("loan_amount", r"loan|borrow|financ"),
             ("vehicle_price", r"price|cost|on-?road|worth"),
             ("net_salary", r"income|salary|earn|take[- ]?home|per month|a month"),
             ("age", r"\bage\b|aged|years|yrs|\bold\b|\bi'?m\b|\bi am\b")]


def parse_product(text):
    t = text.lower()
    for code, words in _PRODUCT_WORDS.items():
        if any(re.search(rf"\b{w}\b", t) for w in words):
            return code
    return None


def parse_keywords(text):
    """Offline multi-fact parser: '42 years, scooter, price 1.1 lakh, pay 45k down' -> fields."""
    out = {}
    for clause in re.split(r",\s|;|\band\b|\.\s", text.lower()):
        nums = parse_amounts(clause)
        if not nums:
            continue
        for field, pat in _KEYWORDS:
            if re.search(pat, clause) and field not in out:
                if field == "age" and not 1 <= nums[0] <= 120:
                    continue
                out[field] = int(nums[0]) if field == "age" else nums[0]
                if field == "net_salary" and re.search(r"year|annual|lpa", clause):
                    out["salary_period"] = "annual"
                break
    p = parse_product(text)
    if p:
        out["product"] = p
    return out


def parse_local(text, step):
    t, nums = text.lower(), parse_amounts(text)
    out = {}
    if step == "age" and nums:
        out["age"] = int(nums[0])
    elif step == "product":
        p = parse_product(text)
        if p:
            out["product"] = p
    elif step == "vehicle_price" and nums:
        out["vehicle_price"] = nums[0]
    elif step == "loan_amount" and nums:
        out["down_payment" if re.search(r"down|upfront|pay\b", t) else "loan_amount"] = nums[0]
    elif step == "net_salary" and nums:
        out["net_salary"] = nums[0]
        if re.search(r"year|annual|p\.?a|lpa", t):
            out["salary_period"] = "annual"
    elif step == "salary_check":
        if re.search(r"year|annual", t):
            out["salary_period"] = "annual"
        elif re.search(r"month", t):
            out["salary_period"] = "monthly"
    elif step == "past_loans":
        yn = parse_yes_no(text)
        if yn is not None:
            out["past_loans_active"] = yn
    elif step == "confirm":
        yn = parse_yes_no(text)
        if yn is not None:
            out["confirm"] = yn
    return out


# ---------------------------------------------------------------- engine
class LoanChatbot:
    def __init__(self, channel="cli", api_key=None, repo=None, use_llm=True):
        self.session_id = uuid.uuid4().hex
        self.channel = channel
        self.repo = repo or Repository()
        key = api_key or os.environ.get("GROQ_API_KEY")
        self.llm = GroqClient(key) if (use_llm and key) else None
        self.step = "consent"
        self.app = {}                 # model-relevant answers (no PII)
        self.pending = {}             # values awaiting another field / a check
        self.offer = None
        self.done = False
        self.started = time.perf_counter()
        self.n_user_turns = 0
        self.last_result = None
        self.decisions = []
        self.repo.start_session(self.session_id, channel)

    # ---------- public API
    def start(self):
        msg = ("Hi! I'm the PGAI Loan Assistant. I can check your two-wheeler loan eligibility in about 2 minutes "
               "with 6 quick questions. We don't ask for your name or contact details.\n"
               "Privacy: please don't share personal identifiers here; any phone numbers, emails or ID numbers "
               "you type are masked before they are stored or sent to the AI service.\n"
               "Do you agree to continue? (yes / no)")
        return self._bot(msg)

    def handle(self, text):
        t0 = time.perf_counter()
        text = (text or "").strip()
        self.n_user_turns += 1
        self.repo.log_message(self.session_id, "user", self.step, self._mask(text))
        if self.done:
            return self._bot("This application is complete. Please start a new chat to apply again.")
        if not text:
            return self._bot("Sorry, I didn't catch that. " + self._question())
        handler = {"consent": self._on_consent,
                   "post_decision": self._on_post_decision}.get(self.step, self._on_field)
        reply, llm_used = handler(text, t0)
        return self._bot(reply, latency_ms=(time.perf_counter() - t0) * 1000, llm_used=llm_used)

    def _on_consent(self, text, t0):
        yn = parse_yes_no(text)
        if yn is None:
            return "Please reply yes to continue or no to stop.", False
        self.repo.update_session(self.session_id, consent=int(yn))
        if not yn:
            self._finish("no_consent")
            return "No problem, nothing has been stored beyond this chat. Have a great day!", False
        self.step = self._next_step()
        return f"Thank you! Let's begin. {self._question()}", False

    # ---------- application fields (LLM-assisted)
    def _on_field(self, text, t0):
        asked, prev_step = self._question(), self.step
        masked = self._mask(text)
        # Fast path: a plain "yes" at the summary goes straight to the model (no LLM round trip)
        if prev_step == "confirm" and parse_local(text, "confirm").get("confirm") is True \
                and not parse_amounts(text) and len(text.split()) <= 4:
            return self._decide(t0), False
        fields, llm_used = self._extract(masked, asked)
        if not llm_used or not fields:                       # local parsers back up the LLM
            kw = parse_keywords(text)
            # a bare answer fills the field just asked; a keyword ("down", "loan", "income") wins over that
            local = {} if set(kw) - {"product"} else parse_local(text, self.step)
            for k, v in {**local, **kw}.items():
                fields.setdefault(k, v)
        elif self.step == "confirm" and "confirm" not in fields:
            fields.update(parse_local(text, "confirm"))
        captured, issues = self._apply(fields)
        if self.done:                                        # hard stop (age outside policy)
            return issues[0], llm_used

        if prev_step == "confirm" and not captured and not issues and fields.get("confirm") is True:
            return self._decide(t0), llm_used
        self.step = self._next_step()
        if self.step == "confirm":
            if prev_step == "confirm" and not captured:
                msg = " ".join(issues) or "Sure, what would you like to change? For example: 'my income is 40,000'."
                return msg, llm_used
            return ("Updated. " if prev_step == "confirm" else "") + self._summary(), llm_used
        reply, used = self._phrase(masked, captured, issues, self._question())
        return reply, llm_used or used

    def _extract(self, masked_text, asked):
        if not self.llm:
            return {}, False
        try:
            raw = self.llm.chat([{"role": "system", "content": EXTRACT_PROMPT % asked},
                                 {"role": "user", "content": masked_text}],
                                json_mode=True, max_tokens=150, temperature=0)
            data = json.loads(raw)
            return ({k: v for k, v in data.items() if v is not None} if isinstance(data, dict) else {}), True
        except Exception as e:  # network / quota / parse -> local fallback
            self.repo.log_message(self.session_id, "system", self.step, f"LLM extract failed: {_err(e)}")
            return {}, False

    def _phrase(self, masked_text, captured, issues, next_q):
        ack = "Got it. " if captured and not issues else ""
        if not captured and not issues:
            ack = "Sorry, I didn't quite catch that. " + HINTS.get(self.step, "")
        fallback = ack + " ".join(issues + [next_q])
        asked_question = "?" in masked_text or re.match(r"(?i)\s*(what|why|how|can|is|do|does|will)\b", masked_text)
        # LLM wording only where it adds value (customer question / validation problem / nothing understood)
        if not self.llm or not (asked_question or issues or not captured):
            return fallback, False
        try:
            payload = {"customer_message": masked_text, "captured_this_turn": captured,
                       "issues": issues, "NEXT_QUESTION": next_q}
            out = self.llm.chat([{"role": "system", "content": REPLY_PROMPT},
                                 {"role": "user", "content": json.dumps(payload)}], max_tokens=120, temperature=0.4)
            return out, True
        except Exception as e:
            self.repo.log_message(self.session_id, "system", self.step, f"LLM phrase failed: {_err(e)}")
            return fallback, False

    def _num(self, v):
        if isinstance(v, (int, float)):
            return float(v)
        vals = parse_amounts(str(v))
        return vals[0] if vals else None

    def _apply(self, f):
        """Validate + store fields. Returns (captured dict for display, list of issue strings)."""
        captured, issues = {}, []

        def ok(field, value, note=""):
            self.app[field] = value
            captured[field] = value
            self.repo.log_field(self.session_id, field, value, "accepted", note)

        def bad(field, value, msg):
            issues.append(msg)
            self.repo.log_field(self.session_id, field, value, "rejected", msg)

        if "age" in f:
            a = self._num(f["age"])
            if a is None or not 1 <= a <= 120:
                bad("age", f["age"], "Please give your age in years.")
            elif a < 18:
                self.repo.log_field(self.session_id, "age", a, "ineligible", "under 18")
                self._finish("ineligible")
                issues.insert(0, "Sorry, applicants must be at least 18 years old, so I can't take this application "
                                 "forward. Thank you for your interest!")
                return captured, issues
            elif a > 69:
                self.repo.log_field(self.session_id, "age", a, "referred", "above model range 18-69")
                self._finish("referred")
                issues.insert(0, "Thanks! For applicants above 69 our branch team reviews applications personally. "
                                 "Please visit your nearest PGAI branch or dealer and they'll help you apply.")
                return captured, issues
            else:
                ok("age", int(a))

        if "product" in f:
            p = str(f["product"]).upper()
            if p in PRODUCTS:
                ok("product", p)
            else:
                bad("product", p, "I can help with motorcycles, scooters, mopeds or electric two-wheelers.")

        if "vehicle_price" in f:
            v = self._num(f["vehicle_price"])
            if v is None or v <= 0:
                bad("vehicle_price", f["vehicle_price"], "Please share the on-road price as an amount in Rs.")
            else:
                ok("vehicle_price", v)

        if "down_payment" in f and "loan_amount" not in f:
            d = self._num(f["down_payment"])
            if d is not None:
                if "vehicle_price" in self.app:
                    f["loan_amount"] = self.app["vehicle_price"] - d
                    self.repo.log_field(self.session_id, "down_payment", d, "accepted", "loan = price - down payment")
                else:
                    self.pending["down_payment"] = d
        if "vehicle_price" in self.app and "down_payment" in self.pending:
            f.setdefault("loan_amount", self.app["vehicle_price"] - self.pending.pop("down_payment"))

        if "loan_amount" in f:
            la = self._num(f["loan_amount"])
            price = self.app.get("vehicle_price")
            if la is None or la <= 0:
                bad("loan_amount", f["loan_amount"], "Please share the loan amount as an amount in Rs.")
            elif price and la > price:   # consistency check only: LTV above 100% isn't a valid loan
                bad("loan_amount", la, f"The loan can't be more than the on-road price of {rs(price)}.")
            else:
                ok("loan_amount", float(round(la)))
        if "loan_amount" in self.app and "vehicle_price" in captured and self.app["loan_amount"] > self.app["vehicle_price"]:
            bad("loan_amount", self.app.pop("loan_amount"),
                "The loan amount was higher than the vehicle price, so I'll need it again.")

        if "salary_period" in f and "net_salary" in self.pending:
            s = self.pending.pop("net_salary")
            ok("net_salary", s / 12 if f["salary_period"] == "annual" else s,
               "converted from yearly" if f["salary_period"] == "annual" else "confirmed monthly")
        elif "net_salary" in f:
            s = self._num(f["net_salary"])
            if s is None or s < 0:
                bad("net_salary", f["net_salary"], "Please share your monthly take-home income as a number.")
            elif f.get("salary_period") == "annual":
                ok("net_salary", s / 12, "converted from yearly")
            elif s > 200000:
                self.pending["net_salary"] = s          # dynamic follow-up: monthly or yearly? (no cap)
                self.app.pop("net_salary", None)
            else:
                ok("net_salary", s)

        if "past_loans_active" in f and isinstance(f["past_loans_active"], bool):
            ok("past_loans_active", f["past_loans_active"])
        return captured, issues

    def _next_step(self):
        if "net_salary" in self.pending:
            return "salary_check"
        for s in FIELD_STEPS:
            key = "past_loans_active" if s == "past_loans" else s
            if key not in self.app:
                return s
        return "confirm"

    def _question(self):
        if self.step == "salary_check":
            return QUESTIONS["salary_check"].format(amount=int(self.pending["net_salary"]))
        if self.step == "confirm":
            return "Is everything correct? (yes / or tell me what to change)"
        return QUESTIONS.get(self.step, "")

    def _summary(self):
        a = self.app
        ltv = a["loan_amount"] / a["vehicle_price"] * 100
        return ("Here's your application:\n"
                f"  - Age: {a['age']}\n"
                f"  - Vehicle: {PRODUCTS[a['product']]}, on-road price {rs(a['vehicle_price'])}\n"
                f"  - Loan amount: {rs(a['loan_amount'])} (down payment {rs(a['vehicle_price'] - a['loan_amount'])}, "
                f"{ltv:.0f}% financed)\n"
                f"  - Net monthly income: {rs(a['net_salary'])}\n"
                f"  - Other active loans/EMIs: {'yes' if a['past_loans_active'] else 'no'}\n"
                "Is everything correct? Reply yes to get your decision, or tell me what to change.")

    # ---------- decision
    def _features(self, loan_amount=None):
        a = self.app
        la = loan_amount or a["loan_amount"]
        return {"LTV": round(la / a["vehicle_price"] * 100, 2), "PAST_LOANS_ACTIVE": int(a["past_loans_active"]),
                "Age": a["age"], "Product_Code_MC": int(a["product"] == "MC"),
                "Loan_Amount": la, "Net_salary": a["net_salary"]}

    def _decide(self, t0):
        feats = self._features()
        result = clf.classify(feats)
        offer = clf.counter_offer(feats, self.app["vehicle_price"]) if result["decision"] == "DECLINED" else None
        ref = "PGAI-" + uuid.uuid4().hex[:8].upper()
        latency = round((time.perf_counter() - t0) * 1000, 2)
        self.repo.log_decision(self.session_id, ref, self.app, feats, result, latency, offer)
        self.last_result = dict(result, reference=ref, decision_latency_ms=latency, features=feats)
        self.decisions.append(self.last_result)
        a = self.app
        if result["decision"] == "APPROVED":
            self._finish("approved")
            return (f"Good news! Your loan of {rs(a['loan_amount'])} for your {PRODUCTS[a['product']]} "
                    f"is provisionally APPROVED.\nReference: {ref} (please note it down)\n"
                    "Next steps:\n"
                    "  1. Quote your reference number at your dealer or in the PGAI app.\n"
                    "  2. Complete KYC there (PAN and Aadhaar) and share your last 3 months' salary slips or bank "
                    "statement. Please don't share these documents in this chat.\n"
                    "  3. After verification and e-signing, the amount is paid directly to your dealer and you can "
                    "take delivery.")
        msg = (f"Thank you. We're unable to approve a loan of {rs(a['loan_amount'])} at this time.\n"
               f"Reference: {ref}\n")
        if offer:
            self.offer = offer
            self.step = "post_decision"
            return msg + (f"Good news: with a higher down payment of {rs(offer['down_payment'])} "
                          f"(loan of {rs(offer['loan_amount'])}), your application would likely be approved. "
                          "Would you like to re-apply with that amount? (yes / no)")
        self._finish("declined")
        return msg + self._decline_next_steps()

    def _decline_next_steps(self):
        return ("What you can do next:\n"
                "  1. Speak to a loan advisor at your nearest branch or dealer about other options.\n"
                "  2. Consider a larger down payment or a lower-priced model; this has the biggest effect.\n"
                "  3. You can re-apply after 3 months, ideally after closing any running EMIs.\n"
                "Thank you for considering PGAI.")

    def _on_post_decision(self, text, t0):
        yn = parse_yes_no(text)
        if yn is None:
            return "Please reply yes to re-apply with the suggested amount, or no to finish.", False
        if not yn:
            self._finish("declined")
            return "Understood. " + self._decline_next_steps(), False
        self.repo.log_field(self.session_id, "loan_amount", self.offer["loan_amount"], "accepted", "counter-offer")
        self.app["loan_amount"] = float(self.offer["loan_amount"])
        self.offer = None
        return self._decide(t0), False

    # ---------- helpers
    def _mask(self, text):
        return mask_text(text)

    def _bot(self, text, latency_ms=None, llm_used=False):
        self.repo.log_message(self.session_id, "bot", self.step, self._mask(text), latency_ms, llm_used)
        return text

    def _finish(self, status):
        self.done = True
        self.step = "done"
        self.repo.update_session(self.session_id, status=status, ended_at=now_utc(),
                                 duration_s=round(time.perf_counter() - self.started, 1),
                                 n_user_turns=self.n_user_turns)

    @property
    def elapsed_s(self):
        return time.perf_counter() - self.started
