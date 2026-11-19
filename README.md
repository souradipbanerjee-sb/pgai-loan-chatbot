# PGAI two-wheeler loan chatbot

A text chatbot that collects an application conversationally, scores it with the frozen
XGBoost model (6 features, threshold 0.50) and returns Approve / Decline with next steps.

## Run

```powershell
& "$env:USERPROFILE\anaconda3\python.exe" loan_chatbot_cli.py --debug        # terminal chat
& "$env:USERPROFILE\anaconda3\Scripts\streamlit.exe" run loan_chatbot_app.py  # web chat
& "$env:USERPROFILE\anaconda3\python.exe" test_conversations.py              # 7 scripted conversations
& "$env:USERPROFILE\anaconda3\python.exe" evaluate_benchmarks.py             # objectives vs benchmarks
```
`--offline` (CLI / tests) runs without Groq. The key lives in `.env` (`GROQ_API_KEY`, `GROQ_MODEL`).

## Architecture

```
 Customer (CLI / Streamlit)
        |  text
        v
 chatbot_engine.LoanChatbot  ---- deterministic state machine (owns flow, validation, decision)
   | consent (no name or contact details are asked)
   | age -> product -> on-road price -> loan or down payment -> net income -> other EMIs
   | -> summary & confirm -> decision -> next steps / counter-offer
   |
   |-- pii.mask_text()  ----------> Groq LLM (openai/gpt-oss-20b)
   |                                  1. extract fields as JSON from free text
   |                                  2. word replies only when the customer asks a question
   |                                     or an answer fails validation
   |                                (one retry if rate-limited; local rule parser +
   |                                 templates if Groq is down)
   |
   |-- loan_classifier.classify()  -> XGBoost  P(decline) >= 0.50 -> DECLINED
   |-- loan_classifier.counter_offer() -> largest loan (lower LTV) that would be approved
   |
   '-- loan_repository (SQLite loan_chatbot.db): sessions, messages, field_events, decisions
```

Why this split: the LLM handles messy language ("around seventy two thousand", "scooty",
"nope, nothing running"), but it never decides, only sees masked text,
and can't invent a decision message. The approve/decline wording is a fixed template
driven only by the model's probability.

## Dynamic question sequence
- Several answers in one message are all captured; only missing fields are asked next.
- The customer can give a down payment instead of a loan amount (loan = price - down payment).
- LTV is calculated (loan / on-road price), not asked; customers don't know their LTV.
- An income above Rs 2 lakh triggers "monthly or yearly?" and yearly figures are converted.
  This is a clarification, not a cap: any income is accepted.
- No minimum or maximum on vehicle price, loan amount or income. The only check is
  consistency: a loan larger than the vehicle price (over 100% financed) is re-asked.
- Amounts in words ("seventy two thousand", "1.2 lakh") are understood, including offline.
- Age < 18 ends the application; age > 69 (outside the training data) is referred to a branch
  instead of scored by the model.
- Any field can be corrected at the summary ("actually my age is 38").
- A decline with a feasible lower loan gets a counter-offer the customer can accept in chat.

## Objectives and benchmarks

| Objective | Target | Justification | Current |
|---|---|---|---|
| a. Completion time (X) | **<= 3 min** median, <= 4 min p95 | At most 8 short turns (consent, 6 loan facts, confirm), fewer when several facts come in one message. At ~15 s to read and type per turn plus ~1 s bot time, a typical application takes about 2-2.5 min, quicker than a dealer-desk form and short enough to keep drop-off low. Corrections add 1-2 turns, hence the 4-min p95. | 141 s median, 156 s p95 (estimated, see note) |
| b. Recall, declined | >= 0.70 | Catching bad loans is the main purpose; a missed decline costs the loan amount. | 0.755 |
| b. Precision, approved | >= 0.97 | Quality of the approved book; the all-approve baseline is 0.947. | 0.980 |
| b. Precision, declined | >= 2x base rate (0.105) | With 5% declines, high precision is unrealistic; a flag must be at least twice as informative as chance. | 0.108 |
| b. Recall, approved | >= 0.60 | Keeps at least 60% of good customers on the instant path. | 0.653 |
| b. Accuracy / ROC-AUC | >= 0.60 / >= 0.75 | Accuracy is only a floor (all-approve scores 0.947 with zero declines caught); AUC checks ranking power. | 0.658 / 0.765 |
| c. Decision time (Y) | **<= 2 s** after final input | Feels instant in chat. The decision path is the model call only (no LLM), so it's well inside the target. | p95 22 ms, max 23 ms |
| d. Repository | 100% of sessions logged | Every message, field value (accepted, rejected, corrected), score, threshold, decision and latency. | 100% |

Model metrics are on the held-out test set (16,259 rows). Latency and completion figures
come from `test_conversations.py`. Completion time is an **estimate** (15 s per human turn
plus measured bot time); measure it with real users before relying on it.

## Privacy and security
- Consent is asked before any data is collected.
- No name or contact details are collected; the applicant is identified only by a
  reference number they quote at the dealer.
- If a customer types identifiers anyway, `mask_text()` replaces them (mobile, email,
  Aadhaar, PAN, account numbers) before the text is logged or sent to the LLM.
- The bot never asks for ID numbers, bank details or OTPs; KYC goes through the official
  channel.
- The customer is never shown the probability score; it's in the log and the operator view.
- Production to-dos: rotate the API key, encrypt the DB at rest, add retention and
  deletion rules.
