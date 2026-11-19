"""
Audit repository: every message, captured feature value, model score and decision.

Two sinks, written in parallel:
  1. SQLite  (loan_chatbot.db)  - always on; permanent for local runs, temporary on Streamlit Cloud
  2. Google Sheets              - on when SHEETS_WEBHOOK_URL is set (env / Streamlit secrets);
                                  rows are posted in the background to a Google Apps Script web app
                                  (google_apps_script.gs), so logging never slows the chat down.

Tables / sheet tabs
  sessions     - session events: started, consent, finished (status, duration, turns)
  messages     - every bot / user turn, PII-masked, with turn latency
  field_events - every feature value captured, rejected or corrected, with the reason
  decisions    - final feature vector, model score, threshold, decision, latencies, counter-offer
"""
import atexit, json, os, queue, sqlite3, threading, time
from datetime import datetime, timezone

import requests

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_chatbot.db")
_lock = threading.Lock()

# Column order = sheet header order
COLUMNS = {
    "sessions": ["ts", "session_id", "event", "channel", "consent", "status", "duration_s", "n_user_turns"],
    "messages": ["ts", "session_id", "role", "step", "text_masked", "latency_ms", "llm_used"],
    "field_events": ["ts", "session_id", "field", "value", "status", "note"],
    "decisions": ["ts", "session_id", "reference", "age", "product", "product_code_mc", "vehicle_price",
                  "loan_amount", "ltv", "net_salary", "past_loans_active", "probability_decline", "threshold",
                  "decision", "model_version", "model_latency_ms", "decision_latency_ms", "counter_offer_json",
                  "features_json"],
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, started_at TEXT, ended_at TEXT, channel TEXT,
  consent INTEGER, status TEXT, duration_s REAL, n_user_turns INTEGER);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts TEXT, role TEXT,
  step TEXT, text_masked TEXT, latency_ms REAL, llm_used INTEGER);
CREATE TABLE IF NOT EXISTS field_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts TEXT, field TEXT,
  value TEXT, status TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts TEXT, reference TEXT,
  age REAL, product TEXT, product_code_mc INTEGER, vehicle_price REAL, loan_amount REAL,
  ltv REAL, net_salary REAL, past_loans_active INTEGER, features_json TEXT,
  probability_decline REAL, threshold REAL, decision TEXT, model_version TEXT,
  model_latency_ms REAL, decision_latency_ms REAL, counter_offer_json TEXT);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class SheetsSink:
    """Background batch writer to the Apps Script web app. Never raises into the chat."""

    def __init__(self, url, token, batch_seconds=1.5):
        self.url, self.token, self.batch_seconds = url, token, batch_seconds
        self.q = queue.Queue()
        self.sent = self.failed = 0
        self.last_error = ""
        self._idle = threading.Event()
        self._idle.set()
        threading.Thread(target=self._run, daemon=True).start()
        atexit.register(self.flush)

    def add(self, table, row):
        self._idle.clear()
        self.q.put({"table": table, "values": [_cell(row.get(c)) for c in COLUMNS[table]]})

    def _run(self):
        while True:
            batch = [self.q.get()]
            time.sleep(self.batch_seconds)                 # collect a burst of rows into one request
            while not self.q.empty() and len(batch) < 200:
                batch.append(self.q.get_nowait())
            self._post(batch)
            if self.q.empty():
                self._idle.set()

    def _post(self, batch):
        payload = {"token": self.token, "headers": COLUMNS, "rows": batch}
        for attempt in range(3):
            try:
                r = requests.post(self.url, json=payload, timeout=20)
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                if r.ok and body.get("ok"):
                    self.sent += len(batch)
                    return
                self.last_error = body.get("error") or f"HTTP {r.status_code}"
            except Exception as e:
                self.last_error = type(e).__name__
            time.sleep(1 + attempt)
        self.failed += len(batch)

    def flush(self, timeout=30):
        """Wait until queued rows are sent (used by tests and at exit)."""
        self._idle.wait(timeout)


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    if isinstance(v, bool):
        return int(v)
    return v


_SHEETS = None


def sheets_sink():
    """One shared sink per process (Streamlit reruns create many Repository objects)."""
    global _SHEETS
    url = os.environ.get("SHEETS_WEBHOOK_URL", "").strip()
    if url and _SHEETS is None:
        _SHEETS = SheetsSink(url, os.environ.get("SHEETS_TOKEN", ""))
    return _SHEETS


class Repository:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(_SCHEMA)
        self.sheets = sheets_sink()

    def _exec(self, sql, args):
        with _lock:
            self.conn.execute(sql, args)
            self.conn.commit()

    def _sheet(self, table, row):
        if self.sheets:
            self.sheets.add(table, row)

    def start_session(self, session_id, channel):
        ts = _now()
        self._exec("INSERT INTO sessions(session_id, started_at, channel, status) VALUES (?,?,?,?)",
                   (session_id, ts, channel, "in_progress"))
        self._sheet("sessions", {"ts": ts, "session_id": session_id, "event": "started",
                                 "channel": channel, "status": "in_progress"})

    def update_session(self, session_id, **cols):
        sets = ", ".join(f"{k}=?" for k in cols)
        self._exec(f"UPDATE sessions SET {sets} WHERE session_id=?", (*cols.values(), session_id))
        event = "finished" if "status" in cols else "consent" if "consent" in cols else "update"
        self._sheet("sessions", dict(cols, ts=_now(), session_id=session_id, event=event))

    def log_message(self, session_id, role, step, text_masked, latency_ms=None, llm_used=False):
        row = {"ts": _now(), "session_id": session_id, "role": role, "step": step,
               "text_masked": text_masked, "latency_ms": latency_ms, "llm_used": int(llm_used)}
        self._exec("INSERT INTO messages(session_id, ts, role, step, text_masked, latency_ms, llm_used) "
                   "VALUES (?,?,?,?,?,?,?)", tuple(row[c] for c in
                                                   ("session_id", "ts", "role", "step", "text_masked",
                                                    "latency_ms", "llm_used")))
        self._sheet("messages", row)

    def log_field(self, session_id, field, value, status, note=""):
        row = {"ts": _now(), "session_id": session_id, "field": field, "value": json.dumps(value),
               "status": status, "note": note}
        self._exec("INSERT INTO field_events(session_id, ts, field, value, status, note) VALUES (?,?,?,?,?,?)",
                   (session_id, row["ts"], field, row["value"], status, note))
        self._sheet("field_events", row)

    def log_decision(self, session_id, reference, app, features, result, decision_latency_ms, offer):
        row = {"ts": _now(), "session_id": session_id, "reference": reference, "age": app["age"],
               "product": app["product"], "product_code_mc": features["Product_Code_MC"],
               "vehicle_price": app["vehicle_price"], "loan_amount": app["loan_amount"], "ltv": features["LTV"],
               "net_salary": app["net_salary"], "past_loans_active": features["PAST_LOANS_ACTIVE"],
               "probability_decline": result["probability_decline"], "threshold": result["threshold"],
               "decision": result["decision"], "model_version": result["model_version"],
               "model_latency_ms": result["latency_ms"], "decision_latency_ms": decision_latency_ms,
               "counter_offer_json": json.dumps(offer), "features_json": json.dumps(features)}
        cols = ["session_id", "ts", "reference", "age", "product", "product_code_mc", "vehicle_price",
                "loan_amount", "ltv", "net_salary", "past_loans_active", "features_json",
                "probability_decline", "threshold", "decision", "model_version", "model_latency_ms",
                "decision_latency_ms", "counter_offer_json"]
        self._exec(f"INSERT INTO decisions({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                   tuple(row[c] for c in cols))
        self._sheet("decisions", row)
