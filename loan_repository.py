"""
Audit repository (SQLite): every message, captured feature value, model score and decision.

Tables
  sessions     - one row per conversation (no name / contact details are collected)
  messages     - every bot / user turn, PII-masked, with turn latency
  field_events - every feature value captured or corrected, with its source text
  decisions    - final feature vector, model score, threshold, decision, latencies
"""
import json, os, sqlite3, threading
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loan_chatbot.db")
_lock = threading.Lock()

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


class Repository:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(_SCHEMA)

    def _exec(self, sql, args):
        with _lock:
            self.conn.execute(sql, args)
            self.conn.commit()

    def start_session(self, session_id, channel):
        self._exec("INSERT INTO sessions(session_id, started_at, channel, status) VALUES (?,?,?,?)",
                   (session_id, _now(), channel, "in_progress"))

    def update_session(self, session_id, **cols):
        sets = ", ".join(f"{k}=?" for k in cols)
        self._exec(f"UPDATE sessions SET {sets} WHERE session_id=?", (*cols.values(), session_id))

    def log_message(self, session_id, role, step, text_masked, latency_ms=None, llm_used=False):
        self._exec("INSERT INTO messages(session_id, ts, role, step, text_masked, latency_ms, llm_used) "
                   "VALUES (?,?,?,?,?,?,?)", (session_id, _now(), role, step, text_masked, latency_ms, int(llm_used)))

    def log_field(self, session_id, field, value, status, note=""):
        self._exec("INSERT INTO field_events(session_id, ts, field, value, status, note) VALUES (?,?,?,?,?,?)",
                   (session_id, _now(), field, json.dumps(value), status, note))

    def log_decision(self, session_id, reference, app, features, result, decision_latency_ms, offer):
        self._exec(
            "INSERT INTO decisions(session_id, ts, reference, age, product, product_code_mc, vehicle_price, "
            "loan_amount, ltv, net_salary, past_loans_active, features_json, probability_decline, threshold, "
            "decision, model_version, model_latency_ms, decision_latency_ms, counter_offer_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, _now(), reference, app["age"], app["product"], features["Product_Code_MC"],
             app["vehicle_price"], app["loan_amount"], features["LTV"], app["net_salary"],
             features["PAST_LOANS_ACTIVE"], json.dumps(features), result["probability_decline"],
             result["threshold"], result["decision"], result["model_version"], result["latency_ms"],
             decision_latency_ms, json.dumps(offer)))
