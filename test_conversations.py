"""
Scripted end-to-end conversations: checks the flow, decisions, PII masking, logging and latency.

Usage: python test_conversations.py [--offline]
"""
import os, sqlite3, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dotenv import load_dotenv
load_dotenv(os.path.join(HERE, ".env"))
from chatbot_engine import LoanChatbot
from loan_repository import DB_PATH

sys.stdout.reconfigure(encoding="utf-8")
USE_LLM = "--offline" not in sys.argv

SCENARIOS = {
    "low LTV, several facts in one message (expect APPROVED)": [
        "yes", "I'm 42 and want a TVS Jupiter scooter, on-road price is 1.1 lakh",
        "I'll pay 45k down", "take home is 52,000 a month", "no, no other EMIs", "yes"],
    "high LTV motorcycle (expect DECLINED + counter-offer)": [
        "yes", "23", "Apache bike", "1,45,000", "I need 1.4 lakh loan", "18000",
        "yes I have a running EMI", "yes", "yes"],
    "yearly income check + correction at summary": [
        "ok", "35", "electric", "120000", "loan 90000", "6 lakh", "yearly", "no",
        "actually my age is 38", "yes"],
    "no caps: Rs 5 lakh bike, Rs 4.5 lakh loan (old limits rejected both)": [
        "yes", "45", "motorcycle, the Ronin, on-road 5 lakh", "loan of 4.5 lakh",
        "1.5 lakh per month", "no", "yes"],
    "customer volunteers phone number (must be masked)": [
        "yes", "29, call me on 9876543210 if needed", "scooter", "95000",
        "down payment 25000", "32000", "no", "yes"],
    "under 18 (expect ineligible)": ["yes", "I am 16"],
    "natural language + customer question (LLM path)": [
        "sure go ahead", "thirty one", "what does on-road price mean?",
        "a moped, around seventy two thousand all inclusive", "can put down about twenty thousand rupees",
        "I earn roughly 28 thousand after tax every month", "nope, nothing running", "yes"],
}

for name, turns in SCENARIOS.items():
    print("=" * 100, f"\nSCENARIO: {name}\n")
    bot = LoanChatbot(channel="test", use_llm=USE_LLM)
    t_start = time.perf_counter()
    print("BOT :", bot.start())
    for t in turns:
        if bot.done:
            break
        print("USER:", t)
        t0 = time.perf_counter()
        reply = bot.handle(t)
        print(f"BOT : {reply}   [{(time.perf_counter() - t0) * 1000:.0f} ms]")
    for d in bot.decisions:
        print(f"\n>> decision {d['decision']} | P(decline) {d['probability_decline']} | model {d['latency_ms']} ms "
              f"| final input -> decision {d['decision_latency_ms']} ms | features {d['features']}")
    print(f">> machine time for whole conversation: {time.perf_counter() - t_start:.1f} s, status done={bot.done}")

# Repository check: a volunteered phone number must never be stored
con = sqlite3.connect(DB_PATH)
dump = "\n".join(str(r) for tbl in ("sessions", "messages", "field_events", "decisions")
                 for r in con.execute(f"SELECT * FROM {tbl}"))
print("\n" + "=" * 100)
print("PII leak check in DB:", "FAILED" if "9876543210" in dump else
      f"PASSED (phone number stored only as [MOBILE]: {dump.count('[MOBILE]')} occurrence(s))")
print("LLM failures logged:", con.execute("SELECT COUNT(*) FROM messages WHERE role='system'").fetchone()[0])
for tbl in ("sessions", "messages", "field_events", "decisions"):
    print(f"  {tbl:<13} rows: {con.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]}")
print("\nSessions:")
for r in con.execute("SELECT session_id, status, duration_s, n_user_turns FROM sessions ORDER BY started_at"):
    print(" ", r)
