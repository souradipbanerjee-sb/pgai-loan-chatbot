"""
Checks the project objectives against their benchmarks.

  b. Accuracy   - test-set metrics of the frozen model (models/loan_model_config.json)
  c. Real time  - final input -> decision latency, from the decisions log
  a. CX         - completion time and turns, from the sessions log (completed sessions only)
  d. Repository - every completed session has messages, field events and a decision row

Usage: python evaluate_benchmarks.py
"""
import json, os, sqlite3
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "models", "loan_model_config.json")))
m = cfg["test_metrics"]
BASE_RATE = 854 / 16259   # share of DECLINED in the test set

BENCH = [  # name, value, target, direction
    ("Recall - DECLINED (bad loans caught)", m["recall_declined"], 0.70, ">="),
    ("Precision - APPROVED (approved book quality)", m["precision_approved"], 0.97, ">="),
    ("Precision - DECLINED (>= 2x base rate)", m["precision_declined"], round(2 * BASE_RATE, 3), ">="),
    ("Recall - APPROVED (good applicants passed)", m["recall_approved"], 0.60, ">="),
    ("Overall accuracy", m["accuracy"], 0.60, ">="),
    ("ROC-AUC", m["roc_auc"], 0.75, ">="),
]

con = sqlite3.connect(os.path.join(HERE, "loan_chatbot.db"))
lat = np.array([r[0] for r in con.execute("SELECT decision_latency_ms FROM decisions")])
if len(lat):
    BENCH += [("Decision latency p95 (ms)", float(np.percentile(lat, 95)), 2000, "<="),
              ("Decision latency max (ms)", float(lat.max()), 2000, "<=")]
turn = np.array([r[0] for r in con.execute(
    "SELECT latency_ms FROM messages WHERE role='bot' AND latency_ms IS NOT NULL")])
if len(turn):
    BENCH.append(("Bot reply latency p95 per turn (ms)", float(np.percentile(turn, 95)), 5000, "<="))
done = con.execute("SELECT session_id, n_user_turns FROM sessions WHERE status IN ('approved','declined')").fetchall()
if done:
    # human time estimate: 15 s to read + type per user turn, plus the session's measured bot time
    est = np.array([t * 15 + (con.execute("SELECT COALESCE(SUM(latency_ms),0) FROM messages WHERE session_id=?",
                                          (sid,)).fetchone()[0] / 1000) for sid, t in done])
    BENCH.append(("Est. completion time median (s), 15 s/turn", float(np.median(est)), 180, "<="))
    BENCH.append(("Est. completion time p95 (s), incl. corrections", float(np.percentile(est, 95)), 240, "<="))
    complete = sum(1 for sid, _ in done if all(con.execute(f"SELECT COUNT(*) FROM {t} WHERE session_id=?", (sid,)).fetchone()[0]
                                               for t in ("messages", "field_events", "decisions")))
    BENCH.append(("Completed sessions fully logged (%)", 100 * complete / len(done), 100, ">="))

print(f"Model {cfg['model_version']} | threshold {cfg['threshold']} | test n={int(m['n_test'])}\n")
print(f"{'Objective':<48}{'Actual':>10}{'Target':>10}  Result")
for name, val, target, op in BENCH:
    ok = val >= target if op == ">=" else val <= target
    print(f"{name:<48}{val:>10.3f}{op + str(target):>10}  {'PASS' if ok else 'FAIL'}")
