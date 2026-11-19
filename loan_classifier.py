"""
Loan classifier service: wraps the frozen XGBoost model (threshold 0.50).

classify(features) -> {"probability_decline", "decision", "threshold", "model_version", "latency_ms"}
Model is loaded once at import, so a call takes a few milliseconds.
"""
import json, os, time
import numpy as np, pandas as pd, xgboost as xgb

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")   # shipped with the app
with open(os.path.join(MODELS, "loan_model_config.json")) as fh:
    CONFIG = json.load(fh)
FEATURES = CONFIG["features"]
THRESHOLD = float(CONFIG["threshold"])
MODEL_VERSION = CONFIG["model_version"]
_model = xgb.XGBClassifier()
_model.load_model(os.path.join(MODELS, CONFIG["model_file"]))


def _frame(rows):
    return pd.DataFrame(rows, columns=FEATURES).astype("float32")


def classify(features: dict) -> dict:
    """features: dict with every key in FEATURES (numeric / 0-1)."""
    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    t0 = time.perf_counter()
    p = float(_model.predict_proba(_frame([[features[f] for f in FEATURES]]))[0, 1])
    return {"probability_decline": round(p, 4),
            "decision": "DECLINED" if p >= THRESHOLD else "APPROVED",
            "threshold": THRESHOLD, "model_version": MODEL_VERSION,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


def counter_offer(features: dict, vehicle_price: float, min_ltv: float = 40.0, step: float = 2.5):
    """Largest loan (lower LTV, everything else unchanged) the model would approve, or None."""
    ltvs = np.arange(features["LTV"] - step, min_ltv - 1e-9, -step)
    if len(ltvs) == 0:
        return None
    rows = []
    for ltv in ltvs:
        f = dict(features, LTV=float(ltv), Loan_Amount=float(vehicle_price * ltv / 100))
        rows.append([f[k] for k in FEATURES])
    probs = _model.predict_proba(_frame(rows))[:, 1]
    for ltv, p in zip(ltvs, probs):
        if p < THRESHOLD:
            loan = int(round(vehicle_price * ltv / 100, -2))
            return {"loan_amount": loan, "down_payment": int(vehicle_price - loan),
                    "ltv": round(float(ltv), 1), "probability_decline": round(float(p), 4)}
    return None


if __name__ == "__main__":
    demo = {"LTV": 85.0, "PAST_LOANS_ACTIVE": 1, "Age": 30, "Product_Code_MC": 1,
            "Loan_Amount": 95000, "Net_salary": 30000}
    print(classify(demo))
    print(counter_offer(demo, 95000 / 0.85))
