"""
Freeze the 6-feature XGBoost model with the 0.50 decision threshold for the chatbot.

Training-time script: needs the parent project (training data + trained model), not part of the
deployed app.
Reads  : ../models/xgb_onehot_selected.json (trained by loan_xgboost_onehot.py)
         ../PGAI_XGBoost_data.xlsx
Writes : models/loan_xgb_t050.json      (model shipped with the app)
         models/loan_model_config.json  (features, threshold, version, test metrics)
"""
import json, os, warnings
from datetime import date
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score,
                             precision_recall_fscore_support, accuracy_score)

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
THRESHOLD = 0.50

features = open(os.path.join(MODELS, "xgb_onehot_selected_features.txt")).read().splitlines()[1:]
model = xgb.XGBClassifier()
model.load_model(os.path.join(MODELS, "xgb_onehot_selected.json"))

# Same split as training -> metrics on the untouched test set
df = pd.read_excel(os.path.join(ROOT, "PGAI_XGBoost_data.xlsx"), sheet_name="One-Hot Encoded")
y = df.pop("Decision_DECLINED").astype(int).to_numpy()
X = df.astype("float32")
_, X_tmp, _, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
_, X_te, _, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
p = model.predict_proba(X_te[features])[:, 1]
pred = (p >= THRESHOLD).astype(int)
pr, rc, _, _ = precision_recall_fscore_support(y_te, pred, labels=[0, 1], zero_division=0)
metrics = {"accuracy": accuracy_score(y_te, pred), "roc_auc": roc_auc_score(y_te, p),
           "pr_auc": average_precision_score(y_te, p),
           "precision_approved": pr[0], "recall_approved": rc[0],
           "precision_declined": pr[1], "recall_declined": rc[1],
           "confusion_matrix": confusion_matrix(y_te, pred, labels=[0, 1]).tolist(), "n_test": int(len(y_te))}

APP_MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(APP_MODELS, exist_ok=True)
model.save_model(os.path.join(APP_MODELS, "loan_xgb_t050.json"))
config = {"model_file": "loan_xgb_t050.json", "model_version": "loan_xgb_v1_t050",
          "created": str(date.today()), "threshold": THRESHOLD, "positive_class": "DECLINED",
          "features": features,
          "valid_ranges": {"Age": [18, 69], "LTV": [0.01, 100], "Loan_Amount": [10000, 300000],
                           "Net_salary": [0, 1000000]},
          "test_metrics": {k: (round(float(v), 4) if not isinstance(v, list) else v) for k, v in metrics.items()}}
with open(os.path.join(APP_MODELS, "loan_model_config.json"), "w") as fh:
    json.dump(config, fh, indent=2)
print(json.dumps(config, indent=2))
