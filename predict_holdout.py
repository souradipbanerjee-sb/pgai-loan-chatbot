"""
Score a raw hold-out file (the 17-column format of the project brief) with the frozen model.

Applies in code the same preparation that was done in Excel for the training data, then
scores every row. No row is dropped: the hold-out set needs a prediction for each application.

  1. Read CSV or Excel; column names are matched ignoring case, spaces and underscores.
  2. Identifier / high-cardinality / redundant columns are ignored
     (Pincode, Model_Description, Make_Code, Model_Variant, Qualifications, ... are not model inputs).
  3. Numbers: commas, currency symbols and '%' stripped; text and spreadsheet errors (#VALUE!) -> missing.
  4. Product_Code: upper-cased and trimmed. If blank or unknown, it is recovered from Make_Code or
     Model_Variant (each make belongs to exactly one product type); otherwise treated as non-motorcycle.
  5. PAST_LOANS_ACTIVE: PAST_LOANS_ACTIVE / yes / 1 -> 1; NO_PAST_LOANS / no / 0 -> 0.
  6. Remaining missing values are filled with training medians (mode for PAST_LOANS_ACTIVE).
  7. Score: P(declined) from the 6-feature XGBoost; DECLINED if P >= threshold (0.50).

Every fill, recovery or out-of-range value is recorded per row in the audit file.

Usage:
  python predict_holdout.py HOLDOUT.csv [--out predictions.csv] [--audit predictions_audit.csv]
If the file has a Decision column, accuracy / precision / recall / F1 are printed as well.
"""
import argparse, json, os, re, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loan_classifier as clf  # noqa: E402  (loads the frozen model + threshold)

PRE = json.load(open(os.path.join(HERE, "models", "preprocessing.json"), encoding="utf-8"))
FILL = PRE["fill_values"]
NUMERIC = ["Age", "Loan_Amount", "LTV", "Net_salary"]
WANTED = ["ID", "Age", "Product_Code", "Loan_Amount", "LTV", "Net_salary", "PAST_LOANS_ACTIVE",
          "Make_Code", "Model_Variant", "Model_Description", "Decision"]


def _key(name):
    return re.sub(r"[\s_\-]", "", str(name)).lower()


def read_any(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def standardise_columns(df):
    lookup = {_key(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for col in WANTED:
        src = lookup.get(_key(col))
        out[col] = df[src] if src is not None else ""
    missing = [c for c in ("Age", "Product_Code", "Loan_Amount", "LTV", "Net_salary", "PAST_LOANS_ACTIVE")
               if _key(c) not in lookup]
    return out.fillna(""), missing


def to_number(s):
    s = s.astype(str).str.strip()
    s = s.str.replace(r"[,\s₹%]|^rs\.?|inr", "", regex=True, flags=re.I)
    return pd.to_numeric(s, errors="coerce")       # '#VALUE!', '', 'abc' -> NaN


def product_code(row):
    code = str(row["Product_Code"]).strip().upper()
    if code in PRE["product_codes"]:
        return code, ""
    make = str(row["Make_Code"]).strip().upper()
    if make in PRE["make_to_product"]:
        return PRE["make_to_product"][make], f"Product_Code '{code}' recovered from Make_Code {make}"
    text = f"{row['Model_Variant']} {row['Model_Description']}".upper()
    for kw, prod in PRE["variant_keyword_to_product"].items():
        if kw in text:
            return prod, f"Product_Code '{code}' recovered from model name ({kw})"
    return "UNKNOWN", f"Product_Code '{code}' unknown -> scored as non-motorcycle"


def past_loans(v):
    t = str(v).strip().upper()
    if t in ("PAST_LOANS_ACTIVE", "YES", "Y", "1", "TRUE", "ACTIVE"):
        return 1
    if t in ("NO_PAST_LOANS", "NO", "N", "0", "FALSE", "NONE"):
        return 0
    return np.nan


def prepare(raw):
    """Raw 17-column frame -> (6-feature frame, per-row notes)."""
    df, missing_cols = standardise_columns(raw)
    notes = [[] for _ in range(len(df))]
    if missing_cols:
        print(f"WARNING: columns not found, filled with training defaults: {missing_cols}")

    feats = pd.DataFrame(index=df.index)
    for c in NUMERIC:
        feats[c] = to_number(df[c])
    feats["PAST_LOANS_ACTIVE"] = df["PAST_LOANS_ACTIVE"].map(past_loans)

    prod = df.apply(product_code, axis=1)
    feats["Product_Code_MC"] = [1 if p == "MC" else 0 for p, _ in prod]
    for i, (_, note) in enumerate(prod):
        if note:
            notes[i].append(note)

    for c in NUMERIC + ["PAST_LOANS_ACTIVE"]:
        bad = feats[c].isna()
        for i in np.flatnonzero(bad.to_numpy()):
            notes[i].append(f"{c} missing/invalid ('{df[c].iloc[i]}') -> filled {FILL[c]}")
        feats.loc[bad, c] = FILL[c]

    lo, hi = PRE["policy_age_range"]
    for i in np.flatnonzero(((feats["Age"] < lo) | (feats["Age"] > hi)).to_numpy()):
        notes[i].append(f"Age {feats['Age'].iloc[i]:g} outside policy {lo}-{hi}")
    for c, (a, b) in PRE["training_ranges"].items():
        for i in np.flatnonzero(((feats[c] < a) | (feats[c] > b)).to_numpy()):
            notes[i].append(f"{c} {feats[c].iloc[i]:g} outside training range {a}-{b}")
    return df, feats[clf.FEATURES].astype(float), notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--audit", default="predictions_audit.csv")
    a = ap.parse_args()

    raw = read_any(a.input)
    df, X, notes = prepare(raw)
    p = clf.predict_batch(X)
    decision = np.where(p >= clf.THRESHOLD, "DECLINED", "APPROVED")

    ids = df["ID"].where(df["ID"].astype(str).str.strip() != "", pd.Series(range(1, len(df) + 1), index=df.index).astype(str))
    pd.DataFrame({"ID": ids, "Decision": decision, "Probability_Declined": np.round(p, 6)}).to_csv(a.out, index=False)
    audit = X.copy()
    audit.insert(0, "ID", ids)
    audit["Probability_Declined"] = np.round(p, 6)
    audit["Decision"] = decision
    audit["Data_Notes"] = ["; ".join(n) for n in notes]
    audit.to_csv(a.audit, index=False)

    n_fixed = sum(1 for n in notes if n)
    print(f"Scored {len(df):,} rows with {clf.MODEL_VERSION} (threshold {clf.THRESHOLD})")
    print(f"  DECLINED: {(decision == 'DECLINED').sum():,} ({(decision == 'DECLINED').mean():.1%})   "
          f"APPROVED: {(decision == 'APPROVED').sum():,}")
    print(f"  rows with a data note (filled / recovered / out of range): {n_fixed:,}  -> see {a.audit}")
    print(f"  predictions -> {a.out}")

    truth = df["Decision"].astype(str).str.strip().str.upper()
    if truth.isin(["APPROVED", "DECLINED"]).any():
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_auc_score
        m = truth.isin(["APPROVED", "DECLINED"]).to_numpy()
        y = (truth[m] == "DECLINED").astype(int).to_numpy()
        yp = (decision[m] == "DECLINED").astype(int)
        pr, rc, f1, s = precision_recall_fscore_support(y, yp, labels=[0, 1], zero_division=0)
        print(f"\nLabelled rows: {m.sum():,}  accuracy {accuracy_score(y, yp):.3f}  ROC-AUC {roc_auc_score(y, p[m]):.3f}")
        for i, c in enumerate(["APPROVED", "DECLINED"]):
            print(f"  {c:<9} precision {pr[i]:.3f}  recall {rc[i]:.3f}  F1 {f1[i]:.3f}  n={s[i]}")
        print("  confusion matrix [actual x predicted, APPROVED/DECLINED]:", confusion_matrix(y, yp, labels=[0, 1]).tolist())


if __name__ == "__main__":
    main()
