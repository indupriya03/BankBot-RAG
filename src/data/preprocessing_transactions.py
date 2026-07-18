"""
preprocessing_transactions.py
──────────────────────────────
Preprocessing for the fraud classification model.
Reads data/cleaned/02_transactions_cleaned.csv, saves to data/preprocessed/.

engineer_features() and encode_categoricals() below are the SINGLE SOURCE
OF TRUTH for the fraud model's feature schema — both this file's training
path (preprocess_transactions) AND transaction_risk_lookup.py's single-row
inference path import and call these same functions. This matters: if
they were separately implemented (as they briefly were), a change to one
without the other would silently cause the model to see different
features at training time vs. serving time — training/serving skew, a
classic and nasty-to-debug class of ML bug. One implementation, two
callers, no drift possible.

DATA LEAKAGE NOTE (Synthetic Dataset):
  After EDA and consistency checks, the following columns were found to be
  leaky or artifacts of synthetic data generation and are excluded:
  - geo_anomaly_flag, is_international: 100% correlated with fraud_label
  - velocity_flag, high_amount_flag: partially correlated with fraud_label
  - flag_count: derived from above flags → also leaky
  - merchant_category, city, transaction_type: perfect category separation
    (e.g. Crypto/Dubai/ATM = always fraud in synthetic data)
  Remaining clean features: amount_log, hour_of_day, month,
  is_night_transaction, is_weekend — purely numeric, no leakage.

Outputs:
  transactions_train_smote.csv   Fraud model train split (SMOTE applied)
  transactions_test.csv          Fraud model test split (real distribution)
  scaler_transactions.pkl        StandardScaler for transaction features
  transaction_feature_cols.json  Feature column names for inference
"""

import json
import logging
from pathlib import Path
from typing import Optional
import yaml
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

def _load_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)

_params = _load_params()
RANDOM_STATE = _params["data"]["transactions_random_state"]
TEST_SIZE    = _params["data"]["transactions_test_size"]


FLAG_COLS  = ["is_international", "velocity_flag", "geo_anomaly_flag", "high_amount_flag"]

# CAT_COLS empty — merchant_category, city, transaction_type dropped
# due to perfect fraud separation in synthetic data (data artifact, not signal)
CAT_COLS   = []

# Only clean numeric features remain after leakage removal
SCALE_COLS = ["amount_log", "hour_of_day", "month"]

DROP_COLS  = [
    "transaction_id", "account_id", "timestamp",
    "merchant_name",        # too granular — high cardinality
    "amount_inr",           # replaced by amount_log
    "fraud_reason",         # target-derived → severe data leakage
    "day_of_week",          # replaced by is_weekend
    # ── Synthetic data leakage — flags derived from fraud_label ──────────────
    "geo_anomaly_flag",     # 100% correlated with fraud_label
    "is_international",     # 100% correlated with fraud_label
    "velocity_flag",        # partially correlated with fraud_label
    "high_amount_flag",     # partially correlated with fraud_label
    "flag_count",           # derived from above leaky flags
    # ── Synthetic data artifact — perfect categorical separation ─────────────
    "merchant_category",    # Crypto/Digital/Unknown = always fraud
    "city",                 # Dubai/London/Singapore = always fraud
    "transaction_type",     # ATM/Card-Not-Present/NEFT = always fraud
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric feature engineering — works identically whether df has 1 row
    or 100,000. Called at training time (on the full dataset) and at
    inference time (on a single looked-up transaction) — see
    transaction_risk_lookup.py.

    Clean features produced (no leakage):
      amount_log           — log1p(amount_inr), reduces right skew
      month                — seasonality signal
      is_night_transaction — off-hours flag (22:00–05:00)
      is_weekend           — weekend transaction flag
    """
    df = df.copy()
    df["timestamp"]            = pd.to_datetime(df["timestamp"])
    df["month"]                = df["timestamp"].dt.month
    df["is_night_transaction"] = df["hour_of_day"].apply(
        lambda h: 1 if (h >= 22 or h <= 5) else 0
    )
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    df["amount_log"] = np.log1p(df["amount_inr"])

    return df


def encode_categoricals(df: pd.DataFrame, feature_cols: Optional[list] = None):
    """
    One-hot encodes CAT_COLS. Two modes:

    TRAINING mode (feature_cols=None): plain pd.get_dummies(), returns
    (encoded_df, resulting_column_list) — the caller saves that column
    list as transaction_feature_cols.json, which becomes the fixed schema
    inference must conform to.

    INFERENCE mode (feature_cols given): one-hot encodes, then aligns to
    the given schema exactly — any expected column not produced by this
    row's own categories is filled with 0 (that category just doesn't
    apply here, correctly 0). Any category this row has that never
    appeared in training produces a column with no home in feature_cols;
    it's dropped with a warning, since there's no trained weight for it
    to affect — silently losing that signal is the correct, safe behavior,
    not a bug to suppress.

    NOTE: CAT_COLS is currently empty (leaky categoricals dropped).
    This function is kept for future use when clean categorical features
    are available.
    """
    # Only encode columns that actually exist in df
    cols_to_encode = [c for c in CAT_COLS if c in df.columns]

    if cols_to_encode:
        encoded = pd.get_dummies(df, columns=cols_to_encode,
                                 prefix=cols_to_encode, drop_first=False)
        dummy_cols_created = [
            c for c in encoded.columns
            if any(c.startswith(f"{cc}_") for cc in cols_to_encode)
        ]
        encoded[dummy_cols_created] = encoded[dummy_cols_created].astype(int)
    else:
        encoded = df.copy()

    if feature_cols is None:
        return encoded, list(encoded.columns)

    # Inference mode — align to training schema
    for col in feature_cols:
        if col not in encoded.columns:
            encoded[col] = 0

    if cols_to_encode:
        dummy_cols = {
            c for c in encoded.columns
            if any(c.startswith(f"{cc}_") for cc in cols_to_encode)
        }
        unseen = dummy_cols - set(feature_cols)
        for col in unseen:
            log.warning(
                f"'{col}' never appeared in training data — dropping. "
                f"Consider retraining if this category becomes common."
            )

    return encoded.reindex(columns=feature_cols, fill_value=0)


def preprocess_transactions(cleaned_dir: Path, output_dir: Path) -> None:
    """
    Steps:
      1. Load cleaned transactions
      2. Feature engineering (time, amount) — engineer_features()
      3. Drop leaky/redundant columns
      4. encode_categoricals() — no-op currently (CAT_COLS empty)
      5. Stratified train/test split (ratio set by params.yaml: data.transactions_test_size)      
      6. Scale numeric features (fit on train only)
      7. Apply SMOTE to training set only
      8. Save splits + scaler + feature column names
    """
    log.info("=" * 60)
    log.info("Preprocessing Transactions")
    log.info("=" * 60)

    path = cleaned_dir / "02_transactions_cleaned.csv"
    txn  = pd.read_csv(path)
    log.info(f"Loaded {len(txn):,} rows | fraud rate: {txn['fraud_label'].mean()*100:.1f}%")

    # ── Feature engineering ───────────────────────────────────────────────────
    txn = engineer_features(txn)
    log.info("Feature engineering complete: month, is_night_transaction, is_weekend, amount_log ✅")

    # ── Drop leaky/redundant columns ──────────────────────────────────────────
    txn = txn.drop(columns=DROP_COLS, errors="ignore")
    log.info(f"Dropped {len(DROP_COLS)} leaky/redundant columns ✅")
    log.info(f"Remaining features: {[c for c in txn.columns if c != 'fraud_label']}")

    # ── Encode categoricals (no-op — CAT_COLS empty) ─────────────────────────
    txn, _ = encode_categoricals(txn)
    log.info(f"Shape after encoding: {txn.shape} ✅")

    # ── Train / Test split — BEFORE scaling and SMOTE ────────────────────────
    X = txn.drop(columns=["fraud_label"])
    y = txn["fraud_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)} (before SMOTE)")
    log.info(f"Train fraud: {y_train.sum()} | Test fraud: {y_test.sum()}")

    # ── Scale numeric features — fit on train ONLY ────────────────────────────
    scale_cols = [c for c in SCALE_COLS if c in X_train.columns]
    scaler     = StandardScaler()
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

    joblib.dump(scaler, output_dir / "scaler_transactions.pkl")
    log.info(f"Scaling applied to: {scale_cols} | scaler_transactions.pkl saved ✅")

    # ── SMOTE — training set ONLY ─────────────────────────────────────────────
    log.info(
        "\n⚠️  SMOTE NOTE: Applied to demonstrate MLOps pipeline. "
        "With only 5 clean numeric features remaining after leakage removal, "
        "SMOTE's benefit is limited on this synthetic dataset. "
        "In real-world data with richer features it provides genuine uplift.\n"
    )

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=_params["fraud_smote"]["k_neighbors"])
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    log.info(
        f"After SMOTE → Train: {len(X_train_sm)} | "
        f"Fraud: {y_train_sm.sum()} | Legit: {(y_train_sm==0).sum()}"
    )
    log.info(f"Test stays unchanged: {len(X_test)} rows (real-world distribution)")

    # ── Save splits ───────────────────────────────────────────────────────────
    train_df             = pd.DataFrame(X_train_sm, columns=X_train.columns)
    train_df["fraud_label"] = y_train_sm.values

    test_df              = pd.DataFrame(X_test.values, columns=X_test.columns)
    test_df["fraud_label"] = y_test.values

    train_df.to_csv(output_dir / "transactions_train_smote.csv", index=False)
    test_df.to_csv(output_dir  / "transactions_test.csv",        index=False)

    feature_cols = list(X_train.columns)
    with open(output_dir / "transaction_feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    log.info(f"transactions_train_smote.csv saved: {train_df.shape} ✅")
    log.info(f"transactions_test.csv saved: {test_df.shape} ✅")
    log.info(
        f"transaction_feature_cols.json saved: {len(feature_cols)} features → "
        f"{feature_cols} ✅\n"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    out = Path("data/preprocessed")
    out.mkdir(parents=True, exist_ok=True)
    preprocess_transactions(Path("data/cleaned"), out)