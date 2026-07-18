"""
transaction_risk_lookup.py
────────────────────────────
Bridges a customer's chat message to an actual transaction risk score.

IMPORTANT DATA-SCHEMA NOTE: tickets use `customer_id` (CUST######),
transactions use `account_id` (ACC######) — these are UNRELATED ID
namespaces in this synthetic dataset, not foreign-keyed to each other.
There is no way to go from "who is chatting" to "their transactions"
automatically. The customer must provide the transaction_id (or account_id
to browse recent transactions) directly in the conversation.

Flow:
    1. extract_transaction_id() / extract_account_id() — pull an ID out of
       whatever the customer typed, if present
    2. lookup_transaction() — find that exact record in the transactions table
    3. engineer_single_transaction() — scores that ONE row using the same
       engineer_features()/encode_categoricals() functions imported from
       preprocessing_transactions.py — single source of truth shared with
       the training path, so the two can never silently drift apart
    4. score_transaction() — run the trained model, convert probability to
       a risk tier (see evaluate_model.py's probability_to_risk_level)

Usage:
    from transaction_risk_lookup import TransactionRiskScorer
    scorer = TransactionRiskScorer()
    result = scorer.score_from_message("My transaction TXN500004 looks wrong")
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional
import yaml
import joblib
import pandas as pd

# src/risk/ -> src/ -> project root. Same pattern as nlp_pipeline.py, needed
# because preprocessing_transactions.py lives in src/data/, a sibling folder,
# not the same folder as this file — flat same-folder imports don't reach it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocessing_transactions import engineer_features, encode_categoricals, SCALE_COLS

log = logging.getLogger(__name__)

def _load_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)

_params = _load_params()
# Single source of truth: params.yaml's risk_thresholds section.
# evaluate_model.py reads the same values, so the two can never drift apart.
RISK_THRESHOLDS = {
    "high": _params["risk_thresholds"]["high"],
    "medium": _params["risk_thresholds"]["medium"],
}
TXN_ID_PATTERN = re.compile(r"\bTXN\d+\b", re.IGNORECASE)
ACC_ID_PATTERN = re.compile(r"\bACC\d+\b", re.IGNORECASE)


def extract_transaction_id(text: str) -> Optional[str]:
    match = TXN_ID_PATTERN.search(text)
    return match.group(0).upper() if match else None


def extract_account_id(text: str) -> Optional[str]:
    match = ACC_ID_PATTERN.search(text)
    return match.group(0).upper() if match else None


def probability_to_risk_level(proba: float) -> str:
    if proba >= RISK_THRESHOLDS["high"]:
        return "High"
    elif proba >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


def engineer_single_transaction(row: pd.Series, feature_cols: list, scaler) -> pd.DataFrame:
    """
    Scores ONE transaction using the exact same engineer_features() and
    encode_categoricals() that preprocess_transactions() uses at training
    time — imported, not reimplemented, so the two can never drift apart.
    encode_categoricals(feature_cols=...) handles single-row one-hot
    alignment against the trained schema (see its docstring).
    """
    df = pd.DataFrame([row])
    df = engineer_features(df)
    df = encode_categoricals(df, feature_cols=feature_cols)
    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])
    return df


class TransactionRiskScorer:
    def __init__(
        self,
        cleaned_transactions_path: str = "data/cleaned/02_transactions_cleaned.csv",
        model_path: str = "models/fraud/production/production_fraud_model.pkl",
        scaler_path: str = "data/preprocessed/scaler_transactions.pkl",
        feature_cols_path: str = "data/preprocessed/transaction_feature_cols.json",
    ):
        log.info("Loading transaction lookup table and fraud model...")
        self.transactions = pd.read_csv(cleaned_transactions_path)
        self.transactions["transaction_id"] = self.transactions["transaction_id"].str.upper()
        self.transactions["account_id"] = self.transactions["account_id"].str.upper()

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run the pipeline in order: "
                f"train_model.py --model all -> evaluate_model.py --model all -> "
                f"select_best_model.py — the last step promotes whichever model wins "
                f"to this exact path, which is what this class always loads. "
                f"(Or pass model_path= explicitly to load a specific model instead.)"
            )
        self.model = joblib.load(model_file)

        metadata_path = model_file.parent / "production_model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            log.info(f"Production model: {meta['promoted_model']} "
                     f"(selected by {meta['selection_metric']}={meta['selection_metric_value']:.4f}, "
                     f"promoted {meta['promoted_at_utc']})")

        self.scaler = joblib.load(scaler_path)
        with open(feature_cols_path) as f:
            self.feature_cols = json.load(f)
        log.info(f"Loaded {len(self.transactions)} transactions, model={type(self.model).__name__}")

    def lookup_transaction(self, transaction_id: str) -> Optional[pd.Series]:
        match = self.transactions[self.transactions["transaction_id"] == transaction_id.upper()]
        return match.iloc[0] if len(match) > 0 else None

    def get_account_transactions(self, account_id: str, limit: int = 5) -> pd.DataFrame:
        matches = self.transactions[self.transactions["account_id"] == account_id.upper()]
        return matches.sort_values("timestamp", ascending=False).head(limit)

    def score_transaction_id(self, transaction_id: str) -> dict:
        row = self.lookup_transaction(transaction_id)
        if row is None:
            return {"found": False, "transaction_id": transaction_id}

        features = engineer_single_transaction(row, self.feature_cols, self.scaler)
        proba = float(self.model.predict_proba(features)[0, 1])
        risk_level = probability_to_risk_level(proba)

        return {
            "found": True,
            "transaction_id": transaction_id,
            "account_id": row["account_id"],
            "amount_inr": float(row["amount_inr"]),
            "merchant_name": row["merchant_name"],
            "timestamp": str(row["timestamp"]),
            "actual_fraud_label": int(row["fraud_label"]) if "fraud_label" in row else None,
            "fraud_probability": round(proba, 4),
            "risk_level": risk_level,
        }

    def score_from_message(self, message: str) -> dict:
        """Extracts a transaction ID from free text and scores it. If no
        transaction ID is found but an account ID is, returns recent
        transactions on that account instead so the customer can pick one."""
        txn_id = extract_transaction_id(message)
        if txn_id:
            return self.score_transaction_id(txn_id)

        acc_id = extract_account_id(message)
        if acc_id:
            recent = self.get_account_transactions(acc_id)
            if len(recent) == 0:
                return {"found": False, "account_id": acc_id, "reason": "no_transactions_for_account"}
            return {
                "found": "needs_selection",
                "account_id": acc_id,
                "recent_transactions": recent[["transaction_id", "timestamp", "amount_inr", "merchant_name"]].to_dict("records"),
            }

        return {"found": False, "reason": "no_id_in_message"}


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--message", required=True, help="Customer message text to extract an ID from and score")
    args = parser.parse_args()

    scorer = TransactionRiskScorer()
    result = scorer.score_from_message(args.message)
    print(json.dumps(result, indent=2, default=str))