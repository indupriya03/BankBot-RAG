"""
train_model.py
────────────────
STAGE: Model Training (stage 2 of 4 in the fraud/risk pipeline)

    feature_engineering → MODEL_TRAINING → evaluation → inference
    (preprocessing_        (this file)     (evaluate_    (transaction_
     transactions.py)                       model.py)     risk_lookup.py)

Fits model(s) on the SMOTE-balanced training data and saves them —
nothing else. Evaluation is deliberately a separate script/stage: fitting
a model is the expensive, "shouldn't rerun casually" part; scoring it
against test data is cheap and something you'll want to rerun often
(different thresholds, comparing runs) without re-fitting.

Usage:
    python train_model.py --model xgboost
    python train_model.py --model all          # trains all 4, saves each separately
"""

import argparse
import logging
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def _load_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)

_params = _load_params()

ALL_MODELS = ["random_forest", "logistic_regression", "xgboost", "lightgbm"]


def get_model(model_name: str, random_state: int = None):
    if random_state is None:
        random_state = _params["fraud_classifier"]["random_state"]
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )
    elif model_name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state,
        )
    elif model_name == "xgboost":
        # scale_pos_weight=1 since training data is already SMOTE-balanced (50/50).
        return XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=1, eval_metric="logloss",
            random_state=random_state, n_jobs=-1,
        )
    elif model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=1, random_state=random_state, n_jobs=-1, verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_training_data(preprocessed_dir: Path, feature_cols: list):
    train_df = pd.read_csv(preprocessed_dir / "transactions_train_smote.csv")
    X_train, y_train = train_df[feature_cols], train_df["fraud_label"]
    log.info(f"Train (SMOTE-balanced): {len(X_train)} rows, fraud rate: {y_train.mean():.1%}")
    return X_train, y_train


def train_one_model(model_name: str, X_train, y_train, output_dir: Path) -> Path:
    log.info(f"Training {model_name}...")
    model = get_model(model_name)
    model.fit(X_train, y_train)

    model_path = output_dir / f"{model_name}_fraud_model.pkl"
    joblib.dump(model, model_path)
    log.info(f"✅ {model_name} trained and saved → {model_path}")
    return model_path


def main():
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-dir", default="data/preprocessed")
    parser.add_argument("--output-dir", default="models/fraud")
    parser.add_argument("--model", choices=ALL_MODELS + ["all"], default="xgboost")
    parser.add_argument("--mlflow-experiment", default="bankbot-fraud-risk-training")
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(preprocessed_dir / "transaction_feature_cols.json") as f:
        feature_cols = json.load(f)

    X_train, y_train = load_training_data(preprocessed_dir, feature_cols)

    # ── DagsHub MLflow tracking ─────────────────────────────────────────
    user  = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN", "")

    if not user or not token:
        raise RuntimeError(
            "DAGSHUB_USERNAME and DAGSHUB_TOKEN must be set in .env for MLflow tracking."
        )

    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    # ─────────────────────────────────────────────────────────────────────

    mlflow.set_experiment(args.mlflow_experiment)
    model_names = ALL_MODELS if args.model == "all" else [args.model]

    for model_name in model_names:
        with mlflow.start_run(run_name=f"train-{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("train_fraud_rate", float(y_train.mean()))

            model_path = train_one_model(model_name, X_train, y_train, output_dir)
            mlflow.log_artifact(str(model_path))

    log.info(f"✅ Training stage complete. Models saved to {output_dir}")
    log.info("Next: python evaluate_model.py --model " + args.model)


if __name__ == "__main__":
    main()