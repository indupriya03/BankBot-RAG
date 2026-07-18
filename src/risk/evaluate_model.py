"""
evaluate_model.py
────────────────────
STAGE: Evaluation (stage 3 of 4 in the fraud/risk pipeline)

    feature_engineering → model_training → EVALUATION → inference
    (preprocessing_        (train_model.py) (this file)  (transaction_
     transactions.py)                                     risk_lookup.py)

Loads an ALREADY-TRAINED model (from train_model.py's output) and scores
it against transactions_test.csv — the untouched, real-distribution split.
Never fits anything here; that's train_model.py's job. This separation
means you can re-run evaluation as many times as you want (different
threshold, comparing existing models) without ever re-fitting.

IMPORTANT: fraud is rare in the real world, so accuracy is a misleading
metric — a model predicting "not fraud" for everything could score 95%+
accuracy while catching zero fraud. This reports precision/recall/F1/
ROC-AUC/PR-AUC instead, and PR-AUC is the one to rank models by (see
plot_risk_tier_separation() for a further sanity check on top of that).

Usage:
    python evaluate_model.py --model xgboost
    python evaluate_model.py --model all   # evaluates every trained model, writes comparison table
"""

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

# fraud_label is binary, but the project spec wants 3 risk tiers
# (High -> Fraud, Medium -> Sensitive, Low -> General). The bridge is the
# model's predicted PROBABILITY, not its raw 0/1 output.
RISK_THRESHOLDS = {
    "high": _params["risk_thresholds"]["high"],
    "medium": _params["risk_thresholds"]["medium"],
}

def probability_to_risk_level(proba: float) -> str:
    if proba >= RISK_THRESHOLDS["high"]:
        return "High"
    elif proba >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


def plot_confusion_matrix(cm, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Not Fraud", "Fraud"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_cols: list, out_path: Path, top_n: int = 15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_cols[i] for i in idx][::-1], importances[idx][::-1], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return dict(zip([feature_cols[i] for i in idx], importances[idx].tolist()))


def plot_risk_tier_separation(y_test, y_proba, out_path: Path) -> dict:
    """Sanity check: do the risk tiers actually separate real fraud from
    non-fraud, or are the thresholds arbitrary? Shows the ACTUAL fraud
    rate within each predicted tier — High should be much higher than Low."""
    tiers = [probability_to_risk_level(p) for p in y_proba]
    df = pd.DataFrame({"tier": tiers, "actual_fraud": y_test.values})

    tier_order = ["Low", "Medium", "High"]
    summary = df.groupby("tier")["actual_fraud"].agg(["mean", "count"]).reindex(tier_order).fillna(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(tier_order, summary["mean"] * 100, color=["#55A868", "#DD8452", "#C44E52"])
    for i, (tier, row) in enumerate(summary.iterrows()):
        ax.text(i, row["mean"] * 100 + 1, f"n={int(row['count'])}", ha="center", fontsize=9)
    ax.set_ylabel("Actual fraud rate within tier (%)")
    ax.set_title("Risk Tier Validation: does High actually mean high-risk?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return summary.to_dict()


def evaluate_one_model(
    model_name: str,
    models_dir: Path,
    preprocessed_dir: Path,
    output_dir: Path,
    threshold: float,
) -> dict:
    model_path = models_dir / f"{model_name}_fraud_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run train_model.py --model {model_name} first."
        )
    model = joblib.load(model_path)

    with open(preprocessed_dir / "transaction_feature_cols.json") as f:
        feature_cols = json.load(f)

    test_df = pd.read_csv(preprocessed_dir / "transactions_test.csv")
    X_test, y_test = test_df[feature_cols], test_df["fraud_label"]
    log.info(f"Test (real distribution): {len(X_test)} rows, fraud rate: {y_test.mean():.1%}")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_proba)
    pr_auc    = average_precision_score(y_test, y_proba)
    report    = classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"], zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)

    log.info(f"{model_name} — precision: {precision:.4f} | recall: {recall:.4f} | "
             f"F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    log.info("\n%s", report)

    cm_path = output_dir / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, cm_path, f"Fraud Classifier ({model_name}) — Confusion Matrix")

    fi_path = output_dir / f"{model_name}_feature_importance.png"
    plot_feature_importance(model, feature_cols, fi_path)

    tier_path = output_dir / f"{model_name}_risk_tier_validation.png"
    tier_summary = plot_risk_tier_separation(y_test, y_proba, tier_path)

    report_path = output_dir / f"{model_name}_classification_report.txt"
    report_path.write_text(report)

    metrics_out = {
        "model": model_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n_test": len(X_test),
        "test_fraud_rate": float(y_test.mean()),
        "risk_tier_validation": tier_summary,
        "note": "Evaluated on transactions_test.csv (real, untouched distribution).",
    }
    metrics_path = output_dir / f"{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2, default=str))

    return {
        **metrics_out,
        "cm_path": cm_path, "fi_path": fi_path, "tier_path": tier_path,
        "report_path": report_path, "metrics_path": metrics_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models/fraud")
    parser.add_argument("--preprocessed-dir", default="data/preprocessed")
    parser.add_argument("--output-dir", default="reports/fraud_eval")
    parser.add_argument("--model", choices=ALL_MODELS + ["all"], default="xgboost")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Classification threshold on predicted fraud probability. "
                              "Lower it (e.g. 0.3) to catch more fraud at the cost of more "
                              "false positives — a common tradeoff banks tune deliberately.")
    parser.add_argument("--mlflow-experiment", default="bankbot-fraud-risk-eval")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    all_results = []
    for model_name in model_names:
        with mlflow.start_run(run_name=f"evaluate-{model_name}"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("threshold", args.threshold)

            result = evaluate_one_model(model_name, models_dir, preprocessed_dir, output_dir, args.threshold)

            mlflow.log_param("n_test", result["n_test"])
            mlflow.log_param("test_fraud_rate", result["test_fraud_rate"])
            mlflow.log_metric("precision", result["precision"])
            mlflow.log_metric("recall", result["recall"])
            mlflow.log_metric("f1", result["f1"])
            mlflow.log_metric("roc_auc", result["roc_auc"])
            mlflow.log_metric("pr_auc", result["pr_auc"])

            mlflow.log_artifact(str(result["cm_path"]))
            mlflow.log_artifact(str(result["fi_path"]))
            mlflow.log_artifact(str(result["tier_path"]))
            mlflow.log_artifact(str(result["report_path"]))
            mlflow.log_artifact(str(result["metrics_path"]))

            all_results.append(result)

    if len(all_results) > 1:
        comparison = pd.DataFrame(all_results)[
            ["model", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        ].sort_values("pr_auc", ascending=False)
        comparison_path = output_dir / "model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        log.info("\n" + "=" * 60)
        log.info("MODEL COMPARISON (sorted by PR-AUC — the right metric given fraud is rare)")
        log.info("\n%s", comparison.to_string(index=False))
        log.info(f"\nSaved to {comparison_path}")

    log.info(f"✅ Evaluation stage complete. Reports in {output_dir}")


if __name__ == "__main__":
    main()