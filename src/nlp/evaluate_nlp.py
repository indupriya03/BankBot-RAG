"""
evaluate_nlp.py
───────────────
Evaluation harness for the BankBot-RAG NLP layer, with MLflow tracking.

- Intent: ground truth is the `category` column (Fraud/Loan/KYC/Account Access).
  Scored normally — real, consistent signal (see check_sentiment_consistency.py
  for the comparison: category is 0% inconsistent across duplicate query texts).

- Sentiment: quantitative scoring is SKIPPED. The `sentiment` column is
  53% inconsistent across duplicate query texts (same text carries up to
  5 different labels out of 6 possible) — scoring a classifier against
  labels that contradict themselves for identical input would measure
  label noise, not model quality, even with a caveat attached. Sentiment
  is still run at inference time (3-class: Urgent/Negative/Neutral) for
  the live pipeline — see sentiment_analyzer.py — just not benchmarked
  here against ground truth that doesn't reliably reflect the text.

Runs against data/preprocessed/tickets_eval_unique.csv, which
preprocessing.py already deduplicates.

Usage:
    python evaluate_nlp.py --eval-file data/preprocessed/tickets_eval_unique.csv
"""

import argparse
import json
import logging
from pathlib import Path
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from intent_classifier import classify_intent, INTENT_LABELS

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# If your `category` column in the raw data doesn't already match the
# INTENT_MAP values exactly ("Fraud/Unauthorized", "Loan", "KYC",
# "Account Access"), fill in the mapping here. Left as identity by default.
CATEGORY_NORMALIZATION = {
    # "raw_category_value": "Fraud/Unauthorized",
}


def normalize_category(cat: str) -> str:
    return CATEGORY_NORMALIZATION.get(cat, cat)



def evaluate_intent(eval_df: pd.DataFrame, text_col: str = "query_text_clean",
                     label_col: str = "category") -> dict:
    log.info("Running intent classifier on %d eval queries...", len(eval_df))

    y_true, y_pred, confidences = [], [], []
    for _, row in eval_df.iterrows():
        true_label = normalize_category(row[label_col])
        result = classify_intent(row[text_col])
        y_true.append(true_label)
        y_pred.append(result["intent"])
        confidences.append(result["confidence"])

    labels = sorted(set(y_true) | set(y_pred))

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report   = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=labels,
                                         zero_division=0, output_dict=True)
    cm       = confusion_matrix(y_true, y_pred, labels=labels)

    log.info("Intent — accuracy: %.4f | macro-F1: %.4f | weighted-F1: %.4f",
              acc, macro_f1, weighted_f1)
    log.info("\n%s", report)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "avg_confidence": sum(confidences) / len(confidences),
        "report_text": report,
        "report_dict": report_dict,
        "confusion_matrix": cm,
        "labels": labels,
        "n_samples": len(eval_df),
    }


def plot_confusion_matrix(cm, labels, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/preprocessed/tickets_eval_unique.csv")
    parser.add_argument("--text-col", default="query_text_clean")
    parser.add_argument("--label-col", default="category")
    parser.add_argument("--sentiment-label-col", default="sentiment")
    parser.add_argument("--output-dir", default="reports/nlp_eval")
    parser.add_argument("--mlflow-experiment", default="bankbot-nlp-layer")
    parser.add_argument("--intent-model", default="facebook/bart-large-mnli")
    parser.add_argument("--sentiment-model", default="facebook/bart-large-mnli")
    args = parser.parse_args()


    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(args.eval_file)
    log.info("Loaded %d rows from %s", len(eval_df), args.eval_file)

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

    with mlflow.start_run(run_name="intent+sentiment-zero-shot-eval"):
        # ── Params ──────────────────────────────────────────────────────────
        mlflow.log_param("intent_model", args.intent_model)
        mlflow.log_param("sentiment_model", args.sentiment_model)
        mlflow.log_param("intent_labels", json.dumps(INTENT_LABELS))
        mlflow.log_param("sentiment_output_classes", json.dumps(["Urgent", "Negative", "Neutral"]))
        mlflow.log_param("eval_file", args.eval_file)
        mlflow.log_param("n_eval_samples", len(eval_df))
        mlflow.log_param("approach", "zero-shot-nli")

        # ── Intent evaluation (has ground truth) ───────────────────────────
        intent_results = evaluate_intent(eval_df, args.text_col, args.label_col)

        mlflow.log_metric("intent_accuracy", intent_results["accuracy"])
        mlflow.log_metric("intent_macro_f1", intent_results["macro_f1"])
        mlflow.log_metric("intent_weighted_f1", intent_results["weighted_f1"])
        mlflow.log_metric("intent_avg_confidence", intent_results["avg_confidence"])

        for label, metrics in intent_results["report_dict"].items():
            if isinstance(metrics, dict):  # skip 'accuracy' scalar entry
                safe_label = label.replace("/", "_").replace(" ", "_")
                mlflow.log_metric(f"intent_f1_{safe_label}", metrics.get("f1-score", 0))
                mlflow.log_metric(f"intent_precision_{safe_label}", metrics.get("precision", 0))
                mlflow.log_metric(f"intent_recall_{safe_label}", metrics.get("recall", 0))

        cm_path = output_dir / "intent_confusion_matrix.png"
        plot_confusion_matrix(intent_results["confusion_matrix"], intent_results["labels"],
                              cm_path, "Intent Classifier — Confusion Matrix")
        mlflow.log_artifact(str(cm_path))

        report_path = output_dir / "intent_classification_report.txt"
        report_path.write_text(intent_results["report_text"])
        mlflow.log_artifact(str(report_path))

        # ── Sentiment evaluation skipped ─────────────────────────────────────
        # Consistency check revealed 53% of unique queries have conflicting
        # sentiment labels in the synthetic dataset — same query text appears
        # with up to 5 different sentiment labels (Angry/Anxious/Frustrated/
        # Neutral/Urgent). Category labels show 0% inconsistency.
        # Training or evaluating any classifier against these labels would
        # mean measuring noise, not signal.
        # Sentiment is handled by zero-shot classification (bart-large-mnli)
        # with 3 merged classes (Urgent/Negative/Neutral) at inference time —
        # not benchmarked here since ground truth doesn't reliably reflect text.
        mlflow.log_param("sentiment_evaluation", "skipped_random_labels")
        mlflow.log_param("sentiment_approach", "zero-shot-3-class-inference-only")
        mlflow.log_param("sentiment_finding", "53pct_label_inconsistency_detected")

        # ── Combined metrics.json for DVC ──────────────────────────────────
        metrics_out = {
            "intent": {
                "accuracy"      : intent_results["accuracy"],
                "macro_f1"      : intent_results["macro_f1"],
                "weighted_f1"   : intent_results["weighted_f1"],
                "avg_confidence": intent_results["avg_confidence"],
                "n_samples"     : intent_results["n_samples"],
            },
            "sentiment": {
                "note"    : "Evaluation skipped — 53% label inconsistency in synthetic data",
                "approach": "zero-shot bart-large-mnli, 3 classes at inference time",
            },
        }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_out, indent=2))
        mlflow.log_artifact(str(metrics_path))

        log.info("Evaluation complete. Reports written to %s", output_dir)
        log.info("MLflow run logged under experiment '%s'", args.mlflow_experiment)


if __name__ == "__main__":
    main()