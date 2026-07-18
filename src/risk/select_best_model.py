"""
select_best_model.py
──────────────────────
STAGE: Model Selection / Promotion (stage 4 of 4 in the fraud/risk pipeline)

    feature_engineering → model_training → evaluation → SELECTION
    (preprocessing_        (train_model.py) (evaluate_    (this file)
     transactions.py)                        model.py)

Reads reports/fraud_eval/model_comparison.csv (written by evaluate_model.py
--model all), picks the winner by a chosen metric (default: pr_auc — the
right metric here since fraud is rare and PR-AUC is far less misleading
than accuracy or even ROC-AUC on imbalanced data), and copies that model's
.pkl to a fixed, dynamic production path:

    models/fraud/production/production_fraud_model.pkl

alongside a production_model_metadata.json recording which model won, by
what metric/value, and when. transaction_risk_lookup.py already loads from
this exact path by default — nothing there is hardcoded to a specific
model name; whichever model wins THIS run is what gets served next time
the pipeline runs end-to-end.

Usage:
    python src/risk/select_best_model.py
    python src/risk/select_best_model.py --metric roc_auc
"""

import argparse
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

VALID_METRICS = ["pr_auc", "roc_auc", "f1", "precision", "recall"]


def tag_production_run(training_experiment: str, winner_model: str, metric: str, value: float) -> None:
    """
    Find the MLflow run that trained the winning model and tag it
    'production' on DagsHub — so anyone browsing the Experiments tab can
    see which run is actually serving traffic, without needing to know
    the local models/fraud/production/ folder even exists.

    Also un-tags any previously promoted run, so at most one run per
    experiment is ever marked production=true at a time.

    Wrapped in try/except by the caller — a DagsHub/network hiccup here
    should never break the local promotion, which has already succeeded
    by the time this runs.
    """
    user  = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN", "")

    if not user or not token:
        raise RuntimeError(
            "DAGSHUB_USERNAME and DAGSHUB_TOKEN must be set in .env for MLflow tracking."
        )

    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(training_experiment)
    if experiment is None:
        log.warning(f"⚠️  MLflow experiment '{training_experiment}' not found on DagsHub — "
                    f"skipping run tagging (local promotion is unaffected).")
        return

    # Un-tag whatever was previously marked production, so the tag never
    # points at two runs at once.
    previously_tagged = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.production = 'true'",
    )
    for run in previously_tagged:
        client.set_tag(run.info.run_id, "production", "false")
        log.info(f"   Un-tagged previous production run: {run.info.run_name} ({run.info.run_id})")

    # Find the run that trained the winning model (train_model.py names
    # runs 'train-{model_name}').
    matches = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = 'train-{winner_model}'",
        order_by=["start_time DESC"],
    )
    if not matches:
        log.warning(f"⚠️  No MLflow run named 'train-{winner_model}' found in "
                    f"'{training_experiment}' — production folder is still updated "
                    f"correctly, but no MLflow run was tagged.")
        return

    run = matches[0]
    client.set_tag(run.info.run_id, "production", "true")
    client.set_tag(run.info.run_id, "promoted_metric", metric)
    client.set_tag(run.info.run_id, "promoted_metric_value", str(round(value, 4)))
    log.info(f"✅ Tagged MLflow run {run.info.run_id} ('train-{winner_model}') as production")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-file", default="reports/fraud_eval/model_comparison.csv")
    parser.add_argument("--models-dir", default="models/fraud")
    parser.add_argument("--production-dir", default="models/fraud/production")
    parser.add_argument("--metric", choices=VALID_METRICS, default="pr_auc",
                         help="Metric to rank models by (default: pr_auc — the right "
                              "call given fraud is rare; accuracy/ROC-AUC alone can "
                              "hide a model that misses most fraud).")
    parser.add_argument("--training-experiment", default="bankbot-fraud-risk-training",
                         help="MLflow experiment name train_model.py logged runs under "
                              "(must match its --mlflow-experiment default/value).")
    parser.add_argument("--no-mlflow-tag", action="store_true",
                         help="Skip tagging the winning run on DagsHub (local promotion still happens).")
    args = parser.parse_args()

    comparison_path = Path(args.comparison_file)
    if not comparison_path.exists():
        log.error(
            f"❌ {comparison_path} not found. Run evaluate_model.py --model all first "
            f"(this file is only written when comparing multiple models)."
        )
        raise SystemExit(1)

    comparison = pd.read_csv(comparison_path)
    if args.metric not in comparison.columns:
        log.error(f"❌ Metric '{args.metric}' not found in {comparison_path}. "
                  f"Available columns: {list(comparison.columns)}")
        raise SystemExit(1)

    ranked = comparison.sort_values(args.metric, ascending=False).reset_index(drop=True)
    winner = ranked.iloc[0]
    winner_model = winner["model"]
    winner_value = float(winner[args.metric])

    log.info("=" * 60)
    log.info("MODEL SELECTION")
    log.info("=" * 60)
    log.info(f"Ranking by: {args.metric}")
    log.info("\n%s", ranked.to_string(index=False))
    log.info(f"\n🏆 Winner: {winner_model} ({args.metric}={winner_value:.4f})")

    models_dir = Path(args.models_dir)
    source_path = models_dir / f"{winner_model}_fraud_model.pkl"
    if not source_path.exists():
        log.error(f"❌ {source_path} not found — was it trained? Run train_model.py --model all.")
        raise SystemExit(1)

    production_dir = Path(args.production_dir)
    production_dir.mkdir(parents=True, exist_ok=True)
    production_path = production_dir / "production_fraud_model.pkl"

    shutil.copy2(source_path, production_path)
    log.info(f"✅ Copied {source_path} → {production_path}")

    metadata = {
        "promoted_model": winner_model,
        "selection_metric": args.metric,
        "selection_metric_value": winner_value,
        "all_candidates": ranked.to_dict("records"),
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "production_path": str(production_path),
    }
    metadata_path = production_dir / "production_model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    log.info(f"✅ Metadata written → {metadata_path}")

    if not args.no_mlflow_tag:
        try:
            tag_production_run(args.training_experiment, winner_model, args.metric, winner_value)
        except Exception as e:
            log.warning(f"⚠️  Could not tag production run on DagsHub ({e}). "
                        f"Local promotion succeeded regardless — this only affects "
                        f"visibility in the MLflow UI.")

    log.info("=" * 60)
    log.info("✅ Selection stage complete.")
    log.info(f"   transaction_risk_lookup.py will now load: {winner_model}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()