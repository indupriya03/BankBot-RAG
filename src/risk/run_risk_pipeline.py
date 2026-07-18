"""
run_risk_pipeline.py
─────────────────────
End-to-end risk pipeline runner.
Runs all stages in order:
    1. preprocessing_transactions.py
    2. train_model.py
    3. evaluate_model.py
    4. select_best_model.py (only when --model all; skipped for a single model)

Usage:
    python src/risk/run_risk_pipeline.py
    python src/risk/run_risk_pipeline.py --skip-preprocessing
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def run_stage(cmd: list, stage_name: str) -> None:
    """Run a pipeline stage as subprocess — fails fast if stage errors."""
    log.info("=" * 60)
    log.info(f"STAGE: {stage_name}")
    log.info("=" * 60)
    log.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        log.error(f"❌ Stage '{stage_name}' failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    log.info(f"✅ Stage '{stage_name}' complete\n")


def main():
    parser = argparse.ArgumentParser(description="BankBot-RAG Risk Pipeline")
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing (use existing data/preprocessed/ files)"
    )
    parser.add_argument(
        "--model",
        default="all",
        help="Model(s) to train/evaluate (default: all)"
    )
    args = parser.parse_args()

    python = sys.executable

    log.info("\n" + "=" * 60)
    log.info("  BankBot-RAG — Risk Pipeline")
    log.info("=" * 60 + "\n")

    # ── Stage 1: Preprocessing ────────────────────────────────────────────────
    if not args.skip_preprocessing:
        run_stage(
            [python, "src/data/preprocessing_transactions.py"],
            "Preprocessing Transactions"
        )
    else:
        log.info("⏭️  Skipping preprocessing (--skip-preprocessing flag set)")

    # ── Stage 2: Training ─────────────────────────────────────────────────────
    run_stage(
        [python, "src/risk/train_model.py", "--model", args.model],
        f"Training Models ({args.model})"
    )

    # ── Stage 3: Evaluation ───────────────────────────────────────────────────
    run_stage(
        [python, "src/risk/evaluate_model.py", "--model", args.model],
        f"Evaluating Models ({args.model})"
    )
    # ── Stage 4: Selection / Promotion ───────────────────────────────────────
    if args.model == "all":
        run_stage(
            [python, "src/risk/select_best_model.py"],
            "Selecting Best Model"
        )
    else:
        log.info(f"⏭️  Skipping model selection (--model {args.model}, not 'all' — "
                 f"selection needs a comparison across models)")

    log.info("=" * 60)
    log.info("  ✅ Risk Pipeline Complete!")
    log.info("=" * 60)
    log.info("Outputs:")
    log.info("  models/fraud/          ← trained models")
    log.info("  reports/fraud_eval/    ← evaluation reports + plots")
    log.info("  mlflow runs            ← logged to DagsHub")
    log.info("Next: python src/rag/build_vector_store.py")

if __name__ == "__main__":
    main()