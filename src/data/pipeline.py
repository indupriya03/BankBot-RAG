"""
pipeline.py
────────────────
End-to-end data pipeline for BankBot-RAG.

Orchestrates:
  Step 1 → data_cleaning.py   : raw → cleaned
  Step 2 → preprocessing.py   : cleaned → preprocessed

Usage:
  python src/data/pipeline.py              # full pipeline
  python src/data/pipeline.py --clean-only # cleaning step only
  python src/data/pipeline.py --prep-only  # preprocessing step only

After running:
  dvc add data/preprocessed
  git add data/preprocessed.dvc
  git commit -m "pipeline: cleaned + preprocessed data"
  git push && dvc push
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Make src importable regardless of working directory ───────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.data_cleaning import (
    clean_tickets,
    clean_transactions,
    clean_qa_pairs,
    clean_policy_docs,
)
from src.data.preprocessing import (
    preprocess_tickets,
    preprocess_transactions,
    preprocess_qa_pairs,
    preprocess_policy_docs,
    build_rag_master_index,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR        = Path("data/raw")
CLEANED_DIR    = Path("data/cleaned")
PREPROCESSED_DIR = Path("data/preprocessed")


# ── Step 1: Cleaning ──────────────────────────────────────────────────────────

def run_cleaning() -> None:
    """Run all cleaning steps: tickets, transactions, QA pairs, policy docs."""
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║         STEP 1 — DATA CLEANING                          ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    clean_tickets(
        path       = str(RAW_DIR / "01_support_tickets.csv"),
        output_dir = CLEANED_DIR,
    )
    clean_transactions(
        path       = str(RAW_DIR / "02_transactions.csv"),
        output_dir = CLEANED_DIR,
    )
    clean_qa_pairs(
        path       = str(RAW_DIR / "04_qa_pairs.json"),
        output_dir = CLEANED_DIR,
    )
    clean_policy_docs(
        policy_dir = str(RAW_DIR / "policy_docs"),
        output_dir = CLEANED_DIR,
    )

    elapsed = time.time() - t0
    log.info(f"\n✅ STEP 1 COMPLETE — Cleaning finished in {elapsed:.1f}s")
    log.info(f"   Output → {CLEANED_DIR}/\n")


# ── Step 2: Preprocessing ─────────────────────────────────────────────────────

def run_preprocessing() -> None:
    """Run all preprocessing steps: encode, split, SMOTE, chunk, index."""
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║         STEP 2 — PREPROCESSING                          ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Verify cleaned data exists before proceeding
    required = [
        CLEANED_DIR / "01_support_tickets_cleaned.csv",
        CLEANED_DIR / "02_transactions_cleaned.csv",
        CLEANED_DIR / "04_qa_pairs_cleaned.json",
        CLEANED_DIR / "policy_docs",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        log.error(
            "❌ Cleaned data not found. Run cleaning step first.\n"
            f"   Missing: {missing}"
        )
        sys.exit(1)

    t0 = time.time()

    preprocess_tickets(CLEANED_DIR, PREPROCESSED_DIR)
    preprocess_transactions(CLEANED_DIR, PREPROCESSED_DIR)
    qa_df     = preprocess_qa_pairs(CLEANED_DIR, PREPROCESSED_DIR)
    policy_df = preprocess_policy_docs(CLEANED_DIR, PREPROCESSED_DIR)
    build_rag_master_index(qa_df, policy_df, PREPROCESSED_DIR)

    elapsed = time.time() - t0
    log.info(f"\n✅ STEP 2 COMPLETE — Preprocessing finished in {elapsed:.1f}s")
    log.info(f"   Output → {PREPROCESSED_DIR}/\n")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    """Print all output files with sizes."""
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║         PIPELINE OUTPUT SUMMARY                         ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    expected_outputs = [
        "tickets_train.csv",
        "tickets_val.csv",
        "tickets_test.csv",
        "transactions_train_smote.csv",
        "transactions_test.csv",
        "support_tickets_clean.csv",
        "qa_chunks.json",
        "policy_chunks.json",
        "rag_master_chunks.json",
        "label_encoder_category.pkl",
        "scaler_transactions.pkl",
        "transaction_feature_cols.json",
    ]

    all_ok = True
    for fname in expected_outputs:
        fpath = PREPROCESSED_DIR / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            log.info(f"  ✅  {fname:<45} {size_kb:>8.1f} KB")
        else:
            log.warning(f"  ❌  {fname:<45} NOT FOUND")
            all_ok = False

    log.info("")
    if all_ok:
        log.info("All outputs verified ✅")
        log.info("")
        log.info("Next steps — version your preprocessed data with DVC:")
        log.info("  dvc add data/preprocessed")
        log.info("  git add data/preprocessed.dvc .gitignore")
        log.info("  git commit -m 'pipeline: add preprocessed data'")
        log.info("  git push origin main")
        log.info("  dvc push")
    else:
        log.warning("Some outputs missing — check logs above for errors.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BankBot-RAG data pipeline: cleaning + preprocessing"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--clean-only",
        action="store_true",
        help="Run cleaning step only (data/raw → data/cleaned)"
    )
    group.add_argument(
        "--prep-only",
        action="store_true",
        help="Run preprocessing only (data/cleaned → data/preprocessed)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("")
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║   BankBot-RAG · Data Pipeline                           ║")
    log.info("║   raw → cleaned → preprocessed                         ║")
    log.info("╚══════════════════════════════════════════════════════════╝")

    total_start = time.time()

    if args.clean_only:
        run_cleaning()
    elif args.prep_only:
        run_preprocessing()
        print_summary()
    else:
        # Full pipeline
        run_cleaning()
        run_preprocessing()
        print_summary()

    total_elapsed = time.time() - total_start
    log.info(f"Total pipeline time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()