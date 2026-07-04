"""
data_cleaning.py
────────────────
Cleaning pipeline for BankBot-RAG datasets.
Reads from data/raw/, validates, cleans, saves to data/cleaned/.
Called by pipeline.py — can also be run standalone for testing.
"""

import json
import logging
import os
import re
from pathlib import Path

import pandas as pd

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helper functions ──────────────────────────────────────────────────────────

def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string columns."""
    str_cols = [c for c in df.columns if df[c].dtype == object]
    for col in str_cols:
        df[col] = df[col].str.strip()
    return df


def _standardise_case(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Title-case specified columns, preserving known acronyms."""

    # Acronyms that .str.title() breaks — fix them after title-casing
    ACRONYMS = {
        "Kyc" : "KYC",
        "Upi" : "UPI",
        "Rbi" : "RBI",
        "Atm" : "ATM",
        "Neft": "NEFT",
    }

    for col in cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()
            for wrong, correct in ACRONYMS.items():
                df[col] = df[col].str.replace(
                    wrong, correct, regex=False
                )
    return df


def _remove_duplicates(df: pd.DataFrame, id_col: str, name: str) -> pd.DataFrame:
    """Check and remove duplicate rows based on ID column."""
    before = len(df)
    df = df.drop_duplicates(subset=[id_col])
    after  = len(df)
    removed = before - after
    if removed > 0:
        log.warning(f"[{name}] Removed {removed} duplicate rows on '{id_col}'")
    else:
        log.info(f"[{name}] No duplicates found on '{id_col}' ✅")
    return df


def _log_nulls(df: pd.DataFrame, name: str, expected_null_cols: list = []) -> None:
    """Log null counts — warn only for unexpected nulls."""
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        log.info(f"[{name}] No nulls found ✅")
    else:
        for col, count in null_counts.items():
            if col in expected_null_cols:
                log.info(f"[{name}] '{col}' has {count} nulls — expected ✅")
            else:
                log.warning(f"[{name}] '{col}' has {count} unexpected nulls ⚠️")


def _validate_range(
    df: pd.DataFrame, col: str, min_val, max_val, name: str
) -> pd.DataFrame:
    """Validate numeric column is within expected range — drop violating rows."""
    if col not in df.columns:
        return df
    mask    = df[col].between(min_val, max_val)
    invalid = (~mask).sum()
    if invalid > 0:
        log.warning(
            f"[{name}] '{col}' has {invalid} values outside "
            f"[{min_val}, {max_val}] — dropping"
        )
        df = df[mask].reset_index(drop=True)
    else:
        log.info(f"[{name}] '{col}' range [{min_val}, {max_val}] valid ✅")
    return df


def _cleaning_report(name: str, before: int, after: int) -> None:
    """Log before/after row count summary."""
    log.info(
        f"[{name}] Rows before: {before:,} → after: {after:,} "
        f"(removed {before - after:,})"
    )


# ── Support Tickets ───────────────────────────────────────────────────────────

def clean_tickets(path: str, output_dir: Path) -> pd.DataFrame:
    """
    Clean support tickets dataset.
    Steps:
      - Remove duplicates on ticket_id
      - Strip whitespace from all string columns
      - Standardise case on categorical columns
      - Validate resolution_time_minutes (1–500) and customer_satisfaction (1–5)
      - Parse date_created
    Saves: data/cleaned/01_support_tickets_cleaned.csv
    """
    log.info("=" * 60)
    log.info("Cleaning Support Tickets")
    log.info("=" * 60)

    df     = pd.read_csv(path)
    before = len(df)
    log.info(f"Loaded {before:,} rows from {path}")

    df = _remove_duplicates(df, "ticket_id", "Tickets")
    _log_nulls(df, "Tickets")
    df = _strip_whitespace(df)
    log.info("[Tickets] Whitespace stripped ✅")

    cat_cols = ["category", "sentiment", "risk_level",
                "channel", "escalated", "resolved_by"]
    df = _standardise_case(df, cat_cols)
    log.info(f"[Tickets] Standardised case for: {cat_cols} ✅")

    df = _validate_range(df, "resolution_time_minutes", 1, 500, "Tickets")
    df = _validate_range(df, "customer_satisfaction",   1, 5,   "Tickets")

    df["date_created"] = pd.to_datetime(df["date_created"], errors="coerce")
    invalid_dates = df["date_created"].isna().sum()
    if invalid_dates > 0:
        log.warning(f"[Tickets] {invalid_dates} unparseable dates — dropping")
        df = df.dropna(subset=["date_created"]).reset_index(drop=True)
    else:
        log.info("[Tickets] date_created parsed successfully ✅")

    _cleaning_report("Tickets", before, len(df))

    out = output_dir / "01_support_tickets_cleaned.csv"
    df.to_csv(out, index=False)
    log.info(f"[Tickets] Saved → {out} ✅\n")
    return df


# ── Transactions ──────────────────────────────────────────────────────────────

def clean_transactions(path: str, output_dir: Path) -> pd.DataFrame:
    """
    Clean transactions dataset.
    Steps:
      - Remove duplicates on transaction_id
      - Strip whitespace
      - Validate amount_inr (> 0), hour_of_day (0–23), fraud_label (0/1)
      - Parse timestamp
      - fraud_reason nulls are expected (only fraud rows have a reason)
    Saves: data/cleaned/02_transactions_cleaned.csv
    """
    log.info("=" * 60)
    log.info("Cleaning Transactions")
    log.info("=" * 60)

    df     = pd.read_csv(path)
    before = len(df)
    log.info(f"Loaded {before:,} rows from {path}")

    df = _remove_duplicates(df, "transaction_id", "Transactions")
    _log_nulls(df, "Transactions", expected_null_cols=["fraud_reason"])
    df = _strip_whitespace(df)
    log.info("[Transactions] Whitespace stripped ✅")

    df = _validate_range(df, "amount_inr",   0.01, 1_000_000, "Transactions")
    df = _validate_range(df, "hour_of_day",  0,    23,         "Transactions")

    invalid_labels = ~df["fraud_label"].isin([0, 1])
    if invalid_labels.sum() > 0:
        log.warning(f"[Transactions] {invalid_labels.sum()} invalid fraud_label values — dropping")
        df = df[~invalid_labels].reset_index(drop=True)
    else:
        log.info("[Transactions] fraud_label values valid (0/1 only) ✅")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_ts = df["timestamp"].isna().sum()
    if invalid_ts > 0:
        log.warning(f"[Transactions] {invalid_ts} unparseable timestamps — dropping")
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    else:
        log.info("[Transactions] timestamp parsed successfully ✅")

    _cleaning_report("Transactions", before, len(df))

    out = output_dir / "02_transactions_cleaned.csv"
    df.to_csv(out, index=False)
    log.info(f"[Transactions] Saved → {out} ✅\n")
    return df


# ── QA Pairs ──────────────────────────────────────────────────────────────────

def clean_qa_pairs(path: str, output_dir: Path) -> list:
    """
    Clean QA pairs JSON.
    Steps:
      - Validate all 7 required fields are present per entry
      - Strip whitespace from all string fields
      - Remove duplicate questions
    Saves: data/cleaned/04_qa_pairs_cleaned.json
    """
    log.info("=" * 60)
    log.info("Cleaning QA Pairs")
    log.info("=" * 60)

    with open(path) as f:
        qa_raw = json.load(f)

    # Normalise to list
    if isinstance(qa_raw, list):
        qa_list = qa_raw
    elif isinstance(qa_raw, dict):
        key     = next((k for k in qa_raw if isinstance(qa_raw[k], list)), None)
        qa_list = qa_raw[key] if key else [qa_raw]
    else:
        qa_list = []

    before = len(qa_list)
    log.info(f"Loaded {before} QA pairs from {path}")

    required_fields = [
        "id", "category", "question", "answer",
        "policy_ref", "risk_level", "suggested_action"
    ]

    cleaned        = []
    skipped        = 0
    seen_questions = set()

    for item in qa_list:
        missing = [f for f in required_fields if f not in item]
        if missing:
            log.warning(f"Skipping QA item missing fields: {missing}")
            skipped += 1
            continue

        item = {k: v.strip() if isinstance(v, str) else v for k, v in item.items()}

        q = item["question"].lower()
        if q in seen_questions:
            log.warning(f"Duplicate question skipped: {item['question'][:50]}...")
            skipped += 1
            continue
        seen_questions.add(q)
        cleaned.append(item)

    if skipped > 0:
        log.warning(f"[QA Pairs] Skipped {skipped} invalid/duplicate entries")
    else:
        log.info("[QA Pairs] No duplicates or missing fields found ✅")

    _cleaning_report("QA Pairs", before, len(cleaned))

    out = output_dir / "04_qa_pairs_cleaned.json"
    with open(out, "w") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    log.info(f"[QA Pairs] Saved → {out} ✅\n")
    return cleaned


# ── Policy Docs ───────────────────────────────────────────────────────────────

def clean_policy_docs(policy_dir: str, output_dir: Path) -> dict:
    """
    Clean policy .txt documents.
    Steps:
      - Validate files are readable and non-empty
      - Remove excessive blank lines and trailing whitespace
    Saves: data/cleaned/policy_docs/<filename>.txt
    """
    log.info("=" * 60)
    log.info("Cleaning Policy Documents")
    log.info("=" * 60)

    policy_path = Path(policy_dir)
    out_policy  = output_dir / "policy_docs"
    out_policy.mkdir(parents=True, exist_ok=True)
    docs = {}

    for fpath in sorted(policy_path.glob("*.txt")):
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")

            if not text.strip():
                log.warning(f"[Policy] '{fpath.name}' is empty — skipping")
                continue

            text  = re.sub(r"\n{3,}", "\n\n", text)
            lines = [line.rstrip() for line in text.splitlines()]
            text  = "\n".join(lines).strip()

            out_path = out_policy / fpath.name
            out_path.write_text(text, encoding="utf-8")

            docs[fpath.name] = text
            log.info(
                f"[Policy] '{fpath.name}' cleaned → "
                f"{len(text.split())} words ✅"
            )

        except Exception as e:
            log.error(f"[Policy] Failed to read '{fpath.name}': {e}")

    log.info(f"[Policy] {len(docs)} documents saved → {out_policy}\n")
    return docs


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Running data_cleaning.py in standalone mode")
    print("CWD:", os.getcwd())

    cleaned_dir = Path("data/cleaned")
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    clean_tickets("data/raw/01_support_tickets.csv",  cleaned_dir)
    clean_transactions("data/raw/02_transactions.csv", cleaned_dir)
    clean_qa_pairs("data/raw/04_qa_pairs.json",        cleaned_dir)
    clean_policy_docs("data/raw/policy_docs/",         cleaned_dir)

    log.info("✅ All datasets cleaned successfully")