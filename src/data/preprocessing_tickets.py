"""
preprocessing_tickets.py
────────────────────────
Preprocessing for the intent + sentiment classifiers ONLY.
Reads data/cleaned/01_support_tickets_cleaned.csv, saves to data/preprocessed/.

Deliberately minimal: zero-shot NLI models (bart-large-mnli) are trained on
natural, cased, punctuated text — they don't have a fixed vocabulary the way
TF-IDF does, so aggressive lowercasing/punctuation-stripping isn't obviously
helping and may hurt. This module keeps only the cleaning that removes actual
noise (HTML, URLs) and leaves case/punctuation untouched by default.

If you A/B test raw vs. stripped-noise vs. fully-normalized text (see
evaluate_nlp.py --text-col) and one wins clearly, adjust LIGHT_CLEAN_ONLY below.

Outputs:
  tickets_eval_unique.csv    Deduplicated ticket set for intent/sentiment eval
  support_tickets_clean.csv  Full cleaned tickets (kept for reference/debugging)
"""

import logging
import re
from pathlib import Path
import yaml
import pandas as pd

log = logging.getLogger(__name__)

def _load_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)

_params = _load_params()


# Set False if your A/B test shows the fully-normalized (lowercase, stripped)
# text scores better on your eval set.
LIGHT_CLEAN_ONLY = _params["tickets_preprocessing"]["light_clean_only"]


def light_clean(text: str) -> str:
    """Strip HTML/URLs only. Preserves case and punctuation for NLI models."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def full_clean(text: str) -> str:
    """Legacy normalization: lowercase + strip to a narrow character set.
    Kept for the raw-vs-cleaned A/B comparison and for any traditional-ML
    fallback path."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s₹.,?!'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_tickets(cleaned_dir: Path, output_dir: Path) -> None:
    log.info("=" * 60)
    log.info("Preprocessing Support Tickets (intent + sentiment)")
    log.info("=" * 60)

    path = cleaned_dir / "01_support_tickets_cleaned.csv"
    df   = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows from {path}")

    clean_fn = light_clean if LIGHT_CLEAN_ONLY else full_clean
    df["query_text_clean"]      = df["query_text"].apply(clean_fn)
    df["resolution_text_clean"] = df["resolution_text"].apply(clean_fn)
    # also keep the fully-normalized variant around for the A/B eval comparison
    df["query_text_normalized"] = df["query_text"].apply(full_clean)
    df["query_length"]          = df["query_text_clean"].apply(lambda x: len(x.split()))
    log.info(f"Text cleaned (light_clean={LIGHT_CLEAN_ONLY}) | avg query length: "
              f"{df['query_length'].mean():.1f} words ✅")

    n_before = len(df)
    eval_df  = (
        df.drop_duplicates(subset=["query_text_clean", "category"])
          .reset_index(drop=True)
    )
    n_after = len(eval_df)
    log.info(f"Deduplicated for eval: {n_before} → {n_after} unique rows "
              f"({n_before - n_after} duplicates removed) ✅")
    log.info(f"Category distribution (eval set): {dict(eval_df['category'].value_counts())}")

    if n_after < 30:
        log.warning(
            f"Only {n_after} unique labeled queries available for evaluation. "
            "Treat metrics as directional and prioritize collecting more "
            "distinct real queries before reporting numbers externally."
        )

    eval_df.to_csv(output_dir / "tickets_eval_unique.csv", index=False)
    log.info("tickets_eval_unique.csv saved ✅")

    df.to_csv(output_dir / "support_tickets_clean.csv", index=False)
    log.info("support_tickets_clean.csv saved ✅\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    out = Path("data/preprocessed")
    out.mkdir(parents=True, exist_ok=True)
    preprocess_tickets(Path("data/cleaned"), out)