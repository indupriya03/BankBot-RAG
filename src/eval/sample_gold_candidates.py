"""
sample_gold_candidates.py
───────────────────────────
Pulls a diverse, deduplicated sample of candidate gold queries from your
real ticket dataset (200 rows), stratified across all four categories.

Your query_text is templated (e.g. "I see a transaction of ₹{amt} I didn't
make..." repeated with different amounts) — sampling naively would just
give near-duplicate gold entries. This script normalizes out numbers/IDs
before deduping, so the sample actually covers distinct *phrasings*, not
just distinct amounts.

Output is a DRAFT — it does NOT invent gold_chunk_ids. You still need to:
  1. For each sampled query, run:
       python src/rag/retriever.py --query "<query_text>" --k 10
  2. Read the actual top candidates + the resolution_text shown here
     (which tells you what the correct answer *should* cover)
  3. Fill in gold_chunk_ids yourself, then merge into golden_queries.json

Usage:
    python src/eval/sample_gold_candidates.py --csv path/to/tickets.csv
    python src/eval/sample_gold_candidates.py --csv path/to/tickets.csv --per-category 8
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _normalize(text: str) -> str:
    """Strip amounts, IDs, and numbers so templated rows collapse to one
    representative phrasing per distinct template, instead of one per
    distinct amount/customer."""
    t = text.lower()
    t = re.sub(r"₹[\d,]+", "<amt>", t)
    t = re.sub(r"\b[a-z]{2,5}-?\d{4,}\b", "<id>", t)   # TKT-10001, CUST872246, TXN500004 etc.
    t = re.sub(r"\d+", "<num>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def sample(df: pd.DataFrame, per_category: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["_template"] = df["query_text"].astype(str).apply(_normalize)

    # One representative row per distinct template, per category —
    # this is what gets you phrasing diversity instead of amount diversity.
    deduped = (
        df.sort_values("date_created")
          .groupby(["category", "_template"], as_index=False)
          .first()
    )

    picked = []
    for cat, group in deduped.groupby("category"):
        n = min(per_category, len(group))
        picked.append(group.sample(n=n, random_state=seed))
    return pd.concat(picked, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to the 200-row ticket CSV")
    parser.add_argument("--per-category", type=int, default=8,
                         help="max distinct-phrasing rows to sample per category")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(Path(__file__).parent / "gold_candidates_draft.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    required_cols = {"category", "query_text", "resolution_text", "risk_level"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing expected columns: {missing}")

    sampled = sample(df, args.per_category, args.seed)

    candidates = []
    for _, row in sampled.iterrows():
        candidates.append({
            "query": str(row["query_text"]).strip(),
            "gold_chunk_ids": ["FILL_ME_IN"],
            "gold_type": "FILL_ME_IN  # 'qa' or 'policy' — check via retriever.py",
            "intent": row["category"],
            "risk_level_in_data": row.get("risk_level", ""),
            "resolution_text_for_reference": str(row.get("resolution_text", ""))[:200],
        })

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"queries": candidates}, indent=2))

    print(f"Sampled {len(candidates)} candidate queries across "
          f"{sampled['category'].nunique()} categories -> {out_path}")
    print("\nBreakdown:")
    print(sampled["category"].value_counts().to_string())
    print(
        "\nNext: for each entry, run "
        "`python src/rag/retriever.py --query \"<query>\" --k 10` and fill "
        "in gold_chunk_ids from the real top results before merging into "
        "golden_queries.json."
    )


if __name__ == "__main__":
    main()