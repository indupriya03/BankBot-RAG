"""
eval_retrieval.py
──────────────────
Offline evaluation of src/rag/retriever.py against a hand-labeled gold set.

This is NOT a unit test (no pass/fail assertion) — it produces metrics you
compare across experiments (embedding model, chunk size, k, the
min_policy_results floor). Lives in src/eval, not src/rag or src/tests,
because it imports Retriever rather than being part of it, and because it's
run manually/periodically rather than on every commit.

Usage:
    python src/eval/eval_retrieval.py
    python src/eval/eval_retrieval.py --golden src/eval/golden_queries.json --k 5
    python src/eval/eval_retrieval.py --compare-policy-floor
    python src/eval/eval_retrieval.py --mlflow --experiment rag-retrieval-eval
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# Make src/rag importable regardless of where this is run from
_SRC_DIR = Path(__file__).resolve().parent.parent
_RAG_DIR = _SRC_DIR / "rag"
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

from retriever import Retriever  # noqa: E402


# ── Metrics ───────────────────────────────────────────────────────────────
def recall_at_k(retrieved_ids: list, gold_ids: set, k: int) -> float:
    """1.0 if ANY gold chunk appears in the top-k, else 0.0.
    (Standard for single/near-single-relevant-doc gold sets like this one —
    use a coverage fraction instead if you start labeling multi-chunk golds.)"""
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & gold_ids else 0.0


def precision_at_k(retrieved_ids: list, gold_ids: set, k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for cid in top_k if cid in gold_ids)
    return hits / len(top_k)


def reciprocal_rank(retrieved_ids: list, gold_ids: set) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0

def _safe_metric_name(name: str) -> str:
    """MLflow's REST API (DagsHub) rejects '@' in metric names, even
    though the local file-store backend silently allowed it. Convert
    recall@3 -> recall_at_3 etc."""
    return name.replace("@", "_at_")


# ── Core eval loop ────────────────────────────────────────────────────────
def evaluate(
    retriever: Retriever,
    golden_queries: list,
    k_values: list,
    min_policy_results: int,
) -> dict:
    """Run every gold query through the retriever and aggregate metrics.
    Returns overall metrics plus a breakdown by gold_type (qa vs policy),
    since your policy floor logic specifically targets that split."""
    per_query = []

    for item in golden_queries:
        query    = item["query"]
        gold_ids = set(item["gold_chunk_ids"])
        gtype    = item.get("gold_type", "unknown")

        max_k = max(k_values)
        results = retriever.retrieve(
            query, k=max_k, min_policy_results=min_policy_results
        )
        retrieved_ids = [r["chunk_id"] for r in results]

        row = {
            "query": query,
            "gold_type": gtype,
            "mrr": reciprocal_rank(retrieved_ids, gold_ids),
        }
        for k in k_values:
            row[f"recall@{k}"]    = recall_at_k(retrieved_ids, gold_ids, k)
            row[f"precision@{k}"] = precision_at_k(retrieved_ids, gold_ids, k)
        per_query.append(row)

    def _mean(rows: list, field: str) -> float:
        return round(sum(r[field] for r in rows) / len(rows), 4) if rows else 0.0

    def _summarize(rows: list) -> dict:
        summary = {"n": len(rows), "mrr": _mean(rows, "mrr")}
        for k in k_values:
            summary[f"recall@{k}"]    = _mean(rows, f"recall@{k}")
            summary[f"precision@{k}"] = _mean(rows, f"precision@{k}")
        return summary

    overall = _summarize(per_query)
    by_type = {
        gtype: _summarize([r for r in per_query if r["gold_type"] == gtype])
        for gtype in sorted({r["gold_type"] for r in per_query})
    }

    return {"overall": overall, "by_type": by_type, "per_query": per_query}


def _print_report(title: str, report: dict) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")
    print(f"Overall (n={report['overall']['n']}): {report['overall']}")
    for gtype, summary in report["by_type"].items():
        print(f"  [{gtype}] (n={summary['n']}): {summary}")


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(Path(__file__).parent / "golden_queries.json"))
    parser.add_argument("--persist-dir", default="data/vector_store/chroma_db")
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10],
                         help="k values to compute recall/precision at")
    parser.add_argument("--min-policy-results", type=int, default=1,
                         help="value passed through to Retriever.retrieve")
    parser.add_argument("--compare-policy-floor", action="store_true",
                         help="run twice: min_policy_results=1 vs 0, side by side")
    parser.add_argument("--mlflow", action="store_true",
                         help="log metrics to MLflow (same tracking store as the fraud pipeline)")
    parser.add_argument("--experiment", default="rag-retrieval-eval")
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        sys.exit(
            f"Gold set not found at {golden_path}. Fill in golden_queries.json "
            f"with labeled (query, gold_chunk_ids) pairs first."
        )
    data = json.loads(golden_path.read_text())
    golden_queries = data["queries"]

    placeholders = [q for q in golden_queries if "REPLACE_ME" in q.get("gold_chunk_ids", [])]
    if placeholders:
        print(
            f"⚠️  {len(placeholders)} gold entries still have REPLACE_ME chunk_ids "
            f"— fill these in for accurate metrics. Skipping them for now."
        )
        golden_queries = [q for q in golden_queries if q not in placeholders]

    if not golden_queries:
        sys.exit("No usable gold queries after filtering placeholders — nothing to evaluate.")

    retriever = Retriever(persist_dir=args.persist_dir)

    mlflow_run = None
    if args.mlflow:
        try:
            import mlflow
            import os
            # ── DagsHub MLflow tracking ─────────────────────────────────
            user  = os.getenv("DAGSHUB_USERNAME")
            token = os.getenv("DAGSHUB_TOKEN", "")

            if not user or not token:
                raise RuntimeError(
                    "DAGSHUB_USERNAME and DAGSHUB_TOKEN must be set in .env for MLflow tracking."
                )

            os.environ["MLFLOW_TRACKING_USERNAME"] = user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            # ─────────────────────────────────────────────────────────────
            mlflow.set_experiment(args.experiment)
            mlflow_run = mlflow.start_run()
        except ImportError:
            print("⚠️  --mlflow requested but mlflow isn't installed; skipping logging.")
            args.mlflow = False

    if args.compare_policy_floor:
        report_on  = evaluate(retriever, golden_queries, args.k, min_policy_results=1)
        report_off = evaluate(retriever, golden_queries, args.k, min_policy_results=0)
        _print_report("min_policy_results=1 (current default)", report_on)
        _print_report("min_policy_results=0 (pure similarity)", report_off)

        if mlflow_run:
            import mlflow
            for k, v in report_on["overall"].items():
                mlflow.log_metric(_safe_metric_name(f"policy_floor_on_{k}"), v)
            for k, v in report_off["overall"].items():
                mlflow.log_metric(_safe_metric_name(f"policy_floor_off_{k}"), v)
    else:
        report = evaluate(
            retriever, golden_queries, args.k,
            min_policy_results=args.min_policy_results,
        )
        _print_report(f"Retrieval Eval (min_policy_results={args.min_policy_results})", report)

        if mlflow_run:
            import mlflow
            mlflow.log_param("min_policy_results", args.min_policy_results)
            mlflow.log_param("k_values", str(args.k))
            for k, v in report["overall"].items():
                mlflow.log_metric(_safe_metric_name(k), v)
    if mlflow_run:
        import mlflow
        mlflow.end_run()
        print(f"\nLogged to MLflow experiment: {args.experiment}")


if __name__ == "__main__":
    main()