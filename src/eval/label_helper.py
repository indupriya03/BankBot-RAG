"""
label_helper.py
─────────────────
Takes gold_candidates_draft.json (from sample_gold_candidates.py), runs
every query through the real Retriever ONCE each, and writes a single
worksheet showing the top candidates side-by-side with the ticket's
resolution_text — so you can eyeball and fill in gold_chunk_ids without
running retriever.py's CLI 32 separate times.

This does NOT auto-label anything. It only assembles the evidence you need
to label quickly and correctly (avoiding the earlier qa_QA006 mistake,
which happened from guessing a gold_chunk_id without checking the actual
retrieved content).

Usage:
    python src/eval/label_helper.py
    python src/eval/label_helper.py --candidates src/eval/gold_candidates_draft.json --top-n 5
"""

import argparse
import json
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
_RAG_DIR = _SRC_DIR / "rag"
if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

from retriever import Retriever  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default=str(Path(__file__).parent / "gold_candidates_draft.json"))
    parser.add_argument("--persist-dir", default="data/vector_store/chroma_db")
    parser.add_argument("--top-n", type=int, default=5,
                         help="how many retrieved candidates to show per query")
    parser.add_argument("--out", default=str(Path(__file__).parent / "labeling_worksheet.md"))
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    data = json.loads(candidates_path.read_text())
    queries = data["queries"]

    retriever = Retriever(persist_dir=args.persist_dir)

    lines = ["# Gold Labeling Worksheet", "", f"{len(queries)} queries — fill in `gold_chunk_ids` for each.", ""]

    for i, item in enumerate(queries, 1):
        query = item["query"]
        intent = item.get("intent", "?")
        resolution = item.get("resolution_text_for_reference", "")

        results = retriever.retrieve(query, k=args.top_n)

        lines.append(f"## [{i}] ({intent}) \"{query}\"")
        lines.append("")
        lines.append(f"**Ticket's actual resolution:** {resolution}")
        lines.append("")
        lines.append("**Top retrieved candidates:**")
        for rank, r in enumerate(results, 1):
            preview = r["text"][:120].replace("\n", " ")
            lines.append(f"  {rank}. `{r['chunk_id']}` (distance={r['distance']:.4f}) — {preview}...")
        lines.append("")
        lines.append("**gold_chunk_ids:** _(fill in from above — usually 1, pick the chunk(s) "
                      "that actually match the resolution, not just the top rank)_")
        lines.append("**gold_type:** _(qa / policy)_")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines))
    print(f"Wrote worksheet for {len(queries)} queries -> {out_path}")
    print("Open it, fill in gold_chunk_ids/gold_type per query, then transfer "
          "the finished entries into golden_queries.json.")


if __name__ == "__main__":
    main()