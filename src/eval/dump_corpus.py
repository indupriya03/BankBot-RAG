"""
dump_corpus.py
────────────────
Dumps EVERY chunk in the ChromaDB collection, independent of any query.

Why this matters: label_helper.py builds gold labels only from what a
query's top-5 retrieval happened to surface. That guarantees every gold
chunk is already "findable" by construction, which inflates recall@k —
it can never catch the case where the retriever completely misses a
chunk that should have been top-ranked. With only 33 chunks total, it's
cheap to instead browse the WHOLE corpus once and label against that,
independent of any retriever run.

Usage:
    python src/eval/dump_corpus.py
    python src/eval/dump_corpus.py --persist-dir data/vector_store/chroma_db
"""

import argparse
from pathlib import Path

import chromadb

COLLECTION_NAME = "bankbot_rag"  # matches retriever.py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", default="data/vector_store/chroma_db")
    parser.add_argument("--out", default=str(Path(__file__).parent / "full_corpus.md"))
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.persist_dir)
    collection = client.get_collection(name=COLLECTION_NAME)

    # get() with no query — pulls everything, no embedding call needed
    data = collection.get(include=["documents", "metadatas"])
    ids       = data["ids"]
    docs      = data["documents"]
    metas     = data["metadatas"]

    # Group by source type for easier browsing
    rows = sorted(zip(ids, docs, metas), key=lambda r: (r[2].get("source", ""), r[0]))

    lines = [
        "# Full Corpus Dump",
        "",
        f"{len(ids)} total chunks. Use this to label gold entries against the "
        f"FULL universe, not just what one query's retrieval surfaced.",
        "",
    ]

    current_source = None
    for cid, doc, meta in rows:
        source = meta.get("source", "unknown")
        if source != current_source:
            lines.append(f"\n## source: {source}\n")
            current_source = source

        category = meta.get("category") or meta.get("document") or "?"
        lines.append(f"### `{cid}` ({category})")
        lines.append(f"```\n{doc.strip()}\n```")
        lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print(f"Dumped {len(ids)} chunks -> {args.out}")
    print(
        "\nFor each candidate gold query, read through this file (not just a "
        "retriever run) and pick the truly correct chunk(s), independent of "
        "whether the retriever currently surfaces them. If NONE match, that's "
        "a genuine coverage gap. If one matches but the retriever missed it in "
        "the earlier worksheet, that's a real recall miss worth investigating."
    )


if __name__ == "__main__":
    main()