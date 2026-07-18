"""
retriever.py
─────────────
Query the ChromaDB vector store built by build_vector_store.py.

Usage:
    from retriever import Retriever
    r = Retriever()
    results = r.retrieve("I see a transaction I didn't make", k=5)
    results = r.retrieve("KYC issue", k=3, where={"risk_level": "High"})  # metadata filter
"""

import logging
import os
from pathlib import Path
from typing import Optional
import yaml
import chromadb

from gemini_embeddings import get_query_embedder

log = logging.getLogger(__name__)

def _load_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)

_params = _load_params()

COLLECTION_NAME = "bankbot_rag"


class Retriever:
    def __init__(self, persist_dir: str = None):
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "data/vector_store/chroma_db")
        persist_path = Path(persist_dir)
        if not persist_path.exists():
            raise FileNotFoundError(
                f"No vector store found at {persist_dir}. Run build_vector_store.py first."
            )
        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.embedder = get_query_embedder()
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedder,
        )
        log.info(f"Retriever ready — {self.collection.count()} documents in collection")

    def retrieve(self, query: str, k: int = _params["rag"]["top_k"], where: Optional[dict] = None,
                 min_policy_results: int = _params["rag"]["min_policy_results"]) -> list:
        """
        Args:
            query: user's search query
            k: number of results to return
            where: optional Chroma metadata filter, e.g. {"risk_level": "High"}
                   or {"source": "policy_docs"} — combine with $and/$or per
                   Chroma's filter syntax for multiple conditions.
            min_policy_results: minimum number of policy-document chunks to
                   guarantee in the result set, when the collection has any
                   that match `where`. Pure cosine similarity structurally
                   favors QA-pair chunks over policy chunks for a typical
                   customer query — QA pairs are phrased like a customer
                   message ("Question: ... Answer: ..."), while policy text
                   is formal/regulatory and often chunked at section
                   granularity (covering several sub-topics at once), so it
                   scores worse on pure similarity even when it's the more
                   authoritative source. Set to 0 to disable and get pure
                   similarity ranking.

        Returns:
            List of {"chunk_id", "text", "metadata", "distance"} dicts,
            ordered by relevance (lowest distance first).
        """
        # Pull the full ranked pool (cheap at this collection's scale) so the
        # policy floor below can draw from every matching policy chunk, not
        # just whichever ones happened to land in a small top-k window.
        pool_size = self.collection.count()
        raw = self.collection.query(
            query_texts=[query],
            n_results=pool_size,
            where=where,
        )

        ids       = raw.get("ids", [[]])[0]
        docs      = raw.get("documents", [[]])[0]
        metas     = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        # Chroma returns results already sorted by distance (ascending).
        candidates = [
            {"chunk_id": cid, "text": doc, "metadata": meta, "distance": dist}
            for cid, doc, meta, dist in zip(ids, docs, metas, distances)
        ]

        if min_policy_results <= 0 or not candidates:
            return candidates[:k]

        def _is_qa_chunk(c: dict) -> bool:
            return str(c["chunk_id"]).startswith("qa_")

        policy_candidates = [c for c in candidates if not _is_qa_chunk(c)]

        guaranteed = policy_candidates[:min_policy_results]
        guaranteed_ids = {c["chunk_id"] for c in guaranteed}

        remaining_slots = max(k - len(guaranteed), 0)
        rest = [c for c in candidates if c["chunk_id"] not in guaranteed_ids]
        # rest is already distance-sorted since it's a filtered subsequence
        # of the distance-sorted candidates list
        final = guaranteed + rest[:remaining_slots]

        return sorted(final, key=lambda c: c["distance"])[:k]


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--persist-dir", default=None)
    args = parser.parse_args()

    r = Retriever(persist_dir=args.persist_dir)
    results = r.retrieve(args.query, k=args.k)

    print(f"\nTop {len(results)} results for: \"{args.query}\"\n")
    for i, res in enumerate(results, 1):
        print(f"[{i}] distance={res['distance']:.4f} chunk_id={res['chunk_id']}")
        print(f"    {res['text'][:150]}...")
        print(f"    metadata: {res['metadata']}\n")