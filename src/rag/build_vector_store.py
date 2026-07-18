"""
build_vector_store.py
──────────────────────
Builds a persistent ChromaDB collection from rag_master_chunks.json
(QA pairs + policy doc chunks, produced by preprocessing_rag.py).

Usage:
    export GEMINI_API_KEY=your-key-here   # free at https://aistudio.google.com/apikey
    python build_vector_store.py
"""

import argparse
import json
import logging
from pathlib import Path

import chromadb

from gemini_embeddings import get_document_embedder, BATCH_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

COLLECTION_NAME = "bankbot_rag"
ADD_BATCH_SIZE = BATCH_SIZE  # keep add() calls aligned with the embedding batch size


def build_vector_store(chunks_file: Path, persist_dir: Path, reset: bool = False) -> None:
    log.info("=" * 60)
    log.info("Building Vector Store")
    log.info("=" * 60)

    with open(chunks_file) as f:
        chunks = json.load(f)
    log.info(f"Loaded {len(chunks)} documents from {chunks_file}")

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info(f"Deleted existing collection '{COLLECTION_NAME}' (--reset)")
        except Exception:
            pass

    embedder = get_document_embedder()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0 and not reset:
        log.info(
            f"Collection already has {existing_count} documents. "
            f"Use --reset to rebuild from scratch, or this will just add/upsert on top."
        )

    ids, documents, metadatas = [], [], []
    for chunk in chunks:
        chunk_id = chunk["metadata"].get("chunk_id")
        if not chunk_id:
            log.warning("Skipping chunk with no chunk_id in metadata")
            continue
        ids.append(chunk_id)
        documents.append(chunk["page_content"])
        # Chroma metadata values must be str/int/float/bool — flatten anything else
        clean_meta = {
            k: (v if isinstance(v, (str, int, float, bool)) else str(v))
            for k, v in chunk["metadata"].items()
        }
        metadatas.append(clean_meta)

    log.info(f"Prepared {len(ids)} documents for ingestion (embedding via {embedder.task_type})")

    for i in range(0, len(ids), ADD_BATCH_SIZE):
        batch_ids   = ids[i:i + ADD_BATCH_SIZE]
        batch_docs  = documents[i:i + ADD_BATCH_SIZE]
        batch_metas = metadatas[i:i + ADD_BATCH_SIZE]
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
        log.info(f"  Upserted batch {i//ADD_BATCH_SIZE + 1} ({len(batch_ids)} docs)")

    final_count = collection.count()
    log.info(f"✅ Vector store built: {final_count} documents in collection '{COLLECTION_NAME}'")
    log.info(f"   Persisted to: {persist_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-file", default="data/preprocessed/rag_master_chunks.json")
    parser.add_argument("--persist-dir", default="data/vector_store/chroma_db")
    parser.add_argument("--reset", action="store_true",
                         help="Delete and rebuild the collection from scratch instead of upserting")
    args = parser.parse_args()

    build_vector_store(Path(args.chunks_file), Path(args.persist_dir), reset=args.reset)


if __name__ == "__main__":
    main()