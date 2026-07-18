"""
preprocessing.py
────────────────
Thin orchestrator over the three independent preprocessing modules:

  preprocessing_tickets.py       → intent + sentiment classifiers
  preprocessing_transactions.py  → fraud model
  preprocessing_rag.py           → RAG retrieval (LangChain-based chunking)

Each module only depends on what its own consumer needs (no shared
clean_text, no shared imports across unrelated models). This file exists
so pipeline.py / dvc.yaml can still call one script and get everything,
but each stage below can also be run and DVC-tracked independently if you
want to iterate on, say, RAG chunking without re-running fraud preprocessing.
"""

import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing_tickets import preprocess_tickets
from preprocessing_transactions import preprocess_transactions
from preprocessing_rag import preprocess_qa_pairs, preprocess_policy_docs, build_rag_master_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_all(cleaned_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_tickets(cleaned_dir, output_dir)
    preprocess_transactions(cleaned_dir, output_dir)

    qa_docs     = preprocess_qa_pairs(cleaned_dir, output_dir)
    policy_docs = preprocess_policy_docs(cleaned_dir, output_dir)
    build_rag_master_index(qa_docs, policy_docs, output_dir)

    log.info("✅ All preprocessing complete → %s", output_dir)


if __name__ == "__main__":
    run_all(Path("data/cleaned"), Path("data/preprocessed"))