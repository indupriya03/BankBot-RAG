"""
preprocessing_rag.py
─────────────────────
Preprocessing for the RAG retrieval layer ONLY.
Reads data/cleaned/, saves data/preprocessed/*.json.

Uses LangChain's RecursiveCharacterTextSplitter instead of the old manual
word-count chunker. Two concrete wins over the hand-rolled version:

  1. Recursive splitting tries paragraph → line → sentence → word breaks
     in order, so chunks end on natural boundaries instead of mid-sentence
     at a fixed word count.
  2. Every chunk becomes a langchain_core Document: `page_content` + a
     `metadata` dict. That metadata (category, risk_level, source document,
     chunk position) travels with the chunk into whatever vector store you
     use next (Chroma/FAISS/pgvector all accept Document objects directly),
     so retrieval can filter ("only risk_level=high", "only policy_docs")
     instead of relying purely on similarity search.

Output format: each JSON file is a list of {"page_content": ..., "metadata": {...}}
records — this is exactly langchain's Document schema, so loading is just:

    from langchain_core.documents import Document
    docs = [Document(**r) for r in json.load(open("rag_master_chunks.json"))]

Outputs:
  qa_chunks.json           QA pairs as Documents (rich metadata, no splitting needed)
  policy_chunks.json       Policy docs recursively chunked as Documents
  rag_master_chunks.json   Combined QA + policy Documents for vector store ingestion
"""

import json
import logging
from pathlib import Path
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)


def _load_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)

_params = _load_params()

POLICY_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_params["rag"]["chunk_size"],
    chunk_overlap=_params["rag"]["overlap"],
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)


# ── Section 1: QA Pairs ───────────────────────────────────────────────────────

def preprocess_qa_pairs(cleaned_dir: Path, output_dir: Path) -> list:
    """
    Convert cleaned QA pairs into RAG-ready Documents.

    QA entries are already atomic (one question + one answer), so they
    aren't run through the text splitter — splitting would break the
    question/answer pairing. Instead each becomes one Document with the
    full structured metadata attached, so retrieval can filter by
    category / risk_level / policy_ref directly.
    """
    log.info("=" * 60)
    log.info("Preprocessing QA Pairs → RAG Documents")
    log.info("=" * 60)

    path = cleaned_dir / "04_qa_pairs_cleaned.json"
    with open(path) as f:
        qa_data = json.load(f)
    log.info(f"Loaded {len(qa_data)} QA pairs")

    documents = []
    for item in qa_data:
        text = (
            f"Question: {item.get('question', '')}\n"
            f"Answer: {item.get('answer', '')}"
        )
        documents.append({
            "page_content": text,
            "metadata": {
                "chunk_id"        : f"qa_{item['id']}",
                "source"          : "qa_pairs",
                "category"        : item.get("category", ""),
                "risk_level"      : item.get("risk_level", ""),
                "policy_ref"      : item.get("policy_ref", ""),
                "suggested_action": item.get("suggested_action", ""),
                "word_count"      : len(text.split()),
            },
        })

    with open(output_dir / "qa_chunks.json", "w") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    avg_words = sum(d["metadata"]["word_count"] for d in documents) / len(documents)
    log.info(f"QA documents created: {len(documents)} | avg words: {avg_words:.0f}")
    log.info("qa_chunks.json saved ✅\n")
    return documents


# ── Section 2: Policy Documents ──────────────────────────────────────────────

def preprocess_policy_docs(cleaned_dir: Path, output_dir: Path) -> list:
    """
    Recursively chunk cleaned policy .txt files into Documents using
    LangChain's RecursiveCharacterTextSplitter (paragraph/sentence-aware,
    unlike the old fixed-word-count splitter).
    """
    log.info("=" * 60)
    log.info("Preprocessing Policy Docs → RAG Documents")
    log.info("=" * 60)

    policy_dir = cleaned_dir / "policy_docs"
    documents = []

    for fpath in sorted(policy_dir.glob("*.txt")):
        text = fpath.read_text(encoding="utf-8", errors="ignore")
        doc_name = fpath.stem

        chunks = POLICY_SPLITTER.create_documents(
            texts=[text],
            metadatas=[{"document": doc_name}],
        )

        for i, chunk in enumerate(chunks):
            documents.append({
                "page_content": chunk.page_content,
                "metadata": {
                    "chunk_id"    : f"{doc_name}_chunk_{i+1:03d}",
                    "source"      : "policy_docs",
                    "document"    : doc_name,
                    "chunk_num"   : i + 1,
                    "total_chunks": len(chunks),
                    "word_count"  : len(chunk.page_content.split()),
                    "char_count"  : len(chunk.page_content),
                },
            })

        log.info(f"  {fpath.name:<35} → {len(chunks)} chunks ({len(text.split())} words)")

    with open(output_dir / "policy_chunks.json", "w") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    log.info(f"\nTotal policy chunks: {len(documents)}")
    log.info("policy_chunks.json saved ✅\n")
    return documents


# ── Section 3: Master RAG Index ───────────────────────────────────────────────

def build_rag_master_index(qa_docs: list, policy_docs: list, output_dir: Path) -> None:
    """Combine QA + policy Documents into one master file for vector store ingestion."""
    log.info("=" * 60)
    log.info("Building Master RAG Document Index")
    log.info("=" * 60)

    master = qa_docs + policy_docs
    with open(output_dir / "rag_master_chunks.json", "w") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)

    log.info(f"Master RAG index: {len(master)} total documents")
    log.info(f"  QA pairs   : {len(qa_docs)}")
    log.info(f"  Policy docs: {len(policy_docs)}")
    log.info("rag_master_chunks.json saved ✅")
    log.info(
        "\nTo load into a vector store:\n"
        "  from langchain_core.documents import Document\n"
        "  docs = [Document(**r) for r in json.load(open('rag_master_chunks.json'))]\n"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    out = Path("data/preprocessed")
    out.mkdir(parents=True, exist_ok=True)
    cleaned = Path("data/cleaned")

    qa_docs     = preprocess_qa_pairs(cleaned, out)
    policy_docs = preprocess_policy_docs(cleaned, out)
    build_rag_master_index(qa_docs, policy_docs, out)