"""
eval_generation.py
────────────────────
Evaluates the LLM's GENERATED answers, holding retrieval constant.

eval_retrieval.py already checks whether the retriever finds the right
chunks. This script isolates a different question: GIVEN the correct
chunks, does the LLM generate a faithful, accurate answer? To isolate
that, this fetches the gold chunk(s) directly from ChromaDB by ID (no
embedding search involved) rather than trusting a fresh retrieval — so
any hallucination found here is a generation problem, not a retrieval one.

For each of the 13 gold queries, this:
  1. Fetches the true gold chunk(s) by ID
  2. Calls the real generate_response() Groq pipeline with that context
  3. Uses an LLM-judge to check:
       - Faithfulness: does every claim in the response trace back to
         the provided context? (catches hallucination)
       - Citation accuracy: do the chunk_ids the model CLAIMS it used
         (grounded_in) actually match what it was given? (catches
         citation drift — the qa_QA006 mislabeling earlier in this
         project was exactly this kind of mismatch, just caught by a
         human instead of an automated check)

Usage:
    python src/eval/eval_generation.py
    python src/eval/eval_generation.py --golden src/eval/golden_queries.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

_SRC_DIR = Path(__file__).resolve().parent.parent
for _sub in ("rag", "llm"):
    _p = str(_SRC_DIR / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chromadb  # noqa: E402
from groq import Groq  # noqa: E402
from response_generator import generate_response  # noqa: E402

COLLECTION_NAME = "bankbot_rag"
JUDGE_MODEL = "llama-3.3-70b-versatile"


def fetch_chunks_by_id(collection, chunk_ids: list) -> list:
    """Direct ID lookup — no embedding/query step, so this is the TRUE
    gold context, not whatever a fresh retrieval happens to rank."""
    data = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
    return [
        {"chunk_id": cid, "text": doc, "metadata": meta or {}}
        for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]


def judge_faithfulness(client: Groq, query: str, context_text: str, response_text: str) -> dict:
    """LLM-judge call — flags any claim in the response not supported by
    the context. temperature=0 for consistent judging."""
    judge_prompt = f"""You are a strict fact-checker. Given a CONTEXT and a
RESPONSE that was supposed to be grounded in that context, determine
whether the response makes any factual claim NOT supported by the context.

CONTEXT:
{context_text}

CUSTOMER QUESTION:
{query}

RESPONSE TO CHECK:
{response_text}

Respond with ONLY a JSON object, no other text:
{{
  "faithful": true or false,
  "unsupported_claims": ["specific claims not supported by the context, empty list if none"],
  "answers_question": true or false
}}"""
    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(Path(__file__).parent / "golden_queries.json"))
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", "data/vector_store/chroma_db"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "generation_eval_results.json"))
    args = parser.parse_args()

    data = json.loads(Path(args.golden).read_text())
    queries = [q for q in data["queries"] if "REPLACE_ME" not in q.get("gold_chunk_ids", [])]

    chroma_client = chromadb.PersistentClient(path=args.persist_dir)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    results = []
    for item in queries:
        query    = item["query"]
        gold_ids = item["gold_chunk_ids"]
        intent   = item.get("intent", "Unknown")

        chunks = fetch_chunks_by_id(collection, gold_ids)
        if not chunks:
            print(f"⚠️  Skipping '{query}' — gold chunk_id(s) {gold_ids} not found in vector store")
            continue

        gen_result = generate_response(
            query=query,
            intent={"intent": intent, "confidence": 1.0},
            sentiment={"sentiment": "Neutral", "confidence": 1.0},
            retrieved_chunks=chunks,
        )

        context_text = "\n\n".join(c["text"] for c in chunks)
        judge = judge_faithfulness(groq_client, query, context_text, gen_result["response"])

        cited  = set(gen_result.get("grounded_in", []))
        actual = set(gold_ids)
        # Every cited chunk must actually be one it was given (not invented),
        # and it must have cited at least one.
        citation_correct = bool(cited) and cited.issubset(actual)

        row = {
            "query"              : query,
            "faithful"           : judge.get("faithful"),
            "answers_question"   : judge.get("answers_question"),
            "unsupported_claims" : judge.get("unsupported_claims", []),
            "cited_chunks"       : sorted(cited),
            "gold_chunks"        : gold_ids,
            "citation_correct"   : citation_correct,
            "response"           : gen_result["response"],
        }
        results.append(row)

        status = "✅" if judge.get("faithful") else "❌ HALLUCINATION"
        print(f"{status} | {query[:55]}")
        if not judge.get("faithful"):
            print(f"    unsupported claims: {judge.get('unsupported_claims')}")
        if not citation_correct:
            print(f"    citation mismatch — cited {sorted(cited)}, gold was {gold_ids}")

    n = len(results)
    if n == 0:
        print("No queries evaluated — check golden_queries.json / vector store path.")
        return

    faithful_rate = sum(bool(r["faithful"]) for r in results) / n
    answers_rate  = sum(bool(r["answers_question"]) for r in results) / n
    citation_rate = sum(r["citation_correct"] for r in results) / n

    print("\n" + "=" * 60)
    print(f"Generation Eval — n={n}")
    print(f"  Faithfulness rate      : {faithful_rate:.2%}")
    print(f"  Answers-question rate  : {answers_rate:.2%}")
    print(f"  Citation accuracy rate : {citation_rate:.2%}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nDetailed per-query results saved to {args.out}")


if __name__ == "__main__":
    main()