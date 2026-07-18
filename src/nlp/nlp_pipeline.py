"""
nlp_pipeline.py
───────────────
Unified NLP layer for BankBot-RAG: cleaning → intent → sentiment.

This wraps the three existing modules into one callable so the RAG layer
(and the evaluation script) only need one entry point instead of wiring
preprocessing.clean_text + intent_classifier + sentiment_analyzer by hand
every time.

Usage:
    from nlp_pipeline import process_query
    result = process_query("I see a transaction of ₹10,000 I didn't make")
"""

import logging
import time
import sys
from pathlib import Path

# Add project root to path so src.data imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.preprocessing_tickets import light_clean
from intent_classifier import classify_intent
from sentiment_analyzer import analyze_sentiment


log = logging.getLogger(__name__)

# Routing hints for priority escalation:
# - Fraud always escalates regardless of sentiment (money at risk, time-sensitive).
# - Urgent sentiment always escalates (customer explicitly signals urgency).
# - Negative sentiment alone is NOT enough — most support-chat customers already
#   carry some negative sentiment (that's why they're messaging support), so a
#   blanket "Negative -> high priority" rule would flag most traffic and stop
#   meaning anything. Instead, Negative only escalates when BOTH:
#     (a) intent is one where the customer is functionally blocked from banking
#         entirely (locked out, can't verify identity) — not just unhappy about
#         a decision like a loan status, and
#     (b) the sentiment model is reasonably confident (>=70%), not a borderline call.
HIGH_PRIORITY_INTENTS = {"Fraud/Unauthorized"}
ESCALATION_SENSITIVE_INTENTS = {"Account Access", "KYC"}
NEGATIVE_CONFIDENCE_THRESHOLD = 0.70

def process_query(raw_query: str, apply_cleaning: bool = True) -> dict:
    """
    Run a raw customer query through the full NLP layer.

    Args:
        raw_query      : Raw customer text
        apply_cleaning : Whether to run preprocessing_tickets.light_clean() first
                          (strips HTML/URLs only, preserves case/punctuation —
                          zero-shot NLI models work off natural language, not
                          a fixed vocabulary, so heavy normalization isn't
                          obviously helpful). Default True.

    Returns:
        {
            "raw_query"     : "...",
            "cleaned_query" : "...",
            "intent"        : {...},      # output of classify_intent()
            "sentiment"     : {...},      # output of analyze_sentiment()
            "priority"      : "high" | "normal",
            "latency_ms"    : 842.3
        }
    """
    t0 = time.perf_counter()

    query_for_model = light_clean(raw_query) if apply_cleaning else raw_query

    intent_result    = classify_intent(query_for_model)
    sentiment_result = analyze_sentiment(query_for_model)

    is_high_priority = (
        intent_result["intent"] in HIGH_PRIORITY_INTENTS
        or sentiment_result["sentiment"] == "Urgent"
        or (
            sentiment_result["sentiment"] == "Negative"
            and sentiment_result["confidence"] >= NEGATIVE_CONFIDENCE_THRESHOLD
            and intent_result["intent"] in ESCALATION_SENSITIVE_INTENTS
        )
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    result = {
        "raw_query"    : raw_query,
        "cleaned_query": query_for_model,
        "intent"       : intent_result,
        "sentiment"    : sentiment_result,
        "priority"     : "high" if is_high_priority else "normal",
        "latency_ms"   : latency_ms,
    }

    log.info(
        f"[{result['priority'].upper()}] intent={intent_result['intent']} "
        f"({intent_result['confidence']:.0%}) | "
        f"sentiment={sentiment_result['sentiment']} "
        f"({sentiment_result['confidence']:.0%}) | {latency_ms}ms"
    )
    return result


def process_batch(queries: list, apply_cleaning: bool = True) -> list:
    """Convenience wrapper for scoring a list of queries (used by evaluate_nlp.py)."""
    return [process_query(q, apply_cleaning=apply_cleaning) for q in queries]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    test_queries = [
        "I see a transaction of ₹10,000 I didn't make — this is urgent!",
        "My loan application has been pending for 3 weeks, very frustrated",
        "I cannot upload my KYC documents, need help please",
        "I am locked out of my account",
    ]

    print("\n" + "=" * 70)
    print("  NLP PIPELINE — TEST")
    print("=" * 70)

    for q in test_queries:
        r = process_query(q)
        print(f"\nQuery    : {r['raw_query']}")
        print(f"Intent   : {r['intent']['intent']} ({r['intent']['confidence']:.0%})")
        print(f"Sentiment: {r['sentiment']['sentiment']} ({r['sentiment']['confidence']:.0%})")
        print(f"Priority : {r['priority']} | {r['latency_ms']}ms")