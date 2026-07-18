"""
sentiment_analyzer.py
─────────────────────
Sentiment Analysis for BankBot-RAG.
Uses facebook/bart-large-mnli for zero-shot classification.
Labels: Urgent, Negative, Neutral
"""
from dotenv import load_dotenv
load_dotenv()
import logging
from transformers import pipeline

log = logging.getLogger(__name__)

# ── Labels ────────────────────────────────────────────────────────────────────
SENTIMENT_LABELS = [
    "The customer needs urgent immediate assistance",
    "The customer is upset, frustrated or anxious",
    "The customer is calm and seeking information",
]

SENTIMENT_MAP = {
    "The customer needs urgent immediate assistance" : "Urgent",
    "The customer is upset, frustrated or anxious"  : "Negative",
    "The customer is calm and seeking information"  : "Neutral",
}

NEGATIVE_SENTIMENTS = {"Urgent", "Negative"}

# ── Model (lazy load) ─────────────────────────────────────────────────────────
_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        log.info("Loading sentiment analyzer (facebook/bart-large-mnli)...")
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
        )
        log.info("Sentiment analyzer loaded ✅")
    return _classifier


# ── Main function ─────────────────────────────────────────────────────────────
def analyze_sentiment(query_text: str) -> dict:
    """
    Analyze sentiment of customer query.

    Args:
        query_text: Raw customer query string

    Returns:
        {
            "sentiment"  : "Urgent",
            "confidence" : 0.88,
            "is_negative": True,
            "all_scores" : {"Urgent": 0.88, ...}
        }
    """
    if not query_text or not query_text.strip():
        return {
            "sentiment"  : "Neutral",
            "confidence" : 0.0,
            "is_negative": False,
            "all_scores" : {},
        }

    classifier = _get_classifier()
    result = classifier(query_text, SENTIMENT_LABELS, multi_label=False)

    all_scores = {
        SENTIMENT_MAP[label]: round(score, 4)
        for label, score in zip(result["labels"], result["scores"])
    }
    top_sentiment = SENTIMENT_MAP[result["labels"][0]]
    confidence    = round(result["scores"][0], 4)
    is_negative   = top_sentiment in NEGATIVE_SENTIMENTS

    log.info(f"Sentiment: {top_sentiment} ({confidence:.2%})")
    return {
        "sentiment"  : top_sentiment,
        "confidence" : confidence,
        "is_negative": is_negative,
        "all_scores" : all_scores,
    }


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    test_queries = [
        "I see a transaction of ₹10,000 I didn't make — this is urgent!",
        "My loan application has been pending for 3 weeks, very frustrated",
        "I cannot upload my KYC documents, need help please",
        "I am locked out of my account",
    ]

    print("\n" + "=" * 60)
    print("  SENTIMENT ANALYZER — TEST")
    print("=" * 60)

    for query in test_queries:
        result = analyze_sentiment(query)
        print(f"\nQuery      : {query}")
        print(f"Sentiment  : {result['sentiment']}")
        print(f"Confidence : {result['confidence']:.2%}")
        print(f"Is Negative: {result['is_negative']}")
        print(f"All scores : {result['all_scores']}")