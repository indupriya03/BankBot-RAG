"""
intent_classifier.py
────────────────────
Intent Classification for BankBot-RAG.
Uses distilbert-base-uncased-mnli for zero-shot classification.
Labels: Fraud/Unauthorized, Loan, KYC, Account Access
"""

import logging
from transformers import pipeline

log = logging.getLogger(__name__)

# ── Labels ────────────────────────────────────────────────────────────────────
INTENT_LABELS = [
    "The customer is reporting a fraudulent or unauthorized transaction or payment",
    "The customer is asking about a loan application, loan status or loan repayment",
    "The customer has a KYC verification issue or needs to upload identity documents",
    "The customer cannot access their bank account or is locked out",
] 

INTENT_MAP = {
    "The customer is reporting a fraudulent or unauthorized transaction or payment" : "Fraud/Unauthorized",
    "The customer is asking about a loan application, loan status or loan repayment"        : "Loan",
    "The customer has a KYC verification issue or needs to upload identity documents"            : "KYC",
    "The customer cannot access their bank account or is locked out"            : "Account Access",
}

# ── Model (lazy load) ─────────────────────────────────────────────────────────
_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        log.info("Loading intent classifier (distilbert-base-uncased-mnli)...")
        _classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=-1,
        )
        log.info("Intent classifier loaded ✅")
    return _classifier


# ── Main function ─────────────────────────────────────────────────────────────
def classify_intent(query_text: str) -> dict:
    """
    Classify customer query intent.

    Args:
        query_text: Raw customer query string

    Returns:
        {
            "intent"     : "Fraud/Unauthorized",
            "confidence" : 0.92,
            "all_scores" : {"Fraud/Unauthorized": 0.92, ...}
        }
    """
    if not query_text or not query_text.strip():
        return {
            "intent"    : "Unknown",
            "confidence": 0.0,
            "all_scores": {},
        }

    classifier = _get_classifier()
    result = classifier(query_text, INTENT_LABELS, multi_label=False)

    all_scores = {
        INTENT_MAP[label]: round(score, 4)
        for label, score in zip(result["labels"], result["scores"])
    }
    top_intent    = INTENT_MAP[result["labels"][0]]
    confidence    = round(result["scores"][0], 4)

    log.info(f"Intent: {top_intent} ({confidence:.2%})")
    return {
        "intent"    : top_intent,
        "confidence": confidence,
        "all_scores": all_scores,
    }


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    test_queries = [
        "I see a transaction of ₹10,000 I didn't make",
        "My loan application has been pending for 3 weeks",
        "I cannot upload my KYC documents",
        "I am locked out of my account",
    ]

    print("\n" + "=" * 60)
    print("  INTENT CLASSIFIER — TEST")
    print("=" * 60)

    for query in test_queries:
        result = classify_intent(query)
        print(f"\nQuery     : {query}")
        print(f"Intent    : {result['intent']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"All scores: {result['all_scores']}")