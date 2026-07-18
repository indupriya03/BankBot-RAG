"""
intent_classifier.py
────────────────────
Intent Classification for BankBot-RAG.
Uses facebook/bart-large-mnli for zero-shot classification.
Labels: Fraud/Unauthorized, Loan, KYC, Account Access
"""
from dotenv import load_dotenv
load_dotenv()
import logging
from transformers import pipeline

log = logging.getLogger(__name__)

# ── Labels ────────────────────────────────────────────────────────────────────
INTENT_LABELS = [
    "a fraudulent or unauthorized transaction or suspicious payment",
    "a loan application, loan status or loan repayment",
    "a KYC verification issue or identity document upload",
    "unable to log in, forgotten password, account locked, OTP verification issue, "
    "or a general account servicing request such as balance, credits, transfers, "
    "or account maintenance",
]
INTENT_MAP = {
    "a fraudulent or unauthorized transaction or suspicious payment" : "Fraud/Unauthorized",
    "a loan application, loan status or loan repayment"   : "Loan",
    "a KYC verification issue or identity document upload": "KYC",
    "unable to log in, forgotten password, account locked, OTP verification issue, "
    "or a general account servicing request such as balance, credits, transfers, "
    "or account maintenance": "Account Access",
}
# ── Model (lazy load) ─────────────────────────────────────────────────────────
_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        log.info("Loading intent classifier (facebook/bart-large-mnli)...")
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
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
    result = classifier(
        query_text,
        INTENT_LABELS,
        hypothesis_template="This banking support request is about {}",
        multi_label=False
    )
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
        "I can't log into my account, it keeps saying invalid password",
        "looking to increase my loan amount, is that possible mid-term",
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