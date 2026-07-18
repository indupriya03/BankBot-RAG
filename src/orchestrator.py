"""
orchestrator.py
─────────────────
Ties the full BankBot-RAG pipeline together:

    query -> intent + sentiment (src/nlp) 
          -> fraud classifier (src/risk) [if Fraud intent + TXN ID found]
          -> retrieval (src/rag) 
          -> response (src/llm)

Usage:
    python src/orchestrator.py --query "I see a transaction I didn't make"
    python src/orchestrator.py --query "My transaction TXN500004 looks wrong"
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional


_SRC_DIR = Path(__file__).resolve().parent
for _sub in ("nlp", "rag", "llm", "risk"):
    _path = str(_SRC_DIR / _sub)
    if Path(_path).exists() and _path not in sys.path:
        sys.path.insert(0, _path)

# Also add project root for src.data imports
sys.path.insert(0, str(_SRC_DIR.parent))

import re

from nlp_pipeline import process_query                          # src/nlp
from retriever import Retriever                                  # src/rag
from response_generator import generate_response                 # src/llm
from transaction_risk_lookup import TransactionRiskScorer        # src/risk
from response_templates import (
    get_pending_action, execute_action, decline_action,
    account_unlock_offer_after_verification, account_verification_declined,
    acknowledgment_reply, get_qa_match, QA_CATEGORY_TO_INTENT,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


RETRIEVAL_CONTEXT_TURNS = 2
INTENT_STICKY_CONFIDENCE_THRESHOLD = 0.60

# Keyword routing for lost/stolen card — this is a distinct action (block
# regardless of transaction history) so it must not fall through the
# generic fraud/unauthorized-transaction path.
LOST_STOLEN_KEYWORDS = [
    "lost my card", "card is lost", "i lost my card", "misplaced my card",
    "stolen card", "card was stolen", "card got stolen", "someone stole my card",
]

# Confirming/declining a pending action. Anchored to the OPENING word only —
# "yes i am sure i did not make this transaction" should count as a
# confirmation even though it's 9 words long, because the "not" refers to
# not having made the transaction, not to declining the proposed action.
# We only treat it as a decline if an explicit cancel/contradiction phrase
# shows up (checked separately, and it wins over a leading "yes").
_CONFIRM_PATTERN = re.compile(
    r"^(yes|yeah|yep|yup|sure|confirm|confirmed|go ahead|please (do|proceed|block|unlock)|do it|okay|ok|correct|please|that'?s right|correct,? yes)\b",
    re.IGNORECASE,
)
_DECLINE_PATTERN = re.compile(
    r"^(no|nope|nah|don'?t|do not|cancel|not now|wait|hold on)\b",
    re.IGNORECASE,
)
# Explicit contradiction anywhere in the message overrides a leading "yes" —
# e.g. "yes but actually cancel that" or "yes wait no don't block it".
_DECLINE_OVERRIDE = re.compile(
    r"\b(actually,? no|wait,? no|don'?t (do it|block|proceed|unlock)|do not (do it|block|proceed|unlock)|cancel that|never ?mind)\b",
    re.IGNORECASE,
)
# Cap on message length so a confirmation check doesn't fire on an
# unrelated new question that happens to start with "ok" — a genuine
# yes/no reply to a proposed action is rarely more than ~20 words.
_MAX_CONFIRMATION_WORDS = 20

# Signals that a lockout may be from unauthorized access rather than a
# routine forgotten-password lockout — routed straight to a security-lock
# confirmation instead of the identity-verify-then-unlock flow.
ACCOUNT_COMPROMISE_KEYWORDS = [
    "hacked", "someone accessed", "someone logged", "not me", "wasn't me",
    "don't recognize this login", "unfamiliar activity", "unfamiliar login",
    "suspicious activity", "suspicious login", "unauthorized login",
    "unauthorised login", "someone else logged in", "my account was accessed",
]


def _is_account_compromise(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in ACCOUNT_COMPROMISE_KEYWORDS)


# Signals that the account isn't just password-locked but under a
# compliance/restriction hold — these genuinely can't be fixed by a
# self-service unlock, so the bot shouldn't ask for verification and
# then falsely offer to resolve it itself.
ACCOUNT_HARD_LOCK_KEYWORDS = [
    "account frozen", "account is frozen", "account restricted",
    "account is restricted", "under review", "account closed",
    "account suspended", "compliance hold", "account blocked by the bank",
    "account blocked by bank", "flagged by compliance",
]


def _is_hard_account_lock(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in ACCOUNT_HARD_LOCK_KEYWORDS)


def _loan_already_flagged(conversation_history: list) -> bool:
    """True if a manual-review flag was already confirmed earlier in this
    conversation — used to avoid re-offering the same action on a
    follow-up status question."""
    if not conversation_history:
        return False
    for turn in conversation_history:
        if turn.get("role") == "assistant":
            content = turn.get("content")
            if isinstance(content, dict):
                sa = content.get("suggested_action") or ""
                if "Loan flagged for manual review" in sa:
                    return True
    return False


def _kyc_guidance_already_given(conversation_history: list) -> bool:
    """True if a KYC-intent response was already given earlier in this
    conversation — used to escalate instead of repeating the same
    upload guidance on a follow-up."""
    if not conversation_history:
        return False
    for turn in conversation_history:
        if turn.get("role") == "assistant":
            content = turn.get("content")
            if isinstance(content, dict):
                turn_intent = (content.get("intent") or "").lower()
                if "kyc" in turn_intent:
                    return True
    return False


# Simple, closing-the-loop acknowledgments ("ok", "thanks", "ok thank you").
# These carry no topical signal at all — running them through intent
# classification → retrieval → the LLM not only wastes a call, it risks
# the wrong stale template (loan/KYC/fraud) getting reattached and the
# LLM re-proposing something already resolved. So, same as confirm/decline,
# they're caught here and answered directly, before any of that runs.
# Kept as an explicit set rather than a fuzzy regex — the space of genuine
# "just closing out" replies is small and bounded, and being explicit
# avoids accidentally swallowing a real follow-up question that happens
# to start with "ok" (e.g. "ok so how long will the loan review take").
_ACK_PHRASES = {
    "ok", "okay", "k", "kk",
    "thanks", "thank you", "thanks a lot", "thank you so much",
    "thanks so much", "thank you very much", "many thanks",
    "ok thanks", "okay thanks", "ok thank you", "okay thank you",
    "great thanks", "great thank you", "perfect thanks", "perfect thank you",
    "alright thanks", "cool thanks", "got it thanks", "got it thank you",
    "got it", "noted", "understood", "sounds good", "sounds good thanks",
    "appreciate it", "appreciate that", "great", "perfect", "cool",
    "alright", "good", "nice",
    "nothing thanks", "no nothing else", "thats all thanks","no thanks",
    "nothing", "no", "nope", "none","nah thanks", "nope thanks",
}


def _is_simple_acknowledgment(text: str) -> bool:
    t = re.sub(r"[^\w\s]", "", text.strip().lower())
    t = re.sub(r"\s+", " ", t).strip()
    if not t or len(t.split()) > 6:
        return False
    return t in _ACK_PHRASES


def _get_previous_intent(conversation_history: list) -> Optional[str]:
    """Find the intent of the most recent assistant turn, if tracked."""
    if not conversation_history:
        return None
    for turn in reversed(conversation_history):
        if turn.get("role") == "assistant":
            content = turn.get("content")
            if isinstance(content, dict):
                return content.get("intent")
            return None
    return None


def _get_pending_action(conversation_history: list) -> Optional[dict]:
    """
    Requires streamlit_app.py to store "pending_action" and "fraud_result" on
    the assistant message content dict (see streamlit_app.py's run_query()).
    """
    if not conversation_history:
        return None
    for turn in reversed(conversation_history):
        if turn.get("role") == "assistant":
            content = turn.get("content")
            if isinstance(content, dict) and content.get("pending_action"):
                return {
                    "action" : content["pending_action"],
                    "context": {"fraud_result": content.get("fraud_result")},
                }
            return None
    return None


def _is_confirmation(text: str) -> bool:
    t = text.strip()
    if len(t.split()) > _MAX_CONFIRMATION_WORDS:
        return False
    if _DECLINE_OVERRIDE.search(t.lower()):
        return False
    return bool(_CONFIRM_PATTERN.match(t))


def _is_decline(text: str) -> bool:
    t = text.strip()
    if len(t.split()) > _MAX_CONFIRMATION_WORDS:
        return False
    return bool(_DECLINE_PATTERN.match(t)) or bool(_DECLINE_OVERRIDE.search(t.lower()))


def _is_lost_or_stolen(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in LOST_STOLEN_KEYWORDS)


def _extract_text(content) -> str:
    """Assistant turns in conversation_history may be the full result dict
    (from streamlit_app.py's session state) or plain text — normalize to text."""
    if isinstance(content, dict):
        return content.get("response", "")
    return str(content) if content else ""


def _build_retrieval_query(query: str, conversation_history: list = None,
                           fraud_result: dict = None) -> str:
    """
    Bare short follow-ups ("what should I do now?") carry almost no topical
    signal for embedding-based retrieval on their own. Prepending the last
    couple of USER turns (not assistant — we want what the customer said,
    not to re-retrieve based on the bot's own prior answer) gives the
    embedding something real to match against.
    """
    parts = []

    if conversation_history:
        recent_user_texts = [
            _extract_text(t.get("content"))
            for t in conversation_history
            if t.get("role") == "user"
        ]
        recent_user_texts = [t for t in recent_user_texts if t.strip()][-RETRIEVAL_CONTEXT_TURNS:]
        if recent_user_texts:
            parts.append("Recent conversation: " + " | ".join(recent_user_texts))

    parts.append(f"Current message: {query}")

    if fraud_result and fraud_result.get("found") is True:
        parts.append(
            f"fraud_probability={fraud_result['fraud_probability']:.0%} "
            f"risk={fraud_result['risk_level']}"
        )

    return " ".join(parts)


class BankBotOrchestrator:
    """
    Holds expensive-to-init components (Retriever, TransactionRiskScorer)
    so repeated queries don't pay the init cost twice.
    """

    def __init__(
        self,
        persist_dir: str = "data/vector_store/chroma_db",
        retrieval_k: int = 5,
        fraud_model_path: str = "models/fraud/production/production_fraud_model.pkl",
    ):
        log.info("Initializing orchestrator...")

        # RAG retriever
        self.retriever   = Retriever(persist_dir=persist_dir)
        self.retrieval_k = retrieval_k

        # Fraud classifier (optional — graceful fallback if model not found)
        try:
            self.fraud_scorer = TransactionRiskScorer(model_path=fraud_model_path)
            log.info("Fraud classifier loaded ✅")
        except FileNotFoundError:
            self.fraud_scorer = None
            log.warning(
                "Fraud classifier model not found — "
                "fraud scoring will be skipped. "
                "Run src/risk/run_risk_pipeline.py to train."
            )

        log.info("Orchestrator ready ✅")

    def _build_action_result(self, query: str, outcome: dict, context: dict, t_start: float,
                              pending_action: Optional[str] = None,
                              intent_label: str = "Action Confirmation") -> dict:
        """
        Build a result dict for a confirm/decline/acknowledgment turn. No
        NLP, fraud scoring, retrieval, or LLM call — either the decision
        was already made on a previous turn (confirm/decline) or there's
        no decision to make at all (acknowledgment). Keeps the response
        deterministic and instant.

        pending_action: normally None (the prior pending action is now
        resolved), but for multi-step flows like verify-then-unlock, this
        carries the NEW pending state forward to the next turn.
        intent_label: what to tag this turn's "intent" as in the result —
        distinguishes a resolved action from a plain acknowledgment for
        logging/debugging purposes.
        """
        total_ms = round((time.perf_counter() - t_start) * 1000, 1)
        result = {
            "query"           : query,
            "intent"          : {"intent": intent_label, "confidence": 1.0},
            "sentiment"       : {"sentiment": "Neutral", "confidence": 1.0},
            "priority"        : "high" if outcome["risk_level"] == "High" else "normal",
            "fraud_result"    : context.get("fraud_result"),
            "pending_action"  : pending_action,
            "retrieved_chunks": [],
            "response"        : outcome["response"],
            "risk_level"      : outcome["risk_level"],
            "suggested_action": outcome["suggested_action"],
            "grounded_in"     : [],
            "latency_ms"      : {"nlp": 0.0, "fraud": 0.0, "retrieval": 0.0, "generation": 0.0, "total": total_ms},
        }
        log.info(f"[ACTION] intent={intent_label} suggested_action={result['suggested_action']} total={total_ms}ms")
        return result

    def handle_query(self, query: str, k: int = None, conversation_history: list = None) -> dict:
        """
        Full pipeline for one customer query.

        Args:
            query: raw customer message
            k: retrieval top-k (defaults to self.retrieval_k)
            conversation_history: prior turns as [{"role": "user"/"assistant", "content": ...}, ...],
                oldest first, NOT including the current query. Assistant turns can be either
                plain text or the full result dict this method returns (auto-extracted where
                needed). Used to (a) enrich the retrieval query with recent topic context, and
                (b) give the LLM actual multi-turn memory instead of treating each message as
                an isolated conversation.

        Returns:
            {
                "query"           : "...",
                "intent"          : {"intent": ..., "confidence": ...},
                "sentiment"       : {"sentiment": ..., "confidence": ...},
                "priority"        : "high" | "normal",
                "fraud_result"    : {...} | None,
                "retrieved_chunks": [...],
                "response"        : "...",
                "risk_level"      : "High" | "Medium" | "Low",
                "suggested_action": "...",
                "grounded_in"     : [...],
                "latency_ms"      : {...},
            }
        """
        k = k or self.retrieval_k
        t_start = time.perf_counter()

        # ── Stage 0: is this turn a confirm/decline of a pending action? ─────
        # If the previous turn proposed an action (e.g. "shall I block your
        # card?"), a short "yes"/"no" reply here should NOT go through the
        # full NLP → fraud → retrieval → LLM pipeline. It's pure application
        # logic — execute or cancel the action deterministically and return.
        pending = _get_pending_action(conversation_history)
        if pending:
            # Special case: "account_verify_identity" isn't a yes/no ACTIONS
            # key — it's the identity-check step that must happen before an
            # unlock is even offered. A decline cancels the flow outright;
            # anything else is treated as the verification answer, which
            # advances (deterministically, no LLM) to the actual unlock
            # offer — which THEN goes through the normal yes/no flow.
            if pending["action"] == "account_verify_identity":
                if _is_decline(query):
                    outcome = account_verification_declined()
                    return self._build_action_result(query, outcome, pending["context"], t_start)
                outcome = account_unlock_offer_after_verification()
                return self._build_action_result(
                    query, outcome, pending["context"], t_start,
                    pending_action="unlock_account",
                )
            if _is_confirmation(query):
                outcome = execute_action(pending["action"], pending["context"])
                return self._build_action_result(query, outcome, pending["context"], t_start)
            if _is_decline(query):
                outcome = decline_action(pending["action"])
                return self._build_action_result(query, outcome, pending["context"], t_start)
            # Anything else (a new question, a change of topic) falls through
            # to the normal pipeline — the pending action simply lapses.

        # ── Stage 0.5: simple acknowledgment ("ok", "thanks", "got it") ──────
        # No topical content here at all, so — same reasoning as Stage 0 —
        # skip intent classification, retrieval, and the LLM entirely.
        # This also sidesteps the failure mode where a stale loan/KYC/fraud
        # template gets reattached to a message like "ok thank you" and the
        # LLM re-proposes something that was already resolved.
        if _is_simple_acknowledgment(query):
            outcome = acknowledgment_reply()
            return self._build_action_result(
                query, outcome, {}, t_start, intent_label="Acknowledgment",
            )

        # ── Stage 1: NLP — intent + sentiment ────────────────────────────────
        t0 = time.perf_counter()
        nlp_result = process_query(query)
        nlp_ms     = round((time.perf_counter() - t0) * 1000, 1)

        intent    = nlp_result["intent"]
        sentiment = nlp_result["sentiment"]

        if intent["confidence"] < INTENT_STICKY_CONFIDENCE_THRESHOLD:
            previous_intent = _get_previous_intent(conversation_history)
            if previous_intent:
                log.info(
                    f"Low-confidence intent ({intent['intent']} @ {intent['confidence']:.0%}) "
                    f"— sticking with previous intent '{previous_intent}'"
                )
                intent = {**intent, "intent": previous_intent}

        # Lost/stolen card is routed by keyword, not by the zero-shot
        # classifier — it's a distinct action (block regardless of any
        # transaction) and shouldn't be folded into the general fraud path.
        is_lost_stolen_locked = _is_lost_or_stolen(query)
        if is_lost_stolen_locked:
            intent = {**intent, "intent": "Lost/Stolen Card", "confidence": max(intent["confidence"], 0.95)}

        # ── Stage 2: RAG retrieval (moved before the fraud stage) ────────────
        # A confident QA-pair match is a much more specific signal than the
        # coarse 4-way zero-shot intent label, so retrieval needs to happen
        # BEFORE we decide whether to run the fraud scorer — otherwise a
        # misclassification (e.g. "salary not credited" tagged as Fraud)
        # sends the query down the fraud info-gathering path before we ever
        # see that the top retrieved chunk is actually qa_QA016 / Account
        # Access. This first pass doesn't have fraud details yet to enrich
        # the query with — that's fine, it's re-run below once fraud_result
        # is known, if fraud is actually confirmed.
        t0 = time.perf_counter()
        retrieval_query   = _build_retrieval_query(query, conversation_history, fraud_result=None)
        retrieved_chunks  = self.retriever.retrieve(retrieval_query, k=k)
        retrieval_ms      = round((time.perf_counter() - t0) * 1000, 1)

        qa_match = get_qa_match(retrieved_chunks)
        if qa_match and not is_lost_stolen_locked:
            # The corpus has no dedicated "lost/stolen" QA category, so the
            # closest retrieved chunk is always a fraud-category QA pair —
            # letting this override fire here would silently undo the
            # keyword lock above and drop the query back into the generic
            # fraud info-gathering flow instead of the immediate-block flow.
            qa_category   = qa_match["category"].strip().lower()
            mapped_intent = QA_CATEGORY_TO_INTENT.get(qa_category)
            if mapped_intent and mapped_intent.lower() != intent["intent"].lower():
                log.info(
                    f"QA-pair match ({qa_match['chunk_id']}, category="
                    f"'{qa_match['category']}') overrides classifier intent "
                    f"'{intent['intent']}' → '{mapped_intent}'"
                )
                intent = {**intent, "intent": mapped_intent, "confidence": max(intent["confidence"], 0.9)}

        # ── Stage 3: Fraud Classifier (only for Fraud intent + TXN/ACC ID) ──
        t0           = time.perf_counter()
        fraud_result = None

        if (
            intent["intent"] == "Fraud/Unauthorized"
            and self.fraud_scorer is not None
        ):
            fraud_result = self.fraud_scorer.score_from_message(query)
            if fraud_result.get("found") is True:
                log.info(
                    f"Fraud score: {fraud_result['fraud_probability']:.2%} "
                    f"risk={fraud_result['risk_level']} "
                    f"txn={fraud_result['transaction_id']}"
                )
                # Re-retrieve with fraud details folded into the query now
                # that we have them — improves which policy chunk (high vs
                # medium/low risk section) comes back as the top match.
                t1 = time.perf_counter()
                retrieval_query  = _build_retrieval_query(query, conversation_history, fraud_result)
                retrieved_chunks = self.retriever.retrieve(retrieval_query, k=k)
                retrieval_ms    += round((time.perf_counter() - t1) * 1000, 1)
                qa_match = get_qa_match(retrieved_chunks)  # refresh in case top chunk changed
            elif fraud_result.get("found") == "needs_selection":
                log.info(
                    f"Account found but no TXN ID — "
                    f"{len(fraud_result['recent_transactions'])} recent txns returned"
                )
            else:
                log.info("No TXN/ACC ID in message — skipping fraud scoring")

        fraud_ms = round((time.perf_counter() - t0) * 1000, 1)

        # ── Conversation-state flags — avoid re-proposing/repeating things
        # already done or already said earlier in this conversation ──────────
        loan_already_flagged  = _loan_already_flagged(conversation_history)
        kyc_already_addressed = _kyc_guidance_already_given(conversation_history)
        account_compromise    = _is_account_compromise(query)
        account_hard_lock     = _is_hard_account_lock(query)

        # ── Stage 4: Response generation ──────────────────────────────────────
        t0 = time.perf_counter()
        gen_result = generate_response(
            query=query,
            intent=intent,
            sentiment=sentiment,
            retrieved_chunks=retrieved_chunks,
            fraud_result=fraud_result,   # passed to LLM for richer context
            conversation_history=conversation_history,
            loan_already_flagged=loan_already_flagged,
            kyc_already_addressed=kyc_already_addressed,
            account_compromise=account_compromise,
            account_hard_lock=account_hard_lock,
            qa_match=qa_match,
        )
        generation_ms = round((time.perf_counter() - t0) * 1000, 1)

        total_ms = round((time.perf_counter() - t_start) * 1000, 1)

        # Does this response propose an action the customer needs to
        # confirm? If so, tag it so the next turn's "yes"/"no" can be
        # routed straight to Stage 0 instead of the full pipeline.
        pending_action = get_pending_action(
            intent["intent"], gen_result["risk_level"], fraud_result,
            loan_already_flagged=loan_already_flagged,
            kyc_already_addressed=kyc_already_addressed,
            account_compromise=account_compromise,
            account_hard_lock=account_hard_lock,
            query=query,
            qa_match=qa_match,
        )

        result = {
            "query"           : query,
            "intent"          : intent,
            "sentiment"       : sentiment,
            "priority"        : nlp_result["priority"],
            "fraud_result"    : fraud_result,
            "pending_action"  : pending_action,
            "policy_reference": gen_result.get("policy_reference"),
            "retrieved_chunks": retrieved_chunks,
            "response"        : gen_result["response"],
            "risk_level"      : gen_result["risk_level"],
            "suggested_action": gen_result["suggested_action"],
            "grounded_in"     : gen_result["grounded_in"],
            "latency_ms"      : {
                "nlp"       : nlp_ms,
                "fraud"     : fraud_ms,
                "retrieval" : retrieval_ms,
                "generation": generation_ms,
                "total"     : total_ms,
            },
        }

        log.info(
            f"[{result['priority'].upper()}] "
            f"intent={intent['intent']} "
            f"sentiment={sentiment['sentiment']} "
            f"risk={result['risk_level']} "
            f"total={total_ms}ms "
            f"(nlp={nlp_ms} fraud={fraud_ms} "
            f"retrieval={retrieval_ms} gen={generation_ms})"
        )
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",       required=True)
    parser.add_argument("--k",           type=int, default=5)
    parser.add_argument("--persist-dir", default="data/vector_store/chroma_db")
    parser.add_argument("--fraud-model", default="models/fraud/production/production_fraud_model.pkl")    
    args = parser.parse_args()

    orchestrator = BankBotOrchestrator(
        persist_dir=args.persist_dir,
        retrieval_k=args.k,
        fraud_model_path=args.fraud_model,
    )
    result = orchestrator.handle_query(args.query)

    print("\n" + "=" * 70)
    print("  BANKBOT-RAG — FULL PIPELINE RESULT")
    print("=" * 70)
    print(f"Query           : {result['query']}")
    print(f"Intent          : {result['intent']['intent']} "
          f"({result['intent']['confidence']:.0%})")
    print(f"Sentiment       : {result['sentiment']['sentiment']} "
          f"({result['sentiment']['confidence']:.0%})")
    print(f"Priority        : {result['priority']}")

    if result["fraud_result"] and result["fraud_result"].get("found") is True:
        fr = result["fraud_result"]
        print(f"\nFraud Score     : {fr['fraud_probability']:.2%}")
        print(f"Fraud Risk Level: {fr['risk_level']}")
        print(f"Transaction     : {fr['transaction_id']} | ₹{fr['amount_inr']:,.2f} | {fr['merchant_name']}")

    print(f"\nRisk Level      : {result['risk_level']}")
    print(f"Suggested Action: {result['suggested_action']}")
    print(f"Grounded In     : {result['grounded_in']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nLatency         : {result['latency_ms']}")


if __name__ == "__main__":
    main()