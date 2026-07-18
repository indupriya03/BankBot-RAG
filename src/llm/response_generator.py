"""
response_generator.py
───────────────────────
Groq-based response generation for BankBot-RAG.

Combines:
  - Customer query + detected intent/sentiment (src/nlp/nlp_pipeline.py)
  - Retrieved context (src/rag/retriever.py)
  - Fraud analysis (src/risk/transaction_risk_lookup.py)
  - Structured response template (src/llm/response_templates.py)

into a single grounded, structured, empathetic response.

Requires GROQ_API_KEY in .env — free at https://console.groq.com

Usage:
    from response_generator import generate_response
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv
from groq import Groq

# ── Import templates ──────────────────────────────────────────────────────────
# response_templates.py lives in the same folder (src/llm/)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from response_templates import get_template

load_dotenv()
log = logging.getLogger(__name__)


def _load_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)

_params = _load_params()

MODEL       = _params["llm"]["model"]
MAX_RETRIES = _params["llm"]["max_retries"]

# Number of prior turns sent to the LLM as conversation history
# (each exchange = 2 turns, so this value / 2 = exchanges retained)
MAX_HISTORY_TURNS = _params["llm"]["max_history_turns"]

# ── System prompt — short and clean ──────────────────────────────────────────
SYSTEM_PROMPT = """You are BankBot, SwiftBank's AI banking assistant. You are
the primary channel for support — decisive and capable, not a call-router.

CORE RULES:
1. Answer ONLY using the provided context — never invent policy details,
   amounts, timelines, or procedures not in the context. If the context
   doesn't cover the question, say so honestly rather than guessing.
1b. Be especially careful with SPECIFIC, CHECKABLE details — phone
    numbers, verification steps, exact timeframes. If a specific number
    or procedure isn't explicitly stated in the context, describe the
    action in general terms instead of inventing a plausible-sounding
    specific (e.g. say "verify your identity" rather than naming a
    verification method not in the context).
1b. Be especially careful with SPECIFIC, CHECKABLE details — phone
    numbers, verification steps, exact timeframes. If a specific number
    or procedure isn't explicitly stated in the context, describe the
    action in general terms instead of inventing a plausible-sounding
    specific (e.g. say "verify your identity" rather than naming a
    verification method not in the context).
2. Follow the RESPONSE STRUCTURE provided in the user message exactly —
   it tells you what steps to include and in what order.
3. Generate natural, empathetic text within that structure — don't copy
   the template verbatim, make it warm and conversational.

HOW YOU HANDLE ACTIONS (block card, raise dispute, unlock account, etc.):
- You CAN take these actions — you're not limited to giving directions.
- For anything actionable, PROPOSE the action and ask ONE clear yes/no
  confirmation question in the same turn (the template will tell you
  which action applies). Do not execute it yourself in this response —
  just propose it decisively and ask for the go-ahead.
- Do not say you "cannot" perform account actions, and do not default to
  listing helpline numbers or branch visits as the first answer — only
  mention those as a fallback if the template explicitly calls for it.

CONVERSATION RULES:
- Never ask the customer to repeat information already given in this conversation.
- Simple acknowledgments ("thanks", "ok", "got it") → respond briefly and
  warmly. Do not re-explain policy or recommend escalation again.
- Short follow-ups ("what should I do now?") → interpret in light of prior
  turns, not as an isolated question.
- Never repeat the same helpline/channel instruction in back-to-back turns.
- If the customer pastes raw CSV/comma-separated transaction data → extract
  the relevant fields (TXN ID, amount, merchant, date) naturally and use
  them in your response. Do not echo the raw data back.

Respond with a JSON object with EXACTLY these four fields, no other text:
{
  "response": "customer-facing reply — warm, empathetic, following the provided structure",
  "risk_level": "High" | "Medium" | "Low",
  "suggested_action": "concrete next action for a human agent or system",
  "grounded_in": ["exact chunk_id values ONLY — copy them verbatim from the 'chunk_id=' field shown in each context block. NEVER use section numbers, clause numbers, or any other number that appears inside the document text itself (e.g. '4.5' from '4.5 Business Loan' is NOT a chunk_id)"]
}"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_context_block(retrieved_chunks: list) -> str:
    """Format retrieved chunks for the LLM prompt."""
    blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        ref  = meta.get("policy_ref") or meta.get("document") or "?"
        blocks.append(
            f"[{i}] chunk_id={chunk.get('chunk_id')} "
            f"risk_level={meta.get('risk_level', '?')} ref={ref}\n"
            f"{chunk.get('text', '')}"
        )
    return "\n\n".join(blocks) if blocks else "(no relevant context retrieved)"


def _build_fraud_block(fraud_result: dict) -> str:
    """Format fraud analysis for the LLM prompt."""
    if not fraud_result:
        return ""

    if fraud_result.get("found") is True:
        return (
            f"\nFraud Analysis (USE THESE EXACT DETAILS IN YOUR RESPONSE):\n"
            f"  Transaction ID   : {fraud_result['transaction_id']}\n"
            f"  Amount           : ₹{fraud_result['amount_inr']:,.2f}\n"
            f"  Merchant         : {fraud_result['merchant_name']}\n"
            f"  Timestamp        : {fraud_result.get('timestamp', 'N/A')}\n"
            f"  Fraud Probability: {fraud_result['fraud_probability']:.2%}\n"
            f"  Risk Level       : {fraud_result['risk_level']}\n"
            f"  → Use this risk level to select the correct structured "
            f"response format from the RESPONSE STRUCTURE below.\n"
        )

    if fraud_result.get("found") == "needs_selection":
        lines = [
            f"\nAccount Found — Recent Transactions (ask customer which one looks wrong):\n"
            f"  Account ID: {fraud_result.get('account_id')}\n"
        ]
        for txn in fraud_result.get("recent_transactions", [])[:5]:
            lines.append(
                f"  • {txn['transaction_id']} | "
                f"₹{txn['amount_inr']:,.2f} | "
                f"{txn['merchant_name']} | "
                f"{txn['timestamp']}\n"
            )
        lines.append(
            "  → Ask the customer which transaction looks wrong so you "
            "can investigate further.\n"
        )
        return "".join(lines)

    return ""


def _normalize_history_turn(turn: dict) -> Optional[dict]:
    """
    Normalize a history turn for the LLM messages array.
    Assistant turns in session_state may be the full result dict —
    extract just the response text for the LLM.
    """
    role    = turn.get("role")
    content = turn.get("content")
    if role not in ("user", "assistant"):
        return None
    if isinstance(content, dict):
        content = content.get("response", "")
    if not content or not str(content).strip():
        return None
    return {"role": role, "content": str(content)}


def _build_messages(
    conversation_history: Optional[list],
    user_prompt: str,
) -> list:
    """Build the full messages array for the Groq API call."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        normalized = [
            t for t in (
                _normalize_history_turn(turn)
                for turn in conversation_history
            )
            if t
        ]
        messages.extend(normalized[-MAX_HISTORY_TURNS:])

    messages.append({"role": "user", "content": user_prompt})
    return messages


# ── Main function ─────────────────────────────────────────────────────────────
def generate_response(
    query: str,
    intent: dict,
    sentiment: dict,
    retrieved_chunks: list,
    fraud_result: dict = None,
    conversation_history: list = None,
    model: str = MODEL,
    loan_already_flagged: bool = False,
    kyc_already_addressed: bool = False,
    account_compromise: bool = False,
    account_hard_lock: bool = False,
    qa_match: dict = None,
) -> dict:
    """
    Generate a structured, grounded response using Groq LLM.

    Args:
        query               : raw customer query text
        intent              : {"intent": ..., "confidence": ...}
        sentiment           : {"sentiment": ..., "confidence": ...}
        retrieved_chunks    : output of Retriever.retrieve()
        fraud_result        : output of TransactionRiskScorer (optional)
        conversation_history: prior turns as list of {"role", "content"} dicts
        model               : Groq model name
        loan_already_flagged : loan was already flagged for manual review
            earlier in this conversation (from orchestrator.py's history scan)
        kyc_already_addressed: KYC guidance was already given earlier in
            this conversation
        account_compromise    : current message suggests possible unauthorized
            access rather than a routine lockout
        account_hard_lock     : current message suggests a compliance/
            restriction hold that can't be resolved by self-service unlock
        qa_match              : output of get_qa_match(retrieved_chunks) —
            if present, risk_level/suggested_action are taken directly from
            this QA pair's metadata rather than the LLM's own guess, and
            "policy_reference" is added to the returned dict.

    Returns:
        {
            "response"        : "...",
            "risk_level"      : "High" | "Medium" | "Low",
            "suggested_action": "...",
            "grounded_in"     : [...],
            "policy_reference": "..." | None,
            "raw_model_output": "..."
        }
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GROQ_API_KEY in .env — free at https://console.groq.com"
        )

    client = Groq(api_key=api_key)

    # ── Build prompt components ───────────────────────────────────────────────
    context_block = _build_context_block(retrieved_chunks)
    fraud_block   = _build_fraud_block(fraud_result)

    # ── Get structured template ───────────────────────────────────────────────
    risk_for_template = (
        fraud_result.get("risk_level", "Low")
        if fraud_result and fraud_result.get("found") is True
        else "Low"
    )
    template, qa_grounded = get_template(
        intent=intent.get("intent", ""),
        risk_level=risk_for_template,
        fraud_result=fraud_result,
        loan_already_flagged=loan_already_flagged,
        kyc_already_addressed=kyc_already_addressed,
        account_compromise=account_compromise,
        account_hard_lock=account_hard_lock,
        query=query,
        qa_match=qa_match,
    )

    # ── Assemble user prompt ──────────────────────────────────────────────────
    template_block = (
        f"\nRESPONSE STRUCTURE TO FOLLOW:\n{template}\n"
        if template else ""
    )

    user_prompt = (
        f"Customer query: {query}\n"
        f"Detected intent: {intent.get('intent', 'Unknown')} "
        f"(confidence {intent.get('confidence', 0):.0%})\n"
        f"Detected sentiment: {sentiment.get('sentiment', 'Unknown')} "
        f"(confidence {sentiment.get('confidence', 0):.0%})\n"
        f"{fraud_block}"
        f"{template_block}"
        f"\nRetrieved context:\n{context_block}"
    )

    messages = _build_messages(conversation_history, user_prompt)

    # ── Call Groq with retry ──────────────────────────────────────────────────
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=_params["llm"]["temperature"],
                response_format={"type": "json_object"},
            )
            raw    = completion.choices[0].message.content
            parsed = json.loads(raw)
            parsed.setdefault("response", "")
            parsed.setdefault("risk_level", "Medium")
            parsed.setdefault("suggested_action", "Escalate to human agent")
            parsed.setdefault("grounded_in", [])
            parsed["raw_model_output"] = raw

            # QA-pair metadata is ground truth ONLY when it's actually what
            # produced this response (qa_grounded). If a deterministic
            # action flow fired instead (a real fraud score, a confirmed
            # lockout, etc.), a QA chunk that happened to also be retrieved
            # must NOT override that flow's own risk_level/suggested_action
            # — e.g. a real 99%-probability fraud alert should never be
            # silently downgraded by a coincidentally-matched FAQ's static
            # risk_level.
            if qa_grounded and qa_match:
                parsed["risk_level"]       = qa_match["risk_level"]
                parsed["suggested_action"] = qa_match["suggested_action"] or parsed["suggested_action"]
                if qa_match["chunk_id"] not in parsed["grounded_in"]:
                    parsed["grounded_in"].append(qa_match["chunk_id"])
                parsed["policy_reference"] = qa_match["policy_ref"]
            else:
                parsed["policy_reference"] = None

            return parsed

        except json.JSONDecodeError as e:
            log.warning(
                f"Non-JSON output (attempt {attempt+1}/{MAX_RETRIES}): {e}"
            )
            last_error = e

        except Exception as e:
            wait = 2 ** attempt
            log.warning(
                f"Groq call failed (attempt {attempt+1}/{MAX_RETRIES}): {e}. "
                f"Retrying in {wait}s..."
            )
            last_error = e
            time.sleep(wait)

    # ── Fallback response ─────────────────────────────────────────────────────
    log.error(f"All {MAX_RETRIES} attempts failed — returning fallback.")
    return {
        "response": (
            "I'm having trouble generating a response right now. "
            "Your query has been noted and will be reviewed by a support "
            "agent shortly. You can also reach us at 1800-123-4567 (24x7)."
        ),
        "risk_level"      : "Medium",
        "suggested_action": "Escalate to human agent — response generation failed",
        "grounded_in"     : [],
        "policy_reference": None,
        "raw_model_output": None,
        "error"           : str(last_error),
    }


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    mock_intent    = {"intent": "Fraud/Unauthorized", "confidence": 0.92}
    mock_sentiment = {"sentiment": "Urgent", "confidence": 0.71}
    mock_chunks    = [
        {
            "chunk_id": "qa_QA001",
            "text": (
                "Question: I noticed an unauthorized transaction of ₹15,000. "
                "What should I do?\n"
                "Answer: Please report immediately via our 24x7 helpline or app. "
                "We will block your card and raise a dispute within 1 working hour."
            ),
            "metadata": {
                "risk_level"       : "High",
                "policy_ref"       : "Fraud Handling Policy §3, §4",
                "suggested_action" : "Block card, raise dispute, escalate to fraud team",
            },
        },
    ]
    mock_fraud = {
        "found"              : True,
        "transaction_id"     : "TXN500004",
        "account_id"         : "ACC123456",
        "amount_inr"         : 41478.01,
        "merchant_name"      : "VPN_Service_Anon",
        "timestamp"          : "2023-06-15 02:34:00",
        "fraud_probability"  : 0.9939,
        "risk_level"         : "High",
        "actual_fraud_label" : 1,
    }

    result = generate_response(
        query="My transaction TXN500004 looks wrong",
        intent=mock_intent,
        sentiment=mock_sentiment,
        retrieved_chunks=mock_chunks,
        fraud_result=mock_fraud,
    )

    print("\n" + "=" * 60)
    print("  RESPONSE GENERATOR — SMOKE TEST")
    print("=" * 60)
    print(f"Response        :\n{result['response']}")
    print(f"\nRisk Level      : {result['risk_level']}")
    print(f"Suggested Action: {result['suggested_action']}")
    print(f"Grounded In     : {result['grounded_in']}")