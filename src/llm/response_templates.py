"""
response_templates.py
──────────────────────
Structured response templates for BankBot-RAG.

Design principle (v2):
  BankBot ACTS like a real bank's AI assistant. For anything it can
  reasonably resolve itself (blocking a card, raising a dispute,
  unlocking an account), it proposes the action and asks for a single
  yes/no confirmation — it does NOT dump helpline numbers and a
  5-step branch-visit checklist as the default answer.

  Two kinds of content live here:
  1. LLM INSTRUCTION TEMPLATES (get_template) — tell the LLM what
     structure/tone to use for the *proposal* turn.
  2. DETERMINISTIC ACTION CONFIRMATIONS (ACTIONS registry) — used by
     orchestrator.py directly (no LLM call) once the customer confirms.
     These are plain Python string templates — no ML/RAG needed here,
     this is just application logic executing a decision already made.
"""
import random
from typing import Optional

# ── Contacts — only surfaced for things BankBot genuinely can't do ──────────
CONTACTS = {
    "helpline_domestic": "1800-123-4567 (toll-free, 24x7)",
    "ombudsman_url"     : "https://cms.rbi.org.in",
}

# Maps a QA pair's dataset "category" field to the intent-classifier's
# label strings. Used to let a confident QA-pair retrieval match override
# a wrong classifier guess (e.g. "salary not credited" mistagged as Fraud
# by the coarse zero-shot classifier, but the top retrieved chunk is
# qa_QA016, category "Account Access").
QA_CATEGORY_TO_INTENT = {
    "fraud"          : "Fraud/Unauthorized",
    "loan"           : "Loan",
    "kyc"            : "KYC",
    "account access" : "Account Access",
}


def get_qa_match(retrieved_chunks: list) -> Optional[dict]:
    if not retrieved_chunks:
        return None
    for chunk in retrieved_chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        if chunk_id.lower().startswith("qa_"):
            meta = chunk.get("metadata", {})
            return {
                "chunk_id": chunk_id,
                "category": meta.get("category", ""),
                "policy_ref": meta.get("policy_ref", "Bank Policy"),
                "risk_level": meta.get("risk_level", "Low"),
                "suggested_action": meta.get("suggested_action", ""),
            }
    return None


# ══════════════════════════════════════════════════════════════════════════
#  1. LLM INSTRUCTION TEMPLATES — the "proposal" turn
# ══════════════════════════════════════════════════════════════════════════

def fraud_high_risk(txn_id: str, amount: str, merchant: str,
                    timestamp: str, fraud_pct: str) -> str:
    return f"""
RESPONSE TEMPLATE — FRAUD, HIGH RISK. ACT, DON'T REDIRECT.

Facts: transaction {txn_id}, ₹{amount} to {merchant} on {timestamp},
fraud probability {fraud_pct}.

Write 3-4 sentences, first person, decisive:
1. Name the transaction and say plainly this looks fraudulent.
2. Say you can block the card right now to stop further misuse.
3. Ask ONE clear yes/no question: "Shall I go ahead and block your card?"

Do NOT list helpline numbers, branch visits, or documents to carry —
you are the primary channel, not a router to one.
Set risk_level="High", suggested_action="Awaiting confirmation to block card".
""".strip()


def fraud_low_medium_risk(txn_id: str, amount: str, merchant: str,
                          timestamp: str, fraud_pct: str, risk: str) -> str:
    return f"""
RESPONSE TEMPLATE — FRAUD, {risk.upper()} RISK. ACT, DON'T REDIRECT.

Facts: transaction {txn_id}, ₹{amount} to {merchant} on {timestamp},
fraud probability {fraud_pct} ({risk} risk — not urgent enough to block
the card outright, but worth formally disputing).

Write 3-4 sentences, first person:
1. Name the transaction, note it looks {risk.lower()} risk.
2. Ask if they recognise the merchant / a family member's purchase / a
   recurring charge — give them a quick out if it's a false alarm.
3. If they don't recognise it, offer to raise a formal dispute now and
   ask ONE yes/no question: "Want me to raise a dispute on this transaction?"

Do NOT list helpline numbers or branch visits as the default path.
Set risk_level="{risk}", suggested_action="Awaiting confirmation to raise dispute".
""".strip()


def fraud_no_txn_id() -> str:
    return """
RESPONSE TEMPLATE — FRAUD REPORTED, NO TRANSACTION ID YET.

Write 2-3 sentences, first person:
1. Acknowledge the fraud concern briefly and seriously.
2. Ask for the transaction ID, amount, and merchant name (or just the
   account/card in question) so you can pull it up and assess risk.

Keep it short — this is just an information request, no action yet.
""".strip()


def fraud_account_found(account_id: str, recent_txns: list) -> str:
    txn_lines = "\n".join(
        f"  • {t['transaction_id']} | ₹{t['amount_inr']:,.2f} | "
        f"{t['merchant_name']} | {t['timestamp']}"
        for t in recent_txns[:5]
    )
    return f"""
RESPONSE TEMPLATE — ACCOUNT FOUND, NEED TRANSACTION SELECTION.

Account {account_id} found. Recent transactions:
{txn_lines}

Write 2 sentences: acknowledge the account, list the transactions above,
and ask which one looks wrong so you can pull its risk score.
""".strip()


def lost_stolen_card() -> str:
    return """
RESPONSE TEMPLATE — LOST OR STOLEN CARD. ACT IMMEDIATELY.

This is not a disputed-transaction case — the card itself is
compromised, so there's no ambiguity to resolve first.

Write 2-3 sentences, first person, urgent but calm:
1. Acknowledge the card is lost/stolen.
2. Say you can block it immediately and issue a replacement.
3. Ask ONE yes/no question: "Shall I block this card right now?"

Do NOT ask clarifying questions about specific transactions —
a lost/stolen card gets blocked regardless of transaction history.
Set risk_level="High", suggested_action="Awaiting confirmation to block card".
""".strip()


def kyc_support() -> str:
    return f"""
RESPONSE TEMPLATE — KYC ISSUE. GUIDE, DON'T LECTURE.

Write 3-4 sentences, first person, practical:
1. Acknowledge the specific KYC problem they described.
2. Give the 1-2 most likely fixes directly (e.g. "make sure the photo
   is under 2MB and well-lit" / "Aadhaar and PAN are the two documents
   that are mandatory").
3. Only if the fix might not be enough, mention they can also visit a
   branch or call {CONTACTS['helpline_domestic']} — as a fallback, not
   the headline.

Avoid a step-by-step numbered checklist unless the customer's message
suggests they're already stuck on multiple things.
""".strip()


def kyc_support_escalate() -> str:
    return """
RESPONSE TEMPLATE — KYC ISSUE, GUIDANCE ALREADY GIVEN, STILL FAILING.

You already gave the standard upload guidance (photo size/clarity,
Aadhaar/PAN) earlier in this conversation, and the customer is
indicating it's still not working. Do NOT repeat that guidance again —
repeating it a second time isn't helpful.

Write 2-3 sentences, first person:
1. Acknowledge the standard checks don't seem to be resolving it.
2. Offer to escalate this to a specialist for manual document review.
3. Ask ONE yes/no question: "Want me to escalate this for manual review?"

Set suggested_action="Awaiting confirmation to escalate KYC for manual review".
""".strip()


# The zero-shot intent classifier only gives us a coarse "Loan" bucket —
# it can't distinguish "where's my application" from "am I eligible."
# Those need different answers (the status template tells the LLM to
# skip criteria/details that an inquiry question actually needs), so we
# split by keyword here — same pattern as LOST_STOLEN_KEYWORDS etc. in
# orchestrator.py.
LOAN_INQUIRY_KEYWORDS = [
    "eligib",          # eligibility, eligible
    "criteria", "criterion",
    "qualify",
    "interest rate", "rate of interest",
    "which document",
    "how much can i", "how much loan", "how much amount",
    "maximum amount", "max amount",
    "tenure",
    "requirement",     # requirement, requirements
    "minimum income",
    "credit score", "cibil",
    "how do i apply", "how to apply", "how can i apply",
]


def _is_loan_inquiry(query: str) -> bool:
    q = (query or "").lower()
    if any(kw in q for kw in LOAN_INQUIRY_KEYWORDS):
        return True
    # "document(s) ... required/needed" is a very common phrasing but word
    # order varies ("documents required" vs "what documents are required
    # for" vs "document needed") — check word co-occurrence rather than a
    # fixed phrase, so order/singular-plural doesn't cause a silent miss.
    if "document" in q and ("require" in q or "need" in q):
        return True
    return False


def loan_inquiry() -> str:
    return """
RESPONSE TEMPLATE — LOAN PRODUCT QUESTION (eligibility, rates, documents,
tenure, amount limits). ANSWER FROM CONTEXT — NOT A STATUS CHECK.

This is a question about loan criteria/terms, not about tracking an
existing application. Answer it directly using the specific numbers and
criteria in the retrieved context (credit score, income, tenure, amount,
rate, documents, etc.) — do NOT redirect to "check status in-app" or
"Loans → Track Application"; there is no application to track here.

Write 2-4 sentences, first person:
1. Answer the specific question with the concrete criteria/figures from
   context.
2. If relevant, mention one closely related requirement (e.g. income
   proof if they asked about credit score).
3. If it fits naturally, close with the concrete next step: visiting any
   branch with the required documents (Aadhaar + PAN, income proof, bank
   statement) to apply.

Answer fully and stop — do NOT ask a follow-up yes/no question (e.g.
"want me to walk you through it?"). Do NOT mention
"Loans → Track Application" or "check status" anywhere — that phrase
belongs only to actual status-check queries.
""".strip()


def loan_support() -> str:
    return f"""
RESPONSE TEMPLATE — LOAN STATUS QUERY.

Write 2-3 sentences, first person:
1. Acknowledge the loan query.
2. Point them to checking status in-app (Loans → Track Application) as
   the fastest path.
3. If they mention a long delay (weeks), acknowledge that's longer than
   typical and offer to flag it for manual review rather than just
   repeating "check the app."

Skip the multi-step document checklist unless they ask what documents
are needed.
""".strip()


def loan_support_already_flagged() -> str:
    return """
RESPONSE TEMPLATE — LOAN STATUS QUERY, ALREADY FLAGGED FOR REVIEW.

This application was already flagged for manual review earlier in this
conversation — do NOT offer to flag it again.

Write 2 sentences, first person:
1. Remind them it's already with the underwriting team on manual review.
2. Give the expected timeline (2-3 business days via SMS/email), and
   mention they can also track status in-app in the meantime.

Do NOT ask "would you like me to flag it for review" — that's done.
""".strip()


# A confident QA-pair match under "Fraud" category could be either: (a) a
# genuine incident report ("I see a transaction I didn't make") — where we
# deliberately still want to ask for the transaction/account ID so the real
# fraud classifier can score it and the bot can take a personalized, decisive
# action — or (b) a general policy/FAQ question ("how long does fraud
# investigation take?", "is my account safe?") that has no specific
# transaction to look up and should just be answered from the QA match.
# This keyword list catches (a); everything else with a QA match falls
# through to the grounded answer instead of a generic "give me your txn ID".
FRAUD_INCIDENT_REPORT_KEYWORDS = [
    "transaction i didn't make", "transaction i did not make",
    "i didn't make this", "i did not make this",
    "don't recognize", "do not recognize", "dont recognize",
    "unauthorized transaction", "unauthorised transaction",
    "multiple small transactions", "multiple transactions",
    "looks wrong", "my transaction", "this transaction",
    "fraudulent charge", "wasn't me", "was not me","otp to transfer", 
    "used my otp", "otp was used", "someone used my otp",
]


def _is_fraud_incident_report(query: str) -> bool:
    q = (query or "").lower()
    return any(kw in q for kw in FRAUD_INCIDENT_REPORT_KEYWORDS)
LOAN_STATUS_COMPLAINT_KEYWORDS = [
    "pending for", "been pending", "no update", "no response",
    "hasn't been approved", "still waiting", "been waiting",
    "not received any update", "still pending",
]

def _is_loan_status_complaint(query: str) -> bool:
    q = (query or "").lower()
    return any(kw in q for kw in LOAN_STATUS_COMPLAINT_KEYWORDS)


def grounded_qa_answer(policy_ref: str) -> str:
    return f"""
RESPONSE TEMPLATE — INFORMATIONAL QUESTION, CONFIDENT POLICY/FAQ MATCH.

The retrieved context below is a confirmed, near-exact match for this
question — answer directly from it rather than improvising or following
a rigid script.

Write 2-4 sentences, first person:
1. Answer the specific question using the facts/figures in the retrieved
   context — nothing invented, nothing from outside it.
2. Naturally mention that this is based on our "{policy_ref}" — weave it
   into a sentence rather than bolting it on ("...as per our {policy_ref}"
   or "(per {policy_ref})").

Do NOT invent a "next action" or propose an action the retrieved content
doesn't call for — the correct risk_level and suggested_action for this
answer are supplied separately and will be used automatically; you only
need to write the customer-facing "response" text.
Do NOT ask a follow-up yes/no question unless the retrieved context
itself describes one.
""".strip()


# Only these phrasings genuinely mean "I can't get into my account" — a
# real lockout that might need identity verification + unlock/security-lock.
# Everything else tagged "Account Access" by the classifier (joint account
# questions, salary-credit questions, general account queries) should be
# answered informationally instead of being dragged into that flow — see
# get_template()'s account branch below.
ACCOUNT_LOCKOUT_KEYWORDS = [
    "locked out", "lock out", "can't log in", "cant log in", "cannot log in",
    "can't login", "cant login", "cannot login", "unable to log in",
    "unable to login", "net banking blocked", "netbanking blocked",
    "account is locked", "account locked", "login not working",
    "can't access my account", "cannot access my account",
]


def _is_account_lockout(query: str) -> bool:
    q = (query or "").lower()
    return any(kw in q for kw in ACCOUNT_LOCKOUT_KEYWORDS)


def account_access_informational() -> str:
    return f"""
RESPONSE TEMPLATE — ACCOUNT ACCESS, GENERAL QUESTION (NOT A LOCKOUT).

Nothing here indicates the customer is actually locked out — answer
their question directly from the retrieved context.

Write 2-3 sentences, first person, answering the specific question asked.
If the retrieved context doesn't clearly cover it, say so honestly and
suggest visiting a branch or calling {CONTACTS['helpline_domestic']} —
as a fallback, not the default.

Do NOT ask for identity verification or offer to unlock anything — this
isn't a lockout.
""".strip()


def account_access_verify() -> str:
    return """
RESPONSE TEMPLATE — ACCOUNT ACCESS ISSUE. VERIFY IDENTITY FIRST.

Nothing in the message suggests the account is compromised — this
looks like a routine lockout. Do NOT offer to unlock it yet; verify
identity first, since account changes shouldn't happen on request alone.

Write 2-3 sentences, first person:
1. Acknowledge the lockout and that you can help resolve it.
2. Explain you need to verify it's them first, for security.
3. Ask ONE verification question: "Can you confirm the last 4 digits of
   your registered mobile number or the email on the account?"

Do NOT ask "shall I unlock your account" in this turn.
Set suggested_action="Awaiting identity verification".
""".strip()


def account_access_compromise() -> str:
    return """
RESPONSE TEMPLATE — ACCOUNT ACCESS ISSUE, POSSIBLE COMPROMISE. ACT URGENTLY.

The customer's message suggests the lockout may be from unauthorized
access (hacked, unfamiliar login, activity they don't recognize) rather
than a routine lockout — treat this as urgent.

Write 3-4 sentences, first person, calm but urgent:
1. Acknowledge the lockout and that it may indicate unauthorized access.
2. Say you can apply a security lock right now to protect the account
   while it's reviewed — this is more appropriate here than a plain
   unlock.
3. Ask ONE yes/no question: "Shall I go ahead and apply a security lock?"

Set risk_level="High", suggested_action="Awaiting confirmation to apply security lock".
""".strip()


def account_access_hard_lock() -> str:
    return f"""
RESPONSE TEMPLATE — ACCOUNT ACCESS ISSUE, NOT SELF-SERVICE (compliance/restriction hold).

The customer's message suggests this is a frozen/restricted/under-review
account, not a routine password lockout. This type of lock genuinely
cannot be lifted by a simple unlock action — it needs manual review, so
do NOT ask for verification details or offer to unlock it yourself; that
would be misleading.

Write 3-4 sentences, first person, honest and direct:
1. Acknowledge the lockout.
2. Explain plainly that this looks like a compliance or security hold
   rather than a simple password lockout, so it isn't something you can
   fix directly — be upfront that this isn't a quick self-service case.
3. Tell them exactly what to do: contact a branch or call
   {CONTACTS['helpline_domestic']} for manual review, and that review
   typically takes a few business days.

Do NOT offer to unlock the account. Do NOT ask for verification details.
Set risk_level="Medium", suggested_action="Directed to branch/helpline — lock requires manual review, not self-service".
""".strip()


def account_unlock_offer_after_verification() -> dict:
    """
    Deterministic (non-LLM) response for the turn right after the customer
    answers the identity-verification question. No LLM call needed — this
    is just the next fixed step in the flow.

    Explains the likely reason for the lockout and that it's a quick,
    self-service fix, before asking to proceed — rather than just asking
    "shall I unlock it?" with no context.
    """
    return {
        "response": (
            "Thanks for confirming that. Lockouts like this are almost "
            "always from a few incorrect login attempts in a row rather "
            "than anything more serious, so it's a quick fix on my end — "
            "no branch visit needed. I can go ahead and unlock your "
            "account right now — shall I go ahead?"
        ),
        "risk_level"      : "Low",
        "suggested_action": "Awaiting confirmation to unlock account",
    }


def account_verification_declined() -> dict:
    return {
        "response": (
            "No problem — I won't make any changes without verifying it's "
            "you first. You're welcome to try again, or visit a branch or "
            f"call {CONTACTS['helpline_domestic']} to verify your identity "
            "another way."
        ),
        "risk_level"      : "Medium",
        "suggested_action": "No action taken — identity not verified",
    }


# ══════════════════════════════════════════════════════════════════════════
#  2. DETERMINISTIC ACTION CONFIRMATIONS — no LLM call, no hallucination risk
#     Used directly by orchestrator.py once the customer says "yes".
# ══════════════════════════════════════════════════════════════════════════

def _ref(prefix: str) -> str:
    return f"{prefix}-{random.randint(1000, 9999)}"


def _block_card_confirmed(context: dict) -> dict:
    fraud_result = context.get("fraud_result") or {}
    account_id   = fraud_result.get("account_id", "your account")
    last4        = str(account_id)[-4:] if len(str(account_id)) >= 4 else "****"
    ref_id       = _ref("FR")
    return {
        "response": (
            f"✅ Done — your card linked to account ending in {last4} has "
            f"been blocked. A replacement card will be issued within "
            f"5–7 business days. I've opened fraud case #{ref_id} and "
            f"escalated it to our fraud team; you'll get updates by SMS "
            f"and email. Anything else I can help with?"
        ),
        "risk_level"      : "High",
        "suggested_action": f"Card blocked · fraud case #{ref_id} escalated",
    }


def _reissue_lost_card_confirmed(context: dict) -> dict:
    ref_id = _ref("LC")
    return {
        "response": (
            f"✅ Done — your card has been blocked so it can't be used by "
            f"anyone else. A replacement is on its way and should arrive "
            f"within 5–7 business days. Reference #{ref_id}. Let me know "
            f"if there's anything else I can help with."
        ),
        "risk_level"      : "High",
        "suggested_action": f"Card blocked · replacement issued · ref #{ref_id}",
    }


def _raise_dispute_confirmed(context: dict) -> dict:
    fraud_result = context.get("fraud_result") or {}
    txn_id       = fraud_result.get("transaction_id", "the transaction")
    ref_id       = _ref("DP")
    return {
        "response": (
            f"✅ Done — I've raised a formal dispute on {txn_id}, reference "
            f"#{ref_id}. Our team will investigate within 7–15 working "
            f"days, and you'll get updates by SMS and email. If it's "
            f"confirmed as fraud, eligible amounts are credited back "
            f"within 10 working days."
        ),
        "risk_level"      : "Medium",
        "suggested_action": f"Dispute raised · ref #{ref_id} · investigation pending",
    }


def _unlock_account_confirmed(context: dict) -> dict:
    ref_id = _ref("AC")
    return {
        "response": (
            f"✅ Done — your account has been unlocked and a temporary "
            f"access code has been sent to your registered mobile number. "
            f"Reference #{ref_id}. You'll be asked to set a new password "
            f"on next login."
        ),
        "risk_level"      : "Low",
        "suggested_action": f"Account unlocked · ref #{ref_id}",
    }


def _security_lock_confirmed(context: dict) -> dict:
    ref_id = _ref("SEC")
    return {
        "response": (
            f"✅ Done — I've placed a security lock on your account and "
            f"flagged it for review by our security team, reference "
            f"#{ref_id}. This prevents any transactions until we confirm "
            f"it's you. You'll receive a call from our security desk "
            f"within a few hours."
        ),
        "risk_level"      : "High",
        "suggested_action": f"Security lock applied · ref #{ref_id} · under review",
    }


def _flag_loan_review_confirmed(context: dict) -> dict:
    ref_id = _ref("LN")
    return {
        "response": (
            f"✅ Done — I've flagged your loan application for manual "
            f"review, reference #{ref_id}. It's now prioritized with our "
            f"underwriting team, and you should hear an update within "
            f"2–3 business days via SMS and email."
        ),
        "risk_level"      : "Low",
        "suggested_action": f"Loan flagged for manual review · ref #{ref_id}",
    }


def _escalate_kyc_confirmed(context: dict) -> dict:
    ref_id = _ref("KYC")
    return {
        "response": (
            f"✅ Done — I've escalated your KYC documents for manual "
            f"review by a specialist, reference #{ref_id}. You'll hear "
            f"back within 1–2 business days via SMS and email."
        ),
        "risk_level"      : "Medium",
        "suggested_action": f"KYC escalated for manual review · ref #{ref_id}",
    }


def acknowledgment_reply() -> dict:
    """
    Deterministic response for simple acknowledgments ("ok", "thanks",
    "got it"). No LLM call, no template, no intent classification needed —
    this is just a polite close, so it's handled the same way confirm/
    decline replies are: caught before the pipeline, not after.
    """
    text = random.choice([
        "You're welcome! Let me know if there's anything else I can help with.",
        "Happy to help — reach out anytime you have another question.",
        "Glad that helped! I'm here if anything else comes up.",
        "Anytime! Feel free to come back if you need anything else.",
    ])
    return {
        "response"        : text,
        "risk_level"      : "Low",
        "suggested_action": "No action needed — acknowledgment",
    }


def _declined_action(context: dict) -> dict:
    action_label = context.get("action_label", "that action")
    return {
        "response": (
            f"No problem — I won't {action_label} for now. Let me know if "
            f"you change your mind, or if there's anything else I can help with."
        ),
        "risk_level"      : "Low",
        "suggested_action": "No action taken — customer declined",
    }


ACTIONS = {
    "block_card"        : {"confirm_fn": _block_card_confirmed,        "label": "block your card"},
    "reissue_lost_card" : {"confirm_fn": _reissue_lost_card_confirmed, "label": "block and reissue your card"},
    "raise_dispute"     : {"confirm_fn": _raise_dispute_confirmed,     "label": "raise a dispute"},
    "unlock_account"    : {"confirm_fn": _unlock_account_confirmed,    "label": "unlock your account"},
    "security_lock"     : {"confirm_fn": _security_lock_confirmed,     "label": "apply a security lock"},
    "flag_loan_review"  : {"confirm_fn": _flag_loan_review_confirmed,  "label": "flag your application for manual review"},
    "escalate_kyc_review": {"confirm_fn": _escalate_kyc_confirmed,     "label": "escalate your KYC for manual review"},
}


def execute_action(action_key: str, context: dict) -> dict:
    """Deterministically build the confirmation response for a pending action.
    No LLM call — this is pure application logic, not a decision."""
    action = ACTIONS.get(action_key)
    if not action:
        return {
            "response"        : "✅ Done — that's been taken care of.",
            "risk_level"      : "Medium",
            "suggested_action": "Action completed",
        }
    return action["confirm_fn"](context)


def decline_action(action_key: str) -> dict:
    label = ACTIONS.get(action_key, {}).get("label", "take that action")
    return _declined_action({"action_label": label})


# ══════════════════════════════════════════════════════════════════════════
#  Template selector — used by response_generator.py for the proposal turn
# ══════════════════════════════════════════════════════════════════════════

def get_template(
    intent: str,
    risk_level: str,
    fraud_result: dict = None,
    loan_already_flagged: bool = False,
    kyc_already_addressed: bool = False,
    account_compromise: bool = False,
    account_hard_lock: bool = False,
    query: str = "",
    qa_match: dict = None,
) -> tuple:
    """Returns (template_str, qa_grounded) based on intent + risk_level +
    fraud_result state.

    qa_grounded is True only when the returned template is
    grounded_qa_answer() — i.e. the response will be answered straight from
    a confident QA-pair match. response_generator.py uses this flag (not
    just "qa_match is truthy") to decide whether it's safe to overwrite
    risk_level/suggested_action with the QA pair's metadata — it must NOT
    do that when a deterministic action flow (real fraud score, confirmed
    lockout, etc.) produced the response instead, even if a QA chunk
    happened to also be retrieved alongside it.

    loan_already_flagged : a manual-review flag was already confirmed
        earlier in this conversation — don't re-offer it, answer the
        follow-up with the status/timeline instead.
    kyc_already_addressed: KYC upload guidance was already given earlier
        in this conversation — don't repeat it, escalate instead.
    account_compromise   : the current message suggests the lockout may
        be due to unauthorized access rather than a routine lockout.
    account_hard_lock    : the current message suggests a compliance/
        restriction hold that can't be resolved by self-service unlock —
        skip verification/unlock entirely and point to a human.
    qa_match              : output of get_qa_match() — if the top retrieved
        chunk is a confident QA-pair hit, prefer answering directly from
        it (with its policy_ref cited) over a rigid per-intent script.
        Deterministic action flows (fraud action proposals, confirmed
        lockouts) still take priority — a QA hit doesn't override those.
    """
    intent_clean = intent.lower()

    if "lost" in intent_clean or "stolen" in intent_clean:
        return lost_stolen_card(), False

    if "fraud" in intent_clean or "unauthorized" in intent_clean:
        if fraud_result is None:
            # No transaction/account ID in the message yet. If this is a
            # genuine incident report ("I see a transaction I didn't
            # make"), keep asking for details so the real fraud classifier
            # can score it and the bot can take a personalized action. If
            # it's a general policy/FAQ question with a confident QA match
            # ("how long does investigation take?"), just answer it —
            # there's no transaction to look up.
            if qa_match and not _is_fraud_incident_report(query):
                return grounded_qa_answer(qa_match["policy_ref"]), True
            return fraud_no_txn_id(), False
        elif fraud_result.get("found") == "needs_selection":
            return fraud_account_found(
                account_id=fraud_result.get("account_id", ""),
                recent_txns=fraud_result.get("recent_transactions", []),
            ), False
        elif fraud_result.get("found") is True:
            txn_id    = fraud_result["transaction_id"]
            amount    = f"{fraud_result['amount_inr']:,.2f}"
            merchant  = fraud_result["merchant_name"]
            timestamp = str(fraud_result.get("timestamp", "N/A"))
            fraud_pct = f"{fraud_result['fraud_probability']:.1%}"
            risk      = fraud_result["risk_level"]

            if risk == "High":
                return fraud_high_risk(txn_id, amount, merchant, timestamp, fraud_pct), False
            else:
                return fraud_low_medium_risk(txn_id, amount, merchant, timestamp, fraud_pct, risk), False
        else:
            if qa_match and not _is_fraud_incident_report(query):
                return grounded_qa_answer(qa_match["policy_ref"]), True
            return fraud_no_txn_id(), False

    elif "kyc" in intent_clean:
        if kyc_already_addressed:
            return kyc_support_escalate(), False
        if qa_match:
            return grounded_qa_answer(qa_match["policy_ref"]), True
        return kyc_support(), False

    elif "loan" in intent_clean:
        if loan_already_flagged:
            return loan_support_already_flagged(), False
        if qa_match and not _is_loan_status_complaint(query):            
            return grounded_qa_answer(qa_match["policy_ref"]), True
        if _is_loan_inquiry(query):
            return loan_inquiry(), False
        return loan_support(), False

    elif "account" in intent_clean or "access" in intent_clean:
        if not _is_account_lockout(query):
            # Not a lockout at all — a general account question. Answer
            # from context instead of assuming a password-reset scenario.
            if qa_match:
                return grounded_qa_answer(qa_match["policy_ref"]), True
            return account_access_informational(), False
        if account_compromise:
            return account_access_compromise(), False
        if account_hard_lock:
            return account_access_hard_lock(), False
        return account_access_verify(), False

    return "", False  # no template for general/unknown intent


def get_pending_action(
    intent: str,
    risk_level: str,
    fraud_result: dict = None,
    loan_already_flagged: bool = False,
    kyc_already_addressed: bool = False,
    account_compromise: bool = False,
    account_hard_lock: bool = False,
    query: str = "",
    qa_match: dict = None,
) -> Optional[str]:
    """
    Determine which action key (if any) the proposal response is offering,
    so orchestrator.py can track it as a pending confirmation for the
    customer's next turn. Returns None if this response doesn't propose
    a confirmable action (e.g. KYC/loan guidance, a hard account lock, an
    informational QA-grounded answer, or an info request).
    """
    intent_clean = intent.lower()

    if "lost" in intent_clean or "stolen" in intent_clean:
        return "reissue_lost_card"

    if "fraud" in intent_clean or "unauthorized" in intent_clean:
        if fraud_result and fraud_result.get("found") is True:
            return "block_card" if fraud_result.get("risk_level") == "High" else "raise_dispute"
        return None  # no txn identified yet — nothing to confirm

    if "account" in intent_clean or "access" in intent_clean:
        if not _is_account_lockout(query):
            return None  # general question, answered informationally — nothing to confirm
        if account_compromise:
            return "security_lock"  # yes/no confirmation
        if account_hard_lock:
            return None  # can't self-service — no action to confirm
        # Routine lockout → identity must be verified before any account
        # change is offered; "account_verify_identity" is a special
        # pending state (not a yes/no ACTIONS key) handled by orchestrator.
        return "account_verify_identity"

    if "loan" in intent_clean:
        if loan_already_flagged:
            return None  # already flagged — the follow-up just states the status
        if qa_match and not _is_loan_status_complaint(query): 
            return None  # answered informationally — nothing to confirm
        if _is_loan_inquiry(query):
            return None  # answering a criteria question, nothing to confirm
        return "flag_loan_review"

    if "kyc" in intent_clean:
        if kyc_already_addressed:
            return "escalate_kyc_review"
        return None  # first-pass guidance (QA-grounded or generic) — nothing to confirm yet

    return None  # general — guidance only, no action to confirm


