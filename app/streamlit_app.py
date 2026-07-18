"""
app.py
───────
SwiftBank BankBot-RAG — Streamlit Chat Interface
3-column layout: pipeline/evaluation metadata lives in the side columns,
the middle column is a clean, customer-facing chat with no risk badges or
pipeline chrome cluttering the conversation.

Run:
    streamlit run app.py
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import warnings
warnings.filterwarnings("ignore", message=".*torchvision.*")


def get_greeting() -> str:
    """Time-of-day greeting based on the server's local time."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"


def md(content: str):
    """Render markdown with left-stripped content (for nicer indentation in code)."""
    cleaned = "\n".join(line.lstrip() for line in content.split("\n"))
    st.markdown(cleaned, unsafe_allow_html=True)

# ── Path setup — same pattern as orchestrator.py ──────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
for _sub in ("nlp", "rag", "llm", "risk"):
    _path = str(_SRC_DIR / _sub)
    if Path(_path).exists() and _path not in sys.path:
        sys.path.insert(0, _path)
sys.path.insert(0, str(_SRC_DIR.parent))
sys.path.insert(0, str(_SRC_DIR))

from orchestrator import BankBotOrchestrator  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SwiftBank — BankBot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
md("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* User bubble */
.user-bubble {
    background: #1B4FD8;
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 48px;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* Bot bubble — clean, customer-facing, no pipeline chrome */
.bot-bubble {
    background: #F1F5F9;
    color: #1E293B;
    padding: 12px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 48px 8px 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Risk badges (used only in the side panel now) */
.badge-high   { background:#FEE2E2; color:#991B1B; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.badge-medium { background:#FEF3C7; color:#92400E; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.badge-low    { background:#DCFCE7; color:#166534; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }

/* Welcome card */
.welcome-card {
    background: linear-gradient(135deg, #1B4FD8 0%, #1E40AF 100%);
    color: white;
    padding: 22px 24px;
    border-radius: 16px;
    margin-bottom: 20px;
}
.welcome-card h3 { margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 700; }
.welcome-card p  { margin: 0; opacity: 0.9; font-size: 0.88rem; }

/* Side-panel metric rows */
.metric-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #E2E8F0;
    font-size: 0.85rem;
    gap: 8px;
}
.metric-label { color: #64748B; flex-shrink: 0; }
.metric-value { color: #1E293B; font-weight: 600; text-align: right; }

/* Source / policy chip */
.source-chip {
    background: #EFF6FF;
    color: #1D4ED8;
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 0.75rem;
    margin: 2px;
    display: inline-block;
}
.policy-chip {
    background: #ECFDF5;
    color: #065F46;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.8rem;
    display: inline-block;
    margin-top: 4px;
}

/* Side panel card */
.side-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
}
.side-card h4 { margin: 0 0 10px 0; font-size: 0.85rem; color: #475569;
                 text-transform: uppercase; letter-spacing: 0.03em; }
</style>
""")


# ── Load orchestrator (cached — loads ONCE, reused for all queries) ───────────
@st.cache_resource(show_spinner="Loading SwiftBank BankBot — please wait...")
def load_orchestrator():
    return BankBotOrchestrator(
        persist_dir="data/vector_store/chroma_db",
        retrieval_k=5,
        fraud_model_path="models/fraud/production/production_fraud_model.pkl",
    )


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages     = []
if "last_result" not in st.session_state:
    st.session_state.last_result  = None

# ── Query processing (shared by example buttons and the chat form) ───────────
def run_query(query_text: str):
    st.session_state.messages.append({"role": "user", "content": query_text})
    with st.spinner("🔍 Analyzing your query..."):
        try:
            orchestrator = load_orchestrator()
            result = orchestrator.handle_query(
                query_text,
                conversation_history=st.session_state.messages[:-1],
            )
            st.session_state.last_result = result
            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "response"        : result["response"],
                    "risk_level"      : result["risk_level"],
                    "suggested_action": result["suggested_action"],
                    "grounded_in"     : result["grounded_in"],
                    "intent"          : result["intent"]["intent"],
                    "pending_action"  : result.get("pending_action"),
                    "fraud_result"    : result.get("fraud_result"),
                    "policy_reference": result.get("policy_reference"),
                },
            })
        except Exception as e:
            import logging
            logging.error(f"Orchestrator error: {e}", exc_info=True)
            st.session_state.last_result = None
            st.session_state.messages.append({
                "role": "assistant",
                "content": {
                    "response": (
                        "I'm having trouble processing that right now. "
                        "Your query has been noted — please try again in a moment, "
                        "or reach us at 1800-123-4567 (24x7)."
                    ),
                    "risk_level": "Medium",
                    "suggested_action": "Escalate to human agent — response generation failed",
                    "grounded_in": [], "intent": None, "pending_action": None,
                    "fraud_result": None, "policy_reference": None,
                },
            })

# ══════════════════════════════════════════════════════════════════════════
#  3-column layout: [ left context panel | chat (customer-facing) | pipeline ]
# ══════════════════════════════════════════════════════════════════════════
left_col, chat_col, right_col = st.columns([1, 2.2, 1], gap="medium")

# ── LEFT COLUMN — branding, examples, sources/policy used ────────────────────
with left_col:
    st.markdown("### 🏦 SwiftBank")
    st.caption("AI Banking Assistant")
    st.divider()

    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        example_queries = [
            "I see a transaction I didn't make",
            "My transaction TXN500004 looks wrong",
            "My loan application has been pending for 3 weeks",
            "I cannot upload my KYC documents",
            "I am locked out of my account",
        ]
        for i, q in enumerate(example_queries):
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                run_query(q)
                st.rerun()
    else:
        result = st.session_state.last_result
        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        st.markdown("#### 📚 Grounding", unsafe_allow_html=True)
        if result and result.get("policy_reference"):
            md(f'<span class="policy-chip">📄 {result["policy_reference"]}</span>')
        if result and result.get("grounded_in"):
            chips = "".join(f'<span class="source-chip">{s}</span>' for s in result["grounded_in"])
            md(f'<div style="margin-top:8px">{chips}</div>')
        if not result or not (result.get("policy_reference") or result.get("grounded_in")):
            st.caption("No policy sources used for this turn.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages    = []
        st.session_state.last_result = None
        st.rerun()

    md("""
    <div style='font-size:0.72rem; color:#94A3B8; margin-top:10px;'>
    Powered by ChromaDB · Groq · Gemini Embeddings · XGBoost
    </div>
    """)


# ── MIDDLE COLUMN — clean, customer-facing chat ───────────────────────────────
with chat_col:
    if not st.session_state.messages:
        md(f"""
        <div class="welcome-card">
            <h3>🏦 {get_greeting()}! Welcome to SwiftBank</h3>
            <p>I'm <strong>BankBot</strong>, your AI banking support assistant.
            Ask me about fraud, loans, KYC, or account access.</p>
        </div>
        """)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">👤 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            data = msg["content"]
            # Deliberately just the response text — no risk badge, no
            # suggested-action chip, no source chips in the customer-facing
            # bubble. That metadata lives in the right-hand panel instead,
            # so the chat itself reads like talking to a real support bot.
            md(f"""
            <div class="bot-bubble">
                🤖 {data['response']}
            </div>
            """)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "query",
                placeholder="Type your banking query here...",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and user_input.strip():
        run_query(user_input.strip())
        st.rerun()


# ── RIGHT COLUMN — pipeline analysis, for evaluation/demo purposes ───────────
with right_col:
    st.markdown("### 🔍 Pipeline Analysis")
    result = st.session_state.last_result

    if result:
        intent    = result["intent"]
        sentiment = result["sentiment"]
        risk      = result["risk_level"]
        latency   = result["latency_ms"]

        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        md(f"""
        <div class="metric-row">
            <span class="metric-label">Intent</span>
            <span class="metric-value">{intent['intent']}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Confidence</span>
            <span class="metric-value">{intent['confidence']:.0%}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Sentiment</span>
            <span class="metric-value">{sentiment['sentiment']}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Priority</span>
            <span class="metric-value">{result['priority'].upper()}</span>
        </div>
        """)
        badge_class = f"badge-{risk.lower()}"
        risk_emoji  = "🔴" if risk == "High" else "🟡" if risk == "Medium" else "🟢"
        md(f"""
        <div class="metric-row">
            <span class="metric-label">Risk Level</span>
            <span class="{badge_class}">{risk_emoji} {risk}</span>
        </div>
        """)
        if result.get("suggested_action"):
            md(f"""
            <div class="metric-row">
                <span class="metric-label">Next Action</span>
                <span class="metric-value">{result['suggested_action']}</span>
            </div>
            """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Fraud score if available
        fr = result.get("fraud_result")
        if fr and fr.get("found") is True:
            st.markdown('<div class="side-card">', unsafe_allow_html=True)
            st.markdown("#### 🚨 Fraud Check", unsafe_allow_html=True)
            md(f"""
            <div class="metric-row">
                <span class="metric-label">Fraud Score</span>
                <span class="metric-value">{fr['fraud_probability']:.1%}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Transaction</span>
                <span class="metric-value">{fr['transaction_id']}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Amount</span>
                <span class="metric-value">₹{fr['amount_inr']:,.0f}</span>
            </div>
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        st.markdown("#### ⚡ Latency", unsafe_allow_html=True)
        md(f"""
        <div class="metric-row">
            <span class="metric-label">NLP</span>
            <span class="metric-value">{latency['nlp']:.0f}ms</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Fraud Check</span>
            <span class="metric-value">{latency['fraud']:.0f}ms</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Retrieval</span>
            <span class="metric-value">{latency['retrieval']:.0f}ms</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Generation</span>
            <span class="metric-value">{latency['generation']:.0f}ms</span>
        </div>
        <div class="metric-row">
            <span class="metric-label"><b>Total</b></span>
            <span class="metric-value"><b>{latency['total']:.0f}ms</b></span>
        </div>
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("Pipeline details will appear here after your first query.")