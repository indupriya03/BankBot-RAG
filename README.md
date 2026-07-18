# 🏦 BankBot-RAG — AI Banking Support & Fraud Intelligence System

[![Tracked on DagsHub](https://img.shields.io/badge/Tracking-DagsHub-blue)](https://dagshub.com/indupriya03/BankBot-RAG)
[![DVC Pipeline](https://img.shields.io/badge/Pipeline-DVC-945DD6)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/Experiments-MLflow-0194E2)](https://mlflow.org/)

> A retrieval-augmented banking support chatbot that answers customer queries, scores transactions for fraud in real time, and executes simulated banking actions (card blocks, escalations) through a single-confirmation conversational flow — combining a 4-way intent classifier, a 4-model fraud comparison, and a grounded RAG pipeline into one orchestrated system.

---
🎬 Demo

https://github.com/indupriya03/BankBot-RAG/blob/main/assets/demo.mp4

---
## 🏆 Highlights

| | |
|---|---|
| 🧠 **4-Model Fraud Comparison** | XGBoost, LightGBM, Random Forest, Logistic Regression compared on PR-AUC (not accuracy) — winner auto-promoted to production |
| 📚 **Policy-Aware Retrieval** | Every answer includes at least one official policy chunk, not just the closest-sounding FAQ — similarity search alone tends to favor FAQs since they're phrased like questions |
| 🎯 **Zero-Shot Intent + Sentiment** | `bart-large-mnli`, chosen deliberately after finding 53% label inconsistency in the synthetic sentiment data |
| 🔁 **Full DVC Pipeline** | 10 stages, `clean → preprocess (×3) → evaluate NLP → train → evaluate → select → build vector store → evaluate retrieval`, reproducible end to end with `dvc repro` |
| ⚙️ **Config-Driven** | Every tunable value — SMOTE params, risk thresholds, RAG chunk size, LLM temperature — lives in `params.yaml`, read directly by the code, not duplicated |
| 🤖 **Rule-Based Action Layer** | Card blocks, lockouts, escalations execute as plain Python (no LLM inference deciding or executing them), gated behind exactly one yes/no confirmation — no repeated asks, no disclaimers |
| 📊 **MLflow + DagsHub Tracking** | Every training/eval/retrieval run logged and comparable — [browse the experiments](https://dagshub.com/indupriya03/BankBot-RAG.mlflow) |

---

## 📁 Project Structure

```
BankBot-RAG/
├── dvc.yaml                    # 10-stage reproducible pipeline
├── params.yaml                 # single source of truth for every tunable value
├── app/
│   └── streamlit_app.py        # frontend — chat + live pipeline analysis panel
│
├── src/
│   ├── data/
│   │   ├── data_cleaning.py            # raw -> cleaned (tickets, transactions, QA, policy docs)
│   │   ├── preprocessing_tickets.py    # -> intent/sentiment eval set
│   │   ├── preprocessing_transactions.py # -> SMOTE-balanced fraud train/test split
│   │   └── preprocessing_rag.py        # -> QA + policy chunks (LangChain splitter)
│   │
│   ├── nlp/
│   │   ├── intent_classifier.py        # zero-shot, 4-way (Fraud/Loan/KYC/Account Access)
│   │   ├── sentiment_analyzer.py       # zero-shot, 3-class (Urgent/Negative/Neutral)
│   │   ├── nlp_pipeline.py             # single entry point: intent + sentiment + derived priority
│   │   └── evaluate_nlp.py             # runs the intent classifier against labeled data, logs results to MLflow
│   │
│   ├── risk/
│   │   ├── train_model.py              # trains 4 fraud models
│   │   ├── evaluate_model.py           # PR-AUC ranked comparison + risk-tier validation
│   │   ├── select_best_model.py        # promotes the winner to models/fraud/production/
│   │   ├── transaction_risk_lookup.py  # chat message -> transaction ID -> live fraud score
│   │   └── run_risk_pipeline.py        # runs all 4 risk stages in order
│   │
│   ├── rag/
│   │   ├── gemini_embeddings.py        # asymmetric doc/query embedding (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY)
│   │   ├── build_vector_store.py       # -> persistent ChromaDB collection
│   │   └── retriever.py                # policy-floor-guaranteed retrieval
│   │
│   ├── llm/
│   │   ├── response_generator.py       # Groq call, JSON-structured, grounded generation
│   │   └── response_templates.py       # deterministic ACTIONS registry + confirmation templates
│   │
│   ├── eval/
│   │   ├── golden_queries.json         # 13 hand-labeled gold retrieval queries
│   │   ├── eval_retrieval.py           # recall@k / precision@k / MRR, MLflow-logged
│   │   ├── dump_corpus.py              # full-corpus dump — avoids top-5-only labeling bias
│   │   ├── sample_gold_candidates.py   # stratified real-ticket sampling for gold-set building
│   │   └── coverage_gaps.md            # documented retrieval + classifier findings
│   │
│   └── orchestrator.py                 # state machine — routes intent -> fraud check -> retrieval -> generation -> action
│
├── notebooks/
│   ├── 1_data_profiling.ipynb          # Phase 1 — raw data profiling
│   └── 02_eda.ipynb                    # Phase 2 — EDA + label consistency audit
│
├── tests/                       # placeholder only, not yet implemented
│   ├── test_intent.py
│   └── test_rag.py
│
├── reports/
│   ├── fraud_eval/              # per-model confusion matrices, feature importance, risk-tier plots
│   └── nlp_eval/                # intent confusion matrix, classification report
│
└── models/fraud/
    └── production/              # auto-promoted winner + promotion metadata
```

---

## 🧩 Architecture

```
Customer message
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  NLP Layer (zero-shot, bart-large-mnli)                  │
│  Intent (4-way) ──┐                                      │
│  Sentiment (Urgent/Negative/Neutral) ─┴─▶ Priority (high/normal)│
└─────────────────────────────────────────────────────────┘
      │
      ▼  (if a transaction ID is present in the message)
┌─────────────────────────────────────────────────────────┐
│  Fraud Scoring — production-promoted XGBoost model        │
│  probability ─▶ risk tier (High ≥0.70 / Medium ≥0.30 / Low)│
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Retrieval — ChromaDB + Gemini embeddings                 │
│  Ranks ALL chunks first (cheap at 33 docs), then applies   │
│  the policy floor, then trims to top-k                    │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Orchestrator (state machine)                              │
│  Sticky intent · confirm/decline detection · action gating │
│  Rule-based ACTIONS registry (block_card, unlock_account,  │
│  raise_dispute, escalate_kyc_review, ...)                  │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Generation — Groq llama-3.3-70b-versatile                │
│  Grounded in retrieved context, JSON-structured output     │
└─────────────────────────────────────────────────────────┘
      │
      ▼
  Response + one yes/no confirmation (if an action is proposed)
```

**Design principle:** the LLM never decides *whether* to block a card or unlock an account — that's the rule-based `ACTIONS` registry's job (plain Python, no model inference). The LLM only generates the grounded natural-language explanation. This keeps every action auditable and testable independent of what the LLM happens to say.

---

## 📊 Results

### Fraud Model Comparison (PR-AUC ranked — not accuracy)

Fraud is rare (12.2% of transactions) — a model predicting "not fraud" for everything scores 95%+ accuracy while catching zero fraud. PR-AUC is the honest metric here.

```
Model                  Precision  Recall    F1      ROC-AUC   PR-AUC
──────────────────────────────────────────────────────────────────
XGBoost (promoted)     0.679      0.776     0.724   0.949     0.8655  ← winner
Random Forest          0.478      0.878     0.619   0.949     0.8626
LightGBM               0.698      0.755     0.726   0.943     0.8601
Logistic Regression    0.405      0.918     0.563   0.938     0.8593
```

> All 4 candidates score within ~0.006 PR-AUC of each other — the real signal is thin with only 5 clean numeric features remaining after leakage removal. XGBoost wins on balanced precision/recall; Logistic Regression trades precision heavily for recall.

### Intent Classification (4-way, zero-shot)

```
Category               Precision  Recall   F1      Support
──────────────────────────────────────────────────────────
Fraud/Unauthorized      0.94      0.97     0.95      32
Loan                    0.82      1.00     0.90      14
KYC                     0.90      0.90     0.90      10
Account Access          0.83      0.50     0.62      10
──────────────────────────────────────────────────────────
Overall accuracy: 0.89   |   Macro-F1: 0.85
```

> Account Access's recall (0.50) is a **documented ground-truth granularity issue, not a model failure** — half its eval samples (salary credit delays, cheque returns, joint-holder requests) aren't actually access/login problems by any normal reading of the category name.

### Retrieval Evaluation (13 hand-labeled gold queries)

```
MRR: 0.910   |   Recall@3/5/10: 1.0 (all)   |   Precision@3: 0.33, @5: 0.22, @10: 0.11
```

> Recall stays perfect while precision falls as k grows — expected given each gold query has ~1 correct chunk out of 34 total. n=13 is small; treated as directional, not a robust statistic. One gold query originally had no matching chunk at all (see Generation Evaluation below) — QA021 was added specifically to close that gap, which is why the corpus grew from 33 to 34 documents mid-project.

### Generation Evaluation (faithfulness, holding retrieval constant)

Retrieval evaluation checks whether the right chunk gets *found*. This
checks a different question: given the *correct* chunk, does the LLM's
answer stay faithful to it? For each of the 13 gold queries, the true
gold chunk is fetched directly by ID (not via a fresh retrieval) and fed
to `generate_response()`, then an LLM-judge checks whether every claim in
the response is actually supported by that chunk.

```
Faithfulness rate      : 76.92%
Answers-question rate  : 61.54%
Citation accuracy rate : 100%   (was 92.31% before a prompt fix — see below)
```

**Two distinct causes found, not one:**

1. **A real coverage gap.** "How do I check my loan application status?"
   had no matching source content at all — the closest available chunk
   (`qa_QA009`, about rejection reasons) caused the LLM to invent a
   plausible-sounding but fabricated answer. Fixed by adding `qa_QA021`,
   a real QA pair covering status tracking, and re-verified: retrieval
   now correctly surfaces it — but the generation still hallucinated on
   this query even with the correct chunk (see #2), showing the two
   failure modes are independent and both needed addressing.

2. **LLM over-elaboration, independent of retrieval quality.** Loan-related
   queries with *genuinely correct* context still had the model inventing
   specific, checkable details not in the source (an invented "last 4
   digits" verification step, a fabricated "manual review" offer, a
   fabricated toll-free number). A `SYSTEM_PROMPT` rule was added
   explicitly warning against inventing specific/checkable details.
   Result: fabricated content became visibly vaguer and less risky
   (specific invented procedures disappeared), but the binary faithfulness
   rate didn't move — a strict pass/fail judge doesn't distinguish
   "invented a phone number" from "added a generic empathetic filler
   line," so the improvement is real but not visible in this metric.
   Documented as a known limitation of binary faithfulness scoring, not
   treated as "the fix didn't work."

**Also found and fixed:** the LLM was citing in-document section numbers
(e.g. `"4.5"` from `"4.5 Business Loan"` inside a policy chunk) as if they
were `chunk_id`s. Tightening the `grounded_in` instruction in
`SYSTEM_PROMPT` to require exact `chunk_id=` values took citation accuracy
from 92.31% → 100%.

---
> Retrieval quality and generation faithfulness are genuinely separate
> properties — a system can retrieve perfectly and still hallucinate, and
> fixing a data gap doesn't automatically fix a model's tendency to
> embellish. n=13 here too; directional, not a robust statistic.
---

## 🔬 Key ML Engineering Decisions

**Zero-shot over trained sentiment classifier:**
```
Checked label consistency across 66 duplicate query texts:
  category  → 0% inconsistent  → reliable, learnable    → trained a classifier
  sentiment → 53% inconsistent → same text, 5 different  → zero-shot instead
                                  labels seen               (can't learn from noise)
```

**Fraud feature leakage removal:**
```
geo_anomaly_flag, is_international  → 100% correlated with fraud_label
merchant_category, city, txn_type   → perfect category separation
                                        (e.g. Crypto/Dubai/ATM = always fraud
                                        in this synthetic data — an artifact,
                                        not real signal)
→ Dropped all of the above. 5 clean numeric features remain.
```

**Policy-aware retrieval:**
```
Similarity search naturally favors FAQs over policy documents — FAQs are
phrased like customer questions ("Question: ... Answer: ..."), while
policy text is formal and covers several sub-topics per chunk, so it
scores worse even when it's the more authoritative source.
→ retriever.py ranks every chunk first, then guarantees at least
  min_policy_results policy chunks make the final answer — so responses
  don't accidentally rely only on a simplified FAQ paraphrase.
```

**Config-driven, not just config-documented:**
```
params.yaml isn't a wishlist — every value in it is read directly by the
script that uses it (verified individually, before/after, for all 9 files).
Change risk_thresholds.high once → evaluate_model.py AND
transaction_risk_lookup.py both pick it up. No hardcoded duplicates,
no drift possible.
```

**Tried and reverted — parallel NLP inference:**
```
Attempted running intent + sentiment classification concurrently
(ThreadPoolExecutor) to cut ~7s NLP latency in half.
→ Caused a genuine PyTorch thread-safety crash (RuntimeError: tensor on
  meta device) during concurrent bart-large-mnli weight loading.
→ Reverted to sequential. A working 5-7s response beats a fast one that
  crashes. Kept the HF_HUB_OFFLINE fix instead, which gave a real,
  safe latency win with no correctness risk.
```

---

## 🚀 Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Fill in: GROQ_API_KEY, GEMINI_API_KEY, DAGSHUB_TOKEN
```

### 3. Run the full pipeline
```bash
dvc repro
```
Runs all 10 stages: clean → preprocess (tickets/transactions/RAG) → evaluate NLP → train fraud models → evaluate fraud models → select & promote best model → build vector store → evaluate retrieval.

### 4. Launch the app
```bash
streamlit run app/streamlit_app.py
```

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|-----------|
| Generation (LLM) | Groq — `llama-3.3-70b-versatile` |
| Retrieval | ChromaDB + Google `gemini-embedding-001` |
| Intent / Sentiment | Zero-shot NLI, `facebook/bart-large-mnli` |
| Fraud Scoring | XGBoost (promoted), compared against LightGBM, Random Forest, Logistic Regression |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Frontend | Streamlit |
| Pipeline | DVC (10 stages, `params.yaml`-driven) |
| Experiment Tracking | MLflow, hosted on DagsHub |
| Environment | Python 3.13 |

---

## 🗂️ Dataset

Synthetic banking dataset (HCL GUVI course project data).

| Dataset | Rows | Notes |
|---|---|---|
| Support Tickets | 200 (66 unique after dedup) | 4 categories, 0% label inconsistency on `category`, 53% on `sentiment` |
| Transactions | 2,000 | 12.2% fraud rate; fraud amounts average ~18x legitimate ones |
| QA Pairs | 21 | Structured with `policy_ref`, `suggested_action`, `risk_level`; QA021 added mid-project to close a documented retrieval coverage gap |
| Policy Docs | 4 files | Chunked into 13 policy chunks + 21 QA chunks = 34 total retrieval documents |
---

## 👤 Author

**Indupriya Chidambararaj**
- 🔗 [LinkedIn](https://www.linkedin.com/in/indupriyachidambararaj/)
- 🐙 [GitHub](https://github.com/indupriya03)
- 📧 indupriya.chidambararaj@gmail.com