# BankingBot-RAG
AI-Powered Banking Support &amp; Fraud Intelligence System using NLP + RAG

# BankBot-RAG 🏦🤖

AI-Powered Banking Support & Fraud Intelligence System using NLP + RAG

## Overview
BankBot-RAG is an intelligent banking assistant that:
- Answers customer queries using Retrieval-Augmented Generation (RAG)
- Detects potential fraud using ML classification
- Routes queries intelligently using NLP intent classification

## Tech Stack
- **NLP:** scikit-learn, sentence-transformers
- **RAG:** FAISS, LangChain, OpenAI / HuggingFace LLM
- **ML:** XGBoost, SHAP
- **Frontend:** Streamlit
- **Language:** Python 3.10+

## Project Structure
BankBot-RAG/
├── data/               # Raw and processed datasets
├── src/
│   ├── nlp/            # Intent classifier, sentiment
│   ├── rag/            # Document loader, embedder, FAISS
│   ├── llm/            # Response generator
│   ├── risk/           # Fraud classifier
│   ├── actions/        # Action engine
│   └── orchestrator.py # Query router
├── app/                # Streamlit dashboard
├── tests/              # Pytest test files
└── notebooks/          # EDA and experiments

## Setup
```bash
git clone https://github.com/indupriya03/BankBot-RAG.git
cd BankBot-RAG
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys
```

## Status
🚧 In Progress — Day 1 Setup