"""
preprocessing.py
────────────────
Preprocessing pipeline for BankBot-RAG.
Reads from data/cleaned/, applies ML preprocessing, saves to data/preprocessed/.
Called by pipeline.py — can also be run standalone for testing.

Outputs:
  tickets_train.csv                 Intent classifier train split
  tickets_val.csv                   Intent classifier val split
  tickets_test.csv                  Intent classifier test split
  transactions_train_smote.csv      Fraud model train split (SMOTE applied)
  transactions_test.csv             Fraud model test split (real distribution)
  qa_chunks.json                    QA pairs as RAG document chunks
  policy_chunks.json                Policy docs chunked for embedding
  rag_master_chunks.json            Combined RAG index (QA + policy)
  label_encoder_category.pkl        Category label encoder
  scaler_transactions.pkl           StandardScaler for transaction features
  transaction_feature_cols.json     Feature column names for inference
"""

import json
import logging
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RANDOM_STATE = 42


# ── Text cleaning helper ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise customer query / resolution text for NLP.
    Keeps: alphanumeric, spaces, basic punctuation, Indian rupee symbol.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>",          "",  text)   # HTML tags
    text = re.sub(r"http\S+|www\.\S+", "", text)  # URLs
    text = re.sub(r"[^a-z0-9\s₹.,?!'-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Section 1: Support Tickets ────────────────────────────────────────────────

def preprocess_tickets(cleaned_dir: Path, output_dir: Path) -> None:
    """
    Preprocess support tickets for intent classification.

    Steps:
      1. Load cleaned tickets
      2. Clean query_text and resolution_text
      3. Build rag_document field (Category + Query + Resolution)
      4. Label encode category column
      5. Stratified 70/15/15 train/val/test split
      6. Save all splits + encoders
    """
    log.info("=" * 60)
    log.info("Preprocessing Support Tickets")
    log.info("=" * 60)

    path = cleaned_dir / "01_support_tickets_cleaned.csv"
    df   = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows from {path}")

    # ── Text cleaning ──────────────────────────────────────────────────────────
    df["query_text_clean"]      = df["query_text"].apply(clean_text)
    df["resolution_text_clean"] = df["resolution_text"].apply(clean_text)
    df["query_length"]          = df["query_text_clean"].apply(lambda x: len(x.split()))
    log.info(f"Text cleaned | avg query length: {df['query_length'].mean():.1f} words ✅")

    # ── RAG document field ─────────────────────────────────────────────────────
    df["rag_document"] = (
        "Category: "   + df["category"]              + "\n"
        "Query: "      + df["query_text_clean"]       + "\n"
        "Resolution: " + df["resolution_text_clean"]
    )
    log.info("rag_document field created ✅")

    # ── Label encoding ─────────────────────────────────────────────────────────
    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category"])

    joblib.dump(le, output_dir / "label_encoder_category.pkl")
    label_map = dict(zip(le.classes_, le.transform(le.classes_)))
    log.info(f"Category encoding: {label_map}")
    log.info("label_encoder_category.pkl saved ✅")

    # ── Train / Val / Test split ───────────────────────────────────────────────
    X = df["query_text_clean"]
    y = df["category_encoded"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    train_df = df.loc[X_train.index]
    val_df   = df.loc[X_val.index]
    test_df  = df.loc[X_test.index]

    train_df.to_csv(output_dir / "tickets_train.csv", index=False)
    val_df.to_csv(output_dir   / "tickets_val.csv",   index=False)
    test_df.to_csv(output_dir  / "tickets_test.csv",  index=False)

    log.info(f"Train : {len(train_df):>4} rows | {dict(y_train.value_counts().sort_index())}")
    log.info(f"Val   : {len(val_df):>4} rows | {dict(y_val.value_counts().sort_index())}")
    log.info(f"Test  : {len(test_df):>4} rows | {dict(y_test.value_counts().sort_index())}")

    # ── Full cleaned file ──────────────────────────────────────────────────────
    df.to_csv(output_dir / "support_tickets_clean.csv", index=False)
    log.info("support_tickets_clean.csv saved ✅\n")


# ── Section 2: Transactions ───────────────────────────────────────────────────

def preprocess_transactions(cleaned_dir: Path, output_dir: Path) -> None:
    """
    Preprocess transaction data for fraud classification.

    Steps:
      1. Load cleaned transactions
      2. Feature engineering (time, amount, flags)
      3. Encode categoricals + drop leaky/redundant columns
      4. Stratified 80/20 train/test split
      5. Scale numeric features (fit on train only)
      6. Apply SMOTE to training set only
      7. Save splits + scaler + feature column names
    """
    log.info("=" * 60)
    log.info("Preprocessing Transactions")
    log.info("=" * 60)

    path = cleaned_dir / "02_transactions_cleaned.csv"
    txn  = pd.read_csv(path)
    txn["timestamp"] = pd.to_datetime(txn["timestamp"])
    log.info(f"Loaded {len(txn):,} rows | fraud rate: {txn['fraud_label'].mean()*100:.1f}%")

    # ── Feature engineering ────────────────────────────────────────────────────
    txn["month"]                = txn["timestamp"].dt.month
    txn["is_night_transaction"] = txn["hour_of_day"].apply(
        lambda h: 1 if (h >= 22 or h <= 5) else 0
    )
    txn["is_weekend"] = txn["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    txn["amount_log"] = np.log1p(txn["amount_inr"])

    flag_cols = ["is_international", "velocity_flag", "geo_anomaly_flag", "high_amount_flag"]
    for col in flag_cols:
        txn[col] = txn[col].map({"Yes": 1, "No": 0})
    txn["flag_count"] = txn[flag_cols].sum(axis=1)

    log.info("Feature engineering complete: month, is_night_transaction, is_weekend, amount_log, flag_count ✅")

    # ── Drop leaky / redundant columns ────────────────────────────────────────
    DROP_COLS = [
        "transaction_id", "account_id", "timestamp",
        "merchant_name",   # too granular — high cardinality
        "amount_inr",      # replaced by amount_log
        "fraud_reason",    # target-derived → severe data leakage
        "day_of_week",     # replaced by is_weekend
    ]
    txn = txn.drop(columns=DROP_COLS, errors="ignore")
    log.info(f"Dropped leaky/redundant columns: {DROP_COLS} ✅")

    # ── One-hot encode categoricals ────────────────────────────────────────────
    CAT_COLS = ["merchant_category", "transaction_type", "city"]
    txn = pd.get_dummies(txn, columns=CAT_COLS, prefix=CAT_COLS, drop_first=False)
    log.info(f"One-hot encoded: {CAT_COLS} | shape after encoding: {txn.shape} ✅")

    # ── Train / Test split — BEFORE scaling and SMOTE ─────────────────────────
    X = txn.drop(columns=["fraud_label"])
    y = txn["fraud_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)} (before SMOTE)")
    log.info(f"Train fraud: {y_train.sum()} | Test fraud: {y_test.sum()}")

    # ── Scale numeric features — fit on train ONLY ────────────────────────────
    SCALE_COLS = [c for c in ["amount_log", "hour_of_day", "month", "flag_count"]
                  if c in X_train.columns]

    scaler = StandardScaler()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])

    joblib.dump(scaler, output_dir / "scaler_transactions.pkl")
    log.info(f"Scaling applied to: {SCALE_COLS} | scaler_transactions.pkl saved ✅")

    # ── SMOTE — training set ONLY ──────────────────────────────────────────────
    log.info(
        "\n⚠️  SMOTE NOTE: EDA showed near-perfect 100/0% splits on merchant_category, "
        "city, and transaction_type in this synthetic dataset. SMOTE is applied to "
        "demonstrate the pipeline — in real-world noisy data it provides genuine benefit. "
        "XGBoost scale_pos_weight is an alternative approach.\n"
    )

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    log.info(f"After SMOTE → Train: {len(X_train_sm)} | Fraud: {y_train_sm.sum()} | Legit: {(y_train_sm==0).sum()}")
    log.info(f"Test stays unchanged: {len(X_test)} rows (real-world distribution)")

    # ── Save splits ────────────────────────────────────────────────────────────
    train_df = pd.DataFrame(X_train_sm, columns=X_train.columns)
    train_df["fraud_label"] = y_train_sm.values

    test_df = pd.DataFrame(X_test.values, columns=X_test.columns)
    test_df["fraud_label"] = y_test.values

    train_df.to_csv(output_dir / "transactions_train_smote.csv", index=False)
    test_df.to_csv(output_dir  / "transactions_test.csv",        index=False)

    # ── Save feature column list for inference consistency ─────────────────────
    feature_cols = list(X_train.columns)
    with open(output_dir / "transaction_feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    log.info(f"transactions_train_smote.csv saved: {train_df.shape} ✅")
    log.info(f"transactions_test.csv saved: {test_df.shape} ✅")
    log.info(f"transaction_feature_cols.json saved: {len(feature_cols)} features ✅\n")


# ── Section 3: QA Pairs ───────────────────────────────────────────────────────

def preprocess_qa_pairs(cleaned_dir: Path, output_dir: Path) -> pd.DataFrame:
    """
    Convert cleaned QA pairs into RAG-ready document chunks.
    Each Q+A entry becomes one chunk combining all fields.
    Saves: qa_chunks.json
    """
    log.info("=" * 60)
    log.info("Preprocessing QA Pairs → RAG Chunks")
    log.info("=" * 60)

    path = cleaned_dir / "04_qa_pairs_cleaned.json"
    with open(path) as f:
        qa_data = json.load(f)

    log.info(f"Loaded {len(qa_data)} QA pairs")

    chunks = []
    for item in qa_data:
        text = (
            f"Category: {item.get('category', '')}\n"
            f"Question: {item.get('question', '')}\n"
            f"Answer: {item.get('answer', '')}\n"
            f"Policy Reference: {item.get('policy_ref', '')}\n"
            f"Risk Level: {item.get('risk_level', '')}\n"
            f"Suggested Action: {item.get('suggested_action', '')}"
        )
        chunks.append({
            "chunk_id"  : f"qa_{item['id']}",
            "source"    : "qa_pairs",
            "category"  : item.get("category", ""),
            "risk_level": item.get("risk_level", ""),
            "text"      : text,
            "word_count": len(text.split()),
        })

    df = pd.DataFrame(chunks)
    df.to_json(output_dir / "qa_chunks.json", orient="records", indent=2)

    log.info(f"QA chunks created: {len(df)} | avg words: {df['word_count'].mean():.0f}")
    log.info("qa_chunks.json saved ✅\n")
    return df


# ── Section 4: Policy Documents ──────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """
    Split text into overlapping word-based chunks.

    Args:
        text       : Cleaned document text
        chunk_size : Target words per chunk (~512 tokens for sentence-transformers)
        overlap    : Overlapping words between consecutive chunks (~64 tokens)

    Returns:
        List of chunk strings
    """
    words  = text.split()
    chunks = []
    start  = 0
    step   = chunk_size - overlap

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
        start += step

    return chunks


def preprocess_policy_docs(cleaned_dir: Path, output_dir: Path) -> pd.DataFrame:
    """
    Chunk cleaned policy .txt files for vector embedding.

    Strategy:
      - 400-word chunks (~512 tokens for sentence-transformers)
      - 50-word overlap to preserve context at chunk boundaries
      - Each chunk stored with metadata (source doc, chunk number)
    Saves: policy_chunks.json
    """
    log.info("=" * 60)
    log.info("Preprocessing Policy Docs → Chunks")
    log.info("=" * 60)

    policy_dir = cleaned_dir / "policy_docs"
    all_chunks = []

    for fpath in sorted(policy_dir.glob("*.txt")):
        text   = fpath.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text, chunk_size=400, overlap=50)

        doc_name = fpath.stem
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id"    : f"{doc_name}_chunk_{i+1:03d}",
                "source"      : "policy_docs",
                "document"    : doc_name,
                "chunk_num"   : i + 1,
                "total_chunks": len(chunks),
                "text"        : chunk,
                "word_count"  : len(chunk.split()),
            })

        log.info(f"  {fpath.name:<35} → {len(chunks)} chunks ({len(text.split())} words)")

    df = pd.DataFrame(all_chunks)
    df.to_json(output_dir / "policy_chunks.json",
               orient="records", indent=2, force_ascii=False)

    log.info(f"\nTotal policy chunks: {len(df)}")
    log.info("policy_chunks.json saved ✅\n")
    return df


# ── Section 5: Master RAG Index ───────────────────────────────────────────────

def build_rag_master_index(qa_df: pd.DataFrame, policy_df: pd.DataFrame,
                           output_dir: Path) -> None:
    """
    Combine QA chunks + policy chunks into one master FAISS input file.
    Saves: rag_master_chunks.json
    """
    log.info("=" * 60)
    log.info("Building Master RAG Chunk Index")
    log.info("=" * 60)

    records = []

    for _, row in qa_df.iterrows():
        records.append({
            "chunk_id"  : row["chunk_id"],
            "source"    : "qa_pairs",
            "category"  : row.get("category", ""),
            "text"      : row["text"],
            "word_count": row["word_count"],
        })

    for _, row in policy_df.iterrows():
        records.append({
            "chunk_id"  : row["chunk_id"],
            "source"    : "policy_docs",
            "category"  : "",
            "text"      : row["text"],
            "word_count": row["word_count"],
        })

    master_df = pd.DataFrame(records)
    master_df.to_json(output_dir / "rag_master_chunks.json",
                      orient="records", indent=2, force_ascii=False)

    log.info(f"Master RAG index: {len(master_df)} total chunks")
    log.info(f"  QA pairs   : {(master_df['source']=='qa_pairs').sum()}")
    log.info(f"  Policy docs: {(master_df['source']=='policy_docs').sum()}")
    log.info("rag_master_chunks.json saved ✅\n")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Running preprocessing.py in standalone mode")
    print("CWD:", os.getcwd())

    cleaned_dir = Path("data/cleaned")
    output_dir  = Path("data/preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_tickets(cleaned_dir, output_dir)
    preprocess_transactions(cleaned_dir, output_dir)
    qa_df     = preprocess_qa_pairs(cleaned_dir, output_dir)
    policy_df = preprocess_policy_docs(cleaned_dir, output_dir)
    build_rag_master_index(qa_df, policy_df, output_dir)

    log.info("✅ All preprocessing complete → data/preprocessed/")