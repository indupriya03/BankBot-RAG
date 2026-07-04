
import pandas as pd
import json
import os

print("="*60)
print("1. SUPPORT TICKETS")
print("="*60)
tickets = pd.read_csv("data/raw/01_support_tickets.csv")
print("Shape:", tickets.shape)
print("\nColumns:", list(tickets.columns))
print("\nHead:\n", tickets.head(3))
print("\nNulls:\n", tickets.isnull().sum())
if "category" in tickets.columns:
    print("\nCategory counts:\n", tickets["category"].value_counts())

print("\n" + "="*60)
print("2. TRANSACTIONS")
print("="*60)
txn = pd.read_csv("data/raw/02_transactions.csv")
print("Shape:", txn.shape)
print("\nColumns:", list(txn.columns))
print("\nHead:\n", txn.head(3))
print("\nNulls:\n", txn.isnull().sum())
fraud_col = [c for c in txn.columns if "fraud" in c.lower()]
if fraud_col:
    print(f"\n{fraud_col[0]} counts:\n", txn[fraud_col[0]].value_counts())

print("\n" + "="*60)
print("3. QA PAIRS (JSON)")
print("="*60)
with open("data/raw/04_qa_pairs.json") as f:
    qa = json.load(f)
print("Type:", type(qa))
if isinstance(qa, list):
    print("Count:", len(qa))
    print("First item:\n", qa[0])
elif isinstance(qa, dict):
    print("Keys:", list(qa.keys())[:10])

print("\n" + "="*60)
print("4. POLICY DOCS FOLDER")
print("="*60)
for f in os.listdir("data/raw/policy_docs"):
    path = os.path.join("data/raw/policy_docs", f)
    size = os.path.getsize(path)
    print(f"{f} — {size} bytes")

print("\n" + "="*60)
print("5. DATASET README")
print("="*60)
with open("data/raw/README.md") as f:
    print(f.read())
