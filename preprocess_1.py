# preprocess_1.py - BOTRGCN_Original structure for benchmark data (respects CSV split per user)
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import os
from datetime import datetime as dt

# Your utilities
from dataset_tool import merge
from utils import norm_user_id, RELATION_TO_TYPE, infer_num_relations

print('Loading raw data')

# --- CONFIG (unchanged) ---
CSV_PATH = "/Users/bala/Documents/RA/Bot Analytics/Benchmark_Data/Non_User_Overlap_Benchmark/Benchmark_data_1m.csv"
EDGES_PATH = "/Users/bala/Documents/RA/Bot Analytics/Benchmark_Data/Non_User_Overlap_Benchmark/edge_from_big_1m_new.csv"

# ---------------------------------------------------------------------
# 1) Load Benchmark CSV through your merge() helper (handles tokens/paths)
# ---------------------------------------------------------------------
print(f"Loading benchmark data from: {CSV_PATH}")
df = merge(CSV_PATH, "209")  # keeps your normalization pipeline
# Normalize user_id consistently
df["user_id"] = df["user_id"].astype(str).map(norm_user_id)

# Make sure split is normalized to lower-case canonical names
df["split"] = (
    df["split"].astype(str).str.strip().str.lower()
      .replace({"validation": "val", "valid": "val", "testing": "test", "trainning": "train"})
)

# ----- For context only: row-level distribution (like your old printout) -----
print("Original data distribution:")
for split_name in ("test", "train", "val"):
    sub = df[df["split"].eq(split_name)]
    if len(sub) == 0:
        continue
    # label as strings may exist; normalize to bot/human strings for this view
    lbl = sub["label"].astype(str).str.lower().str.strip()
    bots = (lbl.isin(["1", "bot", "true", "yes"])).sum()
    humans = len(sub) - bots
    print(f"  {split_name.title()}: {len(sub):,} total, Bots={bots:,} ({(bots/max(len(sub),1))*100:.1f}%), "
          f"Humans={humans:,} ({(humans/max(len(sub),1))*100:.1f}%)")

# ---------------------------------------------------------------------
# 2) Build PER-USER label and split
#    - label: majority vote over that user's rows (1=bot, 0=human)
#    - split: deterministic resolution: prefer train > val > test
# ---------------------------------------------------------------------
# Majority vote labels
lab = (
    df[["user_id", "label"]]
    .assign(label=lambda d: d["label"].astype(str).str.lower().str.strip()
            .map({"bot": 1, "1": 1, "true": 1, "yes": 1, "human": 0, "0": 0, "false": 0, "no": 0})
            .fillna(0).astype(int))
    .groupby("user_id", as_index=True)["label"].mean()
    .apply(lambda m: 1 if m >= 0.5 else 0)
)

# Split resolution per user (prefer train > val > test)
def _resolve_split(ss: pd.Series) -> str:
    s = set(ss)
    if "train" in s: return "train"
    if "val"   in s: return "val"
    if "test"  in s: return "test"
    return "test"

split_per_user = (
    df[["user_id", "split"]]
      .groupby("user_id", as_index=True)["split"]
      .apply(_resolve_split)
)

# Preserve your original user order (first occurrence in the CSV)
user_ids = (df.drop_duplicates(subset=["user_id"], keep="first")["user_id"].tolist())
uid_index = {u: i for i, u in enumerate(user_ids)}

# Build a compact user table for final prints
user_df = pd.DataFrame({
    "user_id": user_ids,
    "label": [int(lab.get(u, 0)) for u in user_ids],
    "split": [split_per_user.get(u, "test") for u in user_ids],
})

print(f"\nAfter deduplication: {len(user_df):,} unique users")
print("Final distribution using original splits (per USER):")
for split_name in ("train", "val", "test"):
    sd = user_df[user_df["split"].eq(split_name)]
    if len(sd) == 0:
        continue
    bots = int(sd["label"].sum())
    humans = len(sd) - bots
    print(f"  {split_name.title()}: {len(sd):,} users, Bots={bots:,} ({(bots/max(len(sd),1))*100:.1f}%), Humans={humans:,} ({(humans/max(len(sd),1))*100:.1f}%)")

print(f"\nFinal stratified sample: {len(user_df):,} users")  # keep your wording
print("Final distribution:")
for split_name in ("train", "val", "test"):
    sd = user_df[user_df["split"].eq(split_name)]
    if len(sd) == 0:
        continue
    bots = int(sd["label"].sum())
    humans = len(sd) - bots
    print(f"  {split_name.title()}: {len(sd):,} users, Bots={bots:,} ({(bots/max(len(sd),1))*100:.1f}%), Humans={humans:,} ({(humans/max(len(sd),1))*100:.1f}%)")

print('Extracting labels and splits')
# Convert to tensors (index order == user_ids order)
labels = torch.LongTensor(user_df["label"].tolist())

train_idx = torch.LongTensor([uid_index[u] for u in user_df[user_df["split"].eq("train")]["user_id"]])
valid_idx = torch.LongTensor([uid_index[u] for u in user_df[user_df["split"].eq("val")]["user_id"]])
test_idx  = torch.LongTensor([uid_index[u] for u in user_df[user_df["split"].eq("test")]["user_id"]])

# Save basic data (unchanged locations/names)
os.makedirs("./processed_data", exist_ok=True)
torch.save(train_idx, "./processed_data/train_idx.pt")
torch.save(valid_idx, "./processed_data/val_idx.pt")
torch.save(test_idx,  "./processed_data/test_idx.pt")
torch.save(labels,    "./processed_data/label.pt")

# ---------------------------------------------------------------------
# 3) Edges (5-relations) from your enhanced file (unchanged paths)
# ---------------------------------------------------------------------
print('Extracting edge_index & edge_type')
edge_index = []
edge_type  = []

if os.path.exists(EDGES_PATH):
    print(f"Loading enhanced 5-relation edges from: {EDGES_PATH}")
    edges_df = pd.read_csv(EDGES_PATH)

    # Normalize ids and relation names
    edges_df["source_id"] = edges_df["source_id"].astype(str).map(norm_user_id)
    edges_df["target_id"] = edges_df["target_id"].astype(str).map(norm_user_id)
    rel_col = "relation" if "relation" in edges_df.columns else (
        "edge_type" if "edge_type" in edges_df.columns else "type"
    )
    if rel_col in edges_df.columns:
        edges_df[rel_col] = edges_df[rel_col].astype(str).str.strip().str.lower()

    # Keep edges with both endpoints inside our users
    edges_df = edges_df[
        edges_df["source_id"].isin(uid_index) &
        edges_df["target_id"].isin(uid_index)
    ].copy()

    # Map relation string -> integer type id
    if rel_col in edges_df.columns:
        edges_df["relation_type"] = edges_df[rel_col].map(RELATION_TO_TYPE)
        edges_df = edges_df.dropna(subset=["relation_type"])
        edges_df["relation_type"] = edges_df["relation_type"].astype(int)
    else:
        edges_df["relation_type"] = 0  # default mentioned

    print(f"Processing {len(edges_df):,} edges...")
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df)):
        s = uid_index[row["source_id"]]
        t = uid_index[row["target_id"]]
        edge_index.append([s, t])
        edge_type.append(int(row["relation_type"]))
else:
    print("No enhanced edges found, creating minimal graph...")
    for i in range(min(100, max(len(user_ids)-1, 0))):
        edge_index.append([i, i+1])
        edge_type.append(0)  # mentioned

# Convert to tensors (matching original format)
if edge_index:
    edge_index = torch.LongTensor(edge_index).T  # [2, E]
    edge_type  = torch.LongTensor(edge_type)
else:
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    edge_type  = torch.zeros(0, dtype=torch.long)

# Save edges
torch.save(edge_index, './processed_data/edge_index.pt')
torch.save(edge_type,  './processed_data/edge_type.pt')

# Save user mapping for preprocess_2.py
torch.save({
    "user_ids": user_ids,
    "uid_index": uid_index,
    "csv_path": CSV_PATH,
}, './processed_data/user_mapping.pt')

# Summary
num_rel = infer_num_relations(edge_type)
bots = int(labels.sum().item())
humans = int((labels == 0).sum().item())

print('Finished preprocessing step 1:')
print(f'  • Users: {len(user_ids):,}')
print(f'  • Edges: {edge_index.shape[1]:,} ({num_rel} relation types)')
print(f'  • Labels: {bots:,} bots, {humans:,} humans')
print(f'  • Train/Val/Test: {len(train_idx)}/{len(valid_idx)}/{len(test_idx)}')
print(f'  • Data saved to: ./processed_data/')
