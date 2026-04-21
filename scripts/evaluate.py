#!/usr/bin/env python
"""
Evaluate a trained TF-Decagon model using the Lloyd et al. false-edge protocol.

Implements the same logic as:
    src/lloyd/assessment_pykeen.py  (Lloyd et al., 2024)

For each SE type in test_edges.tsv, loads the pre-computed false edges from
<dataset_dir>/false_edges/, scores all positives + negatives, and computes
AUROC, AUPRC, AP@50.  Results are saved as CSV with per-SE-type rows and a
median summary printed at the end.  Supports resuming after interruption via
--partial_results.

Run AFTER generate_negatives.py.

Usage:
    # Full evaluation
    python scripts/evaluate.py \\
        --checkpoint outputs/default/best_model.pt \\
        --dataset_dir outputs/default \\
        --out_dir     outputs/default/results/

    # Subset of SE types (fast sanity check)
    python scripts/evaluate.py \\
        --checkpoint outputs/default/best_model.pt \\
        --dataset_dir outputs/default \\
        --out_dir     outputs/default/results/ \\
        --ses C0000039 C0002871

    # Resume after interruption
    python scripts/evaluate.py \\
        --checkpoint outputs/default/best_model.pt \\
        --dataset_dir outputs/default \\
        --out_dir     outputs/default/results/ \\
        --partial_results outputs/default/results/results_temp.csv

Inputs (produced by train.py + generate_negatives.py):
    <dataset_dir>/best_model.pt          (or pass --checkpoint explicitly)
    <dataset_dir>/entity_to_id.json
    <dataset_dir>/relation_to_id.json
    <dataset_dir>/train_tf.pt
    <dataset_dir>/test_edges.tsv
    <dataset_dir>/false_edges/<SE>.tsv   (one per SE type)

Output:
    <out_dir>/results_full.csv           per-SE-type AUROC / AUPRC / AP@50
    <out_dir>/results_temp.csv           incremental save (removed on completion)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.lloyd.decagon_rank_metrics import apk


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True,
                    help="Path to model checkpoint (.pt) saved by train.py")
parser.add_argument("--dataset_dir", required=True,
                    help="Output directory produced by train.py")
parser.add_argument("--out_dir", required=True,
                    help="Directory to write results CSV files")
parser.add_argument("--partial_results", default=None,
                    help="Resume from a previous results_temp.csv")
parser.add_argument("--ses", nargs="+", default=None,
                    help="Subset of SE names to assess (default: all)")
parser.add_argument("--batch_size", type=int, default=4096,
                    help="Scoring batch size (increase if GPU has memory)")
parser.add_argument("--test_edges", default=None,
                    help="Path to test_edges.tsv (default: <dataset_dir>/test_edges.tsv)")
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
out_dir     = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print("Loading model...")
raw = torch.load(args.checkpoint, map_location=device, weights_only=False)

if isinstance(raw, dict):
    # State-dict only: reconstruct model from train_tf + infer embedding_dim.
    from pykeen.models import SimplE
    from pykeen.losses import CrossEntropyLoss

    train_tf = torch.load(
        dataset_dir / "train_tf.pt", weights_only=False, map_location="cpu"
    )
    emb_key = next(
        (k for k in raw if "entity_representations" in k and "weight" in k), None
    )
    if emb_key is None:
        n_ent = train_tf.num_entities
        emb_key = next(
            k for k, v in raw.items()
            if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[0] == n_ent
        )
    embedding_dim = raw[emb_key].shape[1]
    print(f"  State-dict checkpoint — rebuilding SimplE (embedding_dim={embedding_dim})")

    model = SimplE(
        triples_factory=train_tf,
        embedding_dim=embedding_dim,
        loss=CrossEntropyLoss(),
    ).to(device)
    missing, unexpected = model.load_state_dict(raw, strict=False)
    if unexpected:
        print(f"  Ignored keys (regularizer state): {unexpected}")
else:
    model = raw

model.eval()
print(f"  Model type: {type(model).__name__}")

# ---------------------------------------------------------------------------
# 2. Load entity / relation mappings
# ---------------------------------------------------------------------------
with open(dataset_dir / "entity_to_id.json") as f:
    entity_to_id = json.load(f)
with open(dataset_dir / "relation_to_id.json") as f:
    relation_to_id = json.load(f)

id_to_entity   = {v: k for k, v in entity_to_id.items()}
id_to_relation = {v: k for k, v in relation_to_id.items()}
print(f"  {len(entity_to_id):,} entities  |  {len(relation_to_id):,} relations")

# ---------------------------------------------------------------------------
# 3. Load test (holdout) edges
# ---------------------------------------------------------------------------
print("Loading test edges...")
test_path = args.test_edges or dataset_dir / "test_edges.tsv"
test_df = pd.read_csv(test_path, header=None, sep="\t",
                      names=["h", "r", "t"], dtype=str)

test_df["h_id"] = test_df["h"].map(entity_to_id)
test_df["r_id"] = test_df["r"].map(relation_to_id)
test_df["t_id"] = test_df["t"].map(entity_to_id)

n_missing = test_df[["h_id", "r_id", "t_id"]].isna().any(axis=1).sum()
if n_missing:
    print(f"  WARNING: {n_missing} test edges have unmapped entities — dropping.")
    test_df = test_df.dropna(subset=["h_id", "r_id", "t_id"])

test_df[["h_id", "r_id", "t_id"]] = test_df[["h_id", "r_id", "t_id"]].astype(int)
print(f"  {len(test_df):,} test edges across {test_df['r'].nunique()} SE types")

# ---------------------------------------------------------------------------
# 4. Optional SE subset filter
# ---------------------------------------------------------------------------
if args.ses is not None:
    test_df = test_df[test_df["r"].isin(args.ses)]
    print(f"  Filtered to {test_df['r'].nunique()} SE types ({len(test_df):,} edges)")

# ---------------------------------------------------------------------------
# 5. Scoring helper
# ---------------------------------------------------------------------------
false_edges_dir = dataset_dir / "false_edges"


def score_edges(h_ids, r_ids, t_ids):
    """Score (h, r, t) integer arrays. Returns flat numpy array of scores."""
    all_scores = []
    hrt = torch.stack([
        torch.tensor(h_ids, dtype=torch.long),
        torch.tensor(r_ids, dtype=torch.long),
        torch.tensor(t_ids, dtype=torch.long),
    ], dim=1).to(device)
    for start in range(0, len(hrt), args.batch_size):
        batch = hrt[start: start + args.batch_size]
        with torch.no_grad():
            scores = model.score_hrt(batch)  # (batch, 1)
        all_scores.append(scores.squeeze(1).cpu().numpy())
    return np.concatenate(all_scores)


# ---------------------------------------------------------------------------
# 6. Load or initialise results
# ---------------------------------------------------------------------------
if args.partial_results and os.path.exists(args.partial_results):
    results = pd.read_csv(args.partial_results)
    already_done = set(results["Relation"])
    print(f"  Resuming: {len(already_done)} SE types already assessed.")
else:
    results = pd.DataFrame(columns=["Relation", "AUROC", "AUPRC", "AP@50"])
    already_done = set()

# ---------------------------------------------------------------------------
# 7. Assessment loop — per SE type
# ---------------------------------------------------------------------------
se_types = test_df["r"].unique()
total    = len(se_types)
print(f"\nAssessing {total} SE types...")

for i, se_name in enumerate(se_types):

    if se_name in already_done:
        print(f"  [{i+1}/{total}] {se_name}: already done, skipping.")
        continue

    pos_group = test_df[test_df["r"] == se_name]
    pos_h = pos_group["h_id"].tolist()
    pos_r = pos_group["r_id"].tolist()
    pos_t = pos_group["t_id"].tolist()

    false_edge_file = false_edges_dir / f"{se_name}.tsv"
    if not false_edge_file.exists():
        print(f"  [{i+1}/{total}] {se_name}: no false edges file — run generate_negatives.py first.")
        continue

    neg_df = pd.read_csv(false_edge_file, header=None, sep="\t",
                         names=["h", "r", "t"], dtype=str)

    neg_h = [entity_to_id.get(h, -1)   for h in neg_df["h"]]
    neg_r = [relation_to_id.get(r, -1) for r in neg_df["r"]]
    neg_t = [entity_to_id.get(t, -1)   for t in neg_df["t"]]
    valid = [(h, r, t) for h, r, t in zip(neg_h, neg_r, neg_t)
             if h != -1 and r != -1 and t != -1]
    if not valid:
        print(f"  [{i+1}/{total}] {se_name}: no valid negatives, skipping.")
        continue
    neg_h, neg_r, neg_t = zip(*valid)

    all_h = list(pos_h) + list(neg_h)
    all_r = list(pos_r) + list(neg_r)
    all_t = list(pos_t) + list(neg_t)
    labels = [1] * len(pos_h) + [0] * len(neg_h)

    preds = score_edges(all_h, all_r, all_t)

    auroc = roc_auc_score(labels, preds)
    auprc = average_precision_score(labels, preds)

    # AP@50: Decagon's apk() applied at edge-object level
    pos_edges_list = list(zip(pos_h, pos_r, pos_t))
    all_edges_list = list(zip(all_h, all_r, all_t))
    ranked_idx     = np.argsort(preds)[::-1]
    ranked_edges   = [all_edges_list[j] for j in ranked_idx]
    ap50 = apk(pos_edges_list, ranked_edges, k=50)

    results.loc[len(results)] = [se_name, auroc, auprc, ap50]
    results.to_csv(out_dir / "results_temp.csv", index=False)
    print(f"  [{i+1}/{total}] {se_name}: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  AP@50={ap50:.4f}")

# ---------------------------------------------------------------------------
# 8. Save and summarise
# ---------------------------------------------------------------------------
final_path = out_dir / "results_full.csv"
results.to_csv(final_path, index=False)

temp_path = out_dir / "results_temp.csv"
if temp_path.exists() and len(results) == total:
    temp_path.unlink()

print(f"\nResults saved to {final_path}")
print(f"\nMedian over {len(results)} SE types:")
print(results[["AUROC", "AUPRC", "AP@50"]].median().round(4).to_string())
