#!/usr/bin/env python
"""
Generate random negative drug-pair edges for each SE type.

Implements the same logic as:
    src/lloyd/create_false_edges_pykeen.py  (Lloyd et al., 2024)

For each SE type in test_edges.tsv, samples exactly N random drug pairs
not present in the known positive set (train ∪ test).  Results are saved
as one TSV file per SE type inside <dataset_dir>/false_edges/.

Run AFTER train.py and BEFORE evaluate.py.

Usage:
    python scripts/generate_negatives.py --dataset_dir outputs/default
    python scripts/generate_negatives.py --dataset_dir outputs/default --n_cores 8

Inputs (produced by train.py):
    <dataset_dir>/entity_to_id.json
    <dataset_dir>/relation_to_id.json
    <dataset_dir>/train_tf.pt
    <dataset_dir>/test_edges.tsv

Output:
    <dataset_dir>/false_edges/<SE_name>.tsv   (one per SE type, tab-separated)
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", required=True,
                    help="Output directory produced by train.py")
parser.add_argument("--n_cores", type=int, default=None,
                    help="CPU cores for parallel generation (default: all)")
parser.add_argument("--test_edges", default=None,
                    help="Path to test_edges.tsv (default: <dataset_dir>/test_edges.tsv)")
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)

# ---------------------------------------------------------------------------
# Load entity / relation mappings
# ---------------------------------------------------------------------------
with open(dataset_dir / "entity_to_id.json") as f:
    entity_to_id = json.load(f)
with open(dataset_dir / "relation_to_id.json") as f:
    relation_to_id = json.load(f)

# Drug nodes use the "drug:" prefix — equivalent to Lloyd's compound_IDs filter
compound_ids = [name for name in entity_to_id if name.startswith("drug:")]
print(f"  {len(compound_ids)} drug nodes available for negative sampling")

# ---------------------------------------------------------------------------
# Load test (holdout) edges
# ---------------------------------------------------------------------------
test_path = args.test_edges or dataset_dir / "test_edges.tsv"
test_df = pd.read_csv(test_path, header=None, sep="\t",
                      names=["h", "r", "t"], dtype=str)
print(f"  {len(test_df):,} test edges across {test_df['r'].nunique()} SE types")

# ---------------------------------------------------------------------------
# Load train edges from the saved TriplesFactory for exclusion
# ---------------------------------------------------------------------------
try:
    import torch
    train_tf = torch.load(dataset_dir / "train_tf.pt", weights_only=False)
    id_to_entity   = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    mt = train_tf.mapped_triples.numpy()
    train_str = pd.DataFrame({
        "h": [id_to_entity[x] for x in mt[:, 0]],
        "r": [id_to_relation[x] for x in mt[:, 1]],
        "t": [id_to_entity[x] for x in mt[:, 2]],
    })
    all_edges = pd.concat([test_df, train_str], ignore_index=True)
    print(f"  {len(train_str):,} train edges loaded for exclusion")
except Exception as e:
    print(f"  Warning: could not load train_tf.pt ({e}). Using test edges only for exclusion.")
    all_edges = test_df

false_edges_dir = dataset_dir / "false_edges"
false_edges_dir.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Negative sampling — identical logic to create_false_edges.py
# ---------------------------------------------------------------------------
def create_negative_edges(n_fake, pos_edges, entity_list):
    """
    Sample n_fake random drug pairs not in pos_edges.

    Exact replica of create_false_edges.py::create_negative_edges():
        - Excludes edges already in pos_edges (true positives: train ∪ test).
        - Excludes duplicates within neg_edges.
    """
    rel = pos_edges[0][1]
    neg_edges = []
    while len(neg_edges) < n_fake:
        head = np.random.choice(entity_list)
        tail = np.random.choice(entity_list)
        edge = [head, rel, tail]
        if edge not in pos_edges and edge not in neg_edges:
            neg_edges.append(edge)
    return neg_edges


# ---------------------------------------------------------------------------
# Build parallel argument list — one task per SE type not already on disk
# ---------------------------------------------------------------------------
parallel_args = []
se_names_ordered = []

for se_name, group in test_df.groupby("r"):
    false_edge_file = false_edges_dir / f"{se_name}.tsv"
    if false_edge_file.exists():
        print(f"  {se_name}: existing file found, skipping.")
        continue

    # Positive set = test ∪ train for this SE type
    se_all = all_edges[all_edges["r"] == se_name]
    pos_set = se_all[["h", "r", "t"]].values.tolist()
    n_needed = len(group)

    parallel_args.append((n_needed, pos_set, compound_ids))
    se_names_ordered.append(se_name)
    print(f"  {se_name}: need {n_needed} negatives")

# ---------------------------------------------------------------------------
# Generate in parallel (or serially if n_cores=1)
# ---------------------------------------------------------------------------
n_cores = args.n_cores or mp.cpu_count()
n_tasks = len(parallel_args)
print(f"\nGenerating negatives for {n_tasks} SE types on {n_cores} cores...")

np.random.seed(0)   # match Lloyd's global seed

if n_tasks == 0:
    print("Nothing to generate — all SE types already have false edge files.")
else:
    with mp.Pool(n_cores) as pool:
        results = pool.starmap(create_negative_edges, parallel_args)

    # Save one TSV per SE type
    print("Saving...")
    for se_name, neg_edges in zip(se_names_ordered, results):
        pd.DataFrame(neg_edges, columns=["h", "r", "t"]).to_csv(
            false_edges_dir / f"{se_name}.tsv",
            header=False, index=False, sep="\t",
        )

total_files = len(list(false_edges_dir.glob("*.tsv")))
print(f"\nDone. {total_files} files in {false_edges_dir}/")
print(f"\nNext step:")
print(f"  python scripts/evaluate.py --dataset_dir {dataset_dir} "
      f"--checkpoint {dataset_dir}/best_model.pt "
      f"--out_dir {dataset_dir}/results/")
