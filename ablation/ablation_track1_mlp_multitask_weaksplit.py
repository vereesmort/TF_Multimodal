#!/usr/bin/env python
"""
ablation_track1_mlp_multitask_weaksplit.py
------------------------------------------
Multi-task MLP ablation with SimVec-style "weak nodes split" support.

Extends ablation_track1_mlp_multitask.py with two split modes selectable via
``--split_mode {random|weak}``:

  random  (default, original behaviour)
      Stratified random 80/10/10 train/val/test split over all drug pairs.

  weak    (SimVec weak nodes split)
      Reproduces the evaluation protocol from Lukashina et al. (2022).
      The N drugs with the fewest known polypharmacy edges are designated
      "weak/new drugs".  All triples involving at least one weak drug
      ("weak triples") are held out for test and validation only; the
      remaining triples form the training set.  This faithfully emulates
      how a model would perform on genuinely new, data-poor drugs and
      allows direct comparison to SimVec Table 1 numbers.

      Key parameters (match SimVec paper defaults):
        --weak_n  98      Number of weakest-degree drug nodes (≈1/6 of KG)
        --weak_val_frac 0.5  Fraction of weak triples → val (rest → test)

Architecture  (per drug pair)
-------------------------------
  ChemBERTa features (frozen, 768-d)    ESM-2 drug-level features (frozen, 640-d)
          ↓                                        ↓
  DrugBranchMLP  768→512→256            ProtBranchMLP  640→512→256
  (LayerNorm, GELU, Dropout 0.1)        (LayerNorm, GELU, Dropout 0.1)
          ↓                                        ↓
  drug_proj (256)             ←hstack→  prot_proj (256)
                    fused entity vector (512)
                           ↓
             pair operator on (e_A, e_B) → (1024) for sym/concat
                           ↓
             SharedTrunkMLP  1024 → 512 → 256
                           ↓
             SEHead  Linear(256, n_se)   ← one logit per SE type
                           ↓
             sigmoid(logits) → P(SE_i | drug_A, drug_B) for each i

Usage
-----
  # Original random split (unchanged):
  python ablation/ablation_track1_mlp_multitask_weaksplit.py \\
      --drug_raw_chemberta  data/cache/ablation_mlp/drug_raw_chemberta.pt \\
      --drug_raw_esm2       data/cache/ablation_mlp/drug_raw_esm2_via_targets.pt \\
      --drug_to_idx         data/cache/ablation_mlp/drug_to_idx.json \\
      --combo               data/raw/bio-decagon-combo.csv \\
      --output              results/ablation_mlp_mt \\
      --device              cuda

  # SimVec weak-nodes split (for hypothesis testing):
  python ablation/ablation_track1_mlp_multitask_weaksplit.py \\
      --drug_raw_chemberta  data/cache/ablation_mlp/drug_raw_chemberta.pt \\
      --drug_raw_esm2       data/cache/ablation_mlp/drug_raw_esm2_via_targets.pt \\
      --drug_to_idx         data/cache/ablation_mlp/drug_to_idx.json \\
      --combo               data/raw/bio-decagon-combo.csv \\
      --split_mode          weak \\
      --weak_n              98 \\
      --output              results/ablation_mlp_mt_weak \\
      --device              cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Model ─────────────────────────────────────────────────────────────────────

class BranchMLP(nn.Module):
    """Single trainable projection head: in_dim → hidden_dim → out_dim."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = out_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskAblationNet(nn.Module):
    """
    Multi-task variant: shared trunk + one linear output layer covering all SEs.

    Args
    ----
    drug_in_dim  : ChemBERTa input dim (default 768).
    prot_in_dim  : ESM-2 input dim (default 640).
    n_se         : Number of side-effect types (output classes).
    embed_dim    : Branch projection dim (default 256); fused entity = 2*embed_dim.
    trunk_hidden : Hidden dim of the shared trunk MLP (default 512).
    pair_repr    : "sym" | "concat" | "sum".
    dropout      : Applied in all sub-networks.
    mono_in_dim  : If given, adds a mono branch for condition E.
    has_ppi      : If True, adds a PPI branch for condition E.
    """

    def __init__(
        self,
        drug_in_dim:  int,
        prot_in_dim:  int,
        n_se:         int,
        embed_dim:    int   = 256,
        trunk_hidden: int   = 512,
        pair_repr:    str   = "sym",
        dropout:      float = 0.1,
        mono_in_dim:  int | None = None,
        has_ppi:      bool  = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.pair_repr = pair_repr
        self.n_se      = n_se

        self.drug_mlp = BranchMLP(drug_in_dim, embed_dim, dropout)
        self.prot_mlp = BranchMLP(prot_in_dim, embed_dim, dropout)

        self.mono_mlp = BranchMLP(mono_in_dim, embed_dim, dropout) if mono_in_dim else None
        self.ppi_mlp  = BranchMLP(prot_in_dim, embed_dim, dropout) if has_ppi   else None

        fused_dim = 2 * embed_dim
        pair_dim  = 2 * fused_dim if pair_repr in ("sym", "concat") else fused_dim

        # Shared trunk: compresses pair representation before the SE-specific head.
        # Two layers so the trunk can learn shared pharmacological patterns
        # (e.g., interaction mechanisms common across SE classes) before the
        # per-SE head specialises.
        self.trunk = nn.Sequential(
            nn.Linear(pair_dim, trunk_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        trunk_out = trunk_hidden // 2

        # One linear layer → n_se logits.
        # No activation here; sigmoid is applied in the loss or at eval time.
        self.se_head = nn.Linear(trunk_out, n_se)

    def _entity_embed(
        self,
        drug_raw:  torch.Tensor,
        prot_raw:  torch.Tensor,
        mono_raw:  torch.Tensor | None = None,
        ppi_raw:   torch.Tensor | None = None,
    ) -> torch.Tensor:
        d = self.drug_mlp(drug_raw)
        if self.mono_mlp is not None and mono_raw is not None:
            d = (d + self.mono_mlp(mono_raw)) * 0.5

        p = self.prot_mlp(prot_raw)
        if self.ppi_mlp is not None and ppi_raw is not None:
            p = (p + self.ppi_mlp(ppi_raw)) * 0.5

        return torch.cat([d, p], dim=-1)   # (B, 2*embed_dim)

    def forward(
        self,
        drug_a: torch.Tensor,
        prot_a: torch.Tensor,
        drug_b: torch.Tensor,
        prot_b: torch.Tensor,
        mono_a: torch.Tensor | None = None,
        mono_b: torch.Tensor | None = None,
        ppi_a:  torch.Tensor | None = None,
        ppi_b:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns raw logits of shape (B, n_se).
        Apply sigmoid for per-SE probabilities.
        """
        e_a = self._entity_embed(drug_a, prot_a, mono_a, ppi_a)
        e_b = self._entity_embed(drug_b, prot_b, mono_b, ppi_b)

        if self.pair_repr == "sym":
            pair_feat = torch.cat([e_a * e_b, (e_a - e_b).abs()], dim=1)
        elif self.pair_repr == "concat":
            pair_feat = torch.cat([e_a, e_b], dim=1)
        else:
            pair_feat = e_a + e_b

        shared = self.trunk(pair_feat)           # (B, trunk_hidden // 2)
        return self.se_head(shared)              # (B, n_se)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PSEPairMultiTaskDataset(Dataset):
    """
    Drug-pair multi-label dataset.

    One row per UNIQUE drug pair.  The label is a float32 vector of length n_se:
    label[i] = 1.0 if the pair exhibits SE i, 0.0 otherwise.

    Args
    ----
    s1_idx, s2_idx : (N,) int64 index arrays for unique drug pairs.
    label_matrix   : (N, n_se) float32 binary matrix.
    degrees        : (N,) float32 mean degree product for each pair.
    drug_feat      : (n_drugs, drug_dim) float32 tensor.
    prot_feat      : (n_drugs, prot_dim) float32 tensor.
    mono_feat      : optional (n_drugs, mono_dim).
    ppi_feat       : optional (n_drugs, prot_dim).
    """

    def __init__(
        self,
        s1_idx:       np.ndarray,
        s2_idx:       np.ndarray,
        label_matrix: np.ndarray,   # (N, n_se)
        degrees:      np.ndarray,
        drug_feat:    torch.Tensor,
        prot_feat:    torch.Tensor,
        mono_feat:    torch.Tensor | None = None,
        ppi_feat:     torch.Tensor | None = None,
    ) -> None:
        self.s1           = s1_idx.astype(np.int64)
        self.s2           = s2_idx.astype(np.int64)
        self.label_matrix = torch.tensor(label_matrix, dtype=torch.float32)
        self.degrees      = degrees.astype(np.float32)
        self.drug_feat    = drug_feat.float().cpu()
        self.prot_feat    = prot_feat.float().cpu()
        self.mono_feat    = mono_feat.float().cpu() if mono_feat is not None else None
        self.ppi_feat     = ppi_feat.float().cpu()  if ppi_feat  is not None else None

    def __len__(self) -> int:
        return len(self.s1)

    def __getitem__(self, i: int) -> dict:
        a, b = int(self.s1[i]), int(self.s2[i])
        item = {
            "drug_a":  self.drug_feat[a],
            "prot_a":  self.prot_feat[a],
            "drug_b":  self.drug_feat[b],
            "prot_b":  self.prot_feat[b],
            "label":   self.label_matrix[i],        # (n_se,)
            "degree":  torch.tensor(self.degrees[i], dtype=torch.float32),
        }
        if self.mono_feat is not None:
            item["mono_a"] = self.mono_feat[a]
            item["mono_b"] = self.mono_feat[b]
        if self.ppi_feat is not None:
            item["ppi_a"] = self.ppi_feat[a]
            item["ppi_b"] = self.ppi_feat[b]
        return item


# ── Pair / label-matrix construction ─────────────────────────────────────────

def build_multitask_pairs(
    df:          pd.DataFrame,
    drug_to_idx: dict[str, int],
    se_to_col:   dict[str, int],
    seed:        int,
    se_sample:   list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one row per unique drug pair from the positive edges, with a
    multi-label target vector.

    NOTE: This function does NOT add random negative pairs. In multi-task
    training the implicit negatives are the zero entries in the label matrix
    for each pair (i.e., every SE that pair does NOT exhibit). Explicit
    negative drug pairs can optionally be added via ``neg_ratio`` (see
    ``add_negative_pairs``). Whether to add them depends on your evaluation
    philosophy — see the architecture notes below.

    Returns
    -------
    s1_idx       : (N,) int64
    s2_idx       : (N,) int64
    label_matrix : (N, n_se) float32  — multi-hot
    degrees      : (N,) float32  — degree product for each pair
    """
    if se_sample is not None:
        df = df[df["Polypharmacy Side Effect"].isin(se_sample)]

    mask = df["STITCH 1"].isin(drug_to_idx) & df["STITCH 2"].isin(drug_to_idx)
    df   = df.loc[mask]

    stacked_all = pd.concat([df["STITCH 1"], df["STITCH 2"]], ignore_index=True)
    degree      = stacked_all.value_counts().to_dict()

    n_se     = len(se_to_col)
    pair_key = {}   # (min_idx, max_idx) → row in arrays

    for _, row in df.iterrows():
        a_str, b_str, se = row["STITCH 1"], row["STITCH 2"], row["Polypharmacy Side Effect"]
        if se not in se_to_col:
            continue
        ia, ib  = drug_to_idx[a_str], drug_to_idx[b_str]
        key     = (min(ia, ib), max(ia, ib))
        if key not in pair_key:
            pair_key[key] = len(pair_key)

    n_pairs      = len(pair_key)
    s1_idx       = np.empty(n_pairs, dtype=np.int64)
    s2_idx       = np.empty(n_pairs, dtype=np.int64)
    label_matrix = np.zeros((n_pairs, n_se), dtype=np.float32)
    degrees      = np.zeros(n_pairs, dtype=np.float32)

    # Reverse lookup to fill arrays efficiently
    for (ia, ib), row_i in pair_key.items():
        s1_idx[row_i]  = ia
        s2_idx[row_i]  = ib

    # Map drug idx back to string for degree lookup
    idx_to_drug = {v: k for k, v in drug_to_idx.items()}

    for row_i, (ia, ib) in enumerate(pair_key.keys()):
        a_str = idx_to_drug[ia]
        b_str = idx_to_drug[ib]
        degrees[row_i] = float(degree.get(a_str, 1) * degree.get(b_str, 1))

    for _, row in df.iterrows():
        a_str, b_str, se = row["STITCH 1"], row["STITCH 2"], row["Polypharmacy Side Effect"]
        if se not in se_to_col:
            continue
        ia, ib  = drug_to_idx[a_str], drug_to_idx[b_str]
        key     = (min(ia, ib), max(ia, ib))
        row_i   = pair_key[key]
        label_matrix[row_i, se_to_col[se]] = 1.0

    return s1_idx, s2_idx, label_matrix, degrees


def add_negative_pairs(
    s1_idx:       np.ndarray,
    s2_idx:       np.ndarray,
    label_matrix: np.ndarray,
    degrees:      np.ndarray,
    drug_to_idx:  dict[str, int],
    neg_ratio:    int,
    seed:         int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Append explicit all-zero drug pairs (no SE is active) at the given ratio
    relative to the number of positive pairs.

    Explicit negatives help the model calibrate the base rate of non-interaction
    but also inflate memory. Use neg_ratio=1 unless you have a strong reason to
    go higher — the dense multi-label formulation already exposes the model to
    many implicit negatives per row.
    """
    rng       = np.random.RandomState(seed)
    all_drugs = list(drug_to_idx.values())
    n_drugs   = len(all_drugs)
    n_pos     = len(s1_idx)
    n_neg     = n_pos * neg_ratio
    n_se      = label_matrix.shape[1]

    # Sample random pairs that are (very likely) not in the positive set
    pos_set = set(zip(s1_idx.tolist(), s2_idx.tolist()))
    neg_s1  = []
    neg_s2  = []
    tries   = 0
    while len(neg_s1) < n_neg and tries < n_neg * 10:
        ia = all_drugs[rng.randint(n_drugs)]
        ib = all_drugs[rng.randint(n_drugs)]
        if ia != ib and (ia, ib) not in pos_set and (ib, ia) not in pos_set:
            neg_s1.append(ia)
            neg_s2.append(ib)
        tries += 1

    n_actual = len(neg_s1)
    neg_labels  = np.zeros((n_actual, n_se), dtype=np.float32)
    neg_degrees = np.ones(n_actual, dtype=np.float32)   # degree info unavailable for fake pairs

    s1_out  = np.concatenate([s1_idx, np.array(neg_s1, dtype=np.int64)])
    s2_out  = np.concatenate([s2_idx, np.array(neg_s2, dtype=np.int64)])
    lm_out  = np.concatenate([label_matrix, neg_labels], axis=0)
    deg_out = np.concatenate([degrees, neg_degrees])
    return s1_out, s2_out, lm_out, deg_out


# ── SimVec-style weak nodes split ────────────────────────────────────────────

def make_weak_nodes_split(
    s1_idx:       np.ndarray,
    s2_idx:       np.ndarray,
    label_matrix: np.ndarray,
    degrees:      np.ndarray,
    drug_to_idx:  dict[str, int],
    weak_n:       int  = 98,
    val_frac:     float = 0.5,
    seed:         int   = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
    """
    Reproduce the SimVec "weak nodes split" (Lukashina et al., 2022, §Methods).

    Algorithm
    ---------
    1.  Compute the degree of each drug node from all known polypharmacy edges
        (i.e., how many unique pairs it appears in).
    2.  Select the N=weak_n drugs with the *smallest* degree → "weak drugs".
    3.  Mark every pair that involves at least one weak drug as a "weak triple".
    4.  Weak triples are split equally between validation and test (50/50 by
        default, matching SimVec's M/2 construction).
    5.  All remaining ("strong") triples form the training set.

    This faithfully emulates evaluating on new, data-poor drugs: the model
    never sees any pair involving a weak drug during training.

    Parameters
    ----------
    s1_idx, s2_idx : (N_pairs,) int64
        Drug index arrays from build_multitask_pairs.
    label_matrix   : (N_pairs, n_se) float32
    degrees        : (N_pairs,) float32  — degree product per pair (for logging)
    drug_to_idx    : mapping drug_str → row index
    weak_n         : Number of lowest-degree nodes to treat as "new drugs".
                     Paper default = 98 (≈1/6 of the 645-drug KG).
    val_frac       : Fraction of weak triples put into validation (rest → test).
                     Paper uses 0.5 (M/2 val, M/2 test).
    seed           : RNG seed for shuffling weak triples before the val/test cut.

    Returns
    -------
    Six index arrays (train / val / test, each with a companion "is_weak" mask):
        train_idx, val_idx, test_idx  — indices into s1_idx / label_matrix
        is_weak_train, is_weak_val, is_weak_test  — bool arrays (all False for
            train by construction; useful for per-group eval on val/test)
        weak_drug_indices  — the set of drug row-indices designated as weak
            (exposed for logging / downstream analysis)
    """
    rng = np.random.RandomState(seed)

    # ── Step 1: per-drug degree (number of pairs each drug appears in) ────────
    n_pairs = len(s1_idx)
    drug_pair_count: defaultdict[int, int] = defaultdict(int)
    for ia, ib in zip(s1_idx, s2_idx):
        drug_pair_count[int(ia)] += 1
        drug_pair_count[int(ib)] += 1

    # Fill in zero-degree drugs that exist in drug_to_idx but have no pairs
    # (can happen after min_edges filtering)
    all_drug_indices = set(drug_to_idx.values())
    for d in all_drug_indices:
        if d not in drug_pair_count:
            drug_pair_count[d] = 0

    # ── Step 2: select N weakest drugs ───────────────────────────────────────
    sorted_by_degree = sorted(drug_pair_count.items(), key=lambda x: x[1])
    weak_drug_indices = np.array(
        [d for d, _ in sorted_by_degree[:weak_n]], dtype=np.int64
    )
    weak_set = set(weak_drug_indices.tolist())

    # ── Step 3: label pairs as weak / strong ─────────────────────────────────
    is_weak_pair = np.array(
        [int(ia) in weak_set or int(ib) in weak_set
         for ia, ib in zip(s1_idx, s2_idx)],
        dtype=bool,
    )
    weak_pair_idx  = np.where( is_weak_pair)[0]
    strong_pair_idx = np.where(~is_weak_pair)[0]

    # ── Step 4: split weak triples into val / test ────────────────────────────
    # Shuffle so val and test see a representative mix of SEs
    perm = rng.permutation(len(weak_pair_idx))
    weak_pair_idx_shuffled = weak_pair_idx[perm]

    n_val = max(1, int(round(len(weak_pair_idx_shuffled) * val_frac)))
    val_idx  = weak_pair_idx_shuffled[:n_val]
    test_idx = weak_pair_idx_shuffled[n_val:]

    # ── Step 5: all strong pairs → train ─────────────────────────────────────
    train_idx = strong_pair_idx

    # Companion is_weak masks (always False for train by construction)
    is_weak_train = np.zeros(len(train_idx), dtype=bool)
    is_weak_val   = np.ones(len(val_idx),   dtype=bool)
    is_weak_test  = np.ones(len(test_idx),  dtype=bool)

    return (
        train_idx, val_idx, test_idx,
        is_weak_train, is_weak_val, is_weak_test,
        weak_drug_indices,
    )


def log_weak_split_stats(
    s1_idx:           np.ndarray,
    s2_idx:           np.ndarray,
    label_matrix:     np.ndarray,
    train_idx:        np.ndarray,
    val_idx:          np.ndarray,
    test_idx:         np.ndarray,
    weak_drug_indices: np.ndarray,
    drug_to_idx:      dict[str, int],
) -> None:
    """Print a concise summary of the weak-nodes split to stdout."""
    n_drugs = len(drug_to_idx)
    n_weak  = len(weak_drug_indices)

    idx_to_drug = {v: k for k, v in drug_to_idx.items()}
    weak_set    = set(weak_drug_indices.tolist())

    # Degree of each weak drug in the full pair set
    drug_pair_count: defaultdict[int, int] = defaultdict(int)
    for ia, ib in zip(s1_idx, s2_idx):
        drug_pair_count[int(ia)] += 1
        drug_pair_count[int(ib)] += 1

    weak_degrees = sorted([drug_pair_count.get(d, 0) for d in weak_set])
    strong_degrees = sorted(
        [cnt for d, cnt in drug_pair_count.items() if d not in weak_set]
    )

    print(
        f"\n{'─'*60}\n"
        f"  Weak nodes split summary\n"
        f"{'─'*60}\n"
        f"  Total drugs:        {n_drugs}\n"
        f"  Weak drugs (N):     {n_weak}  ({100*n_weak/n_drugs:.1f}% of KG)\n"
        f"  Weak drug degree    min={weak_degrees[0]}  "
        f"median={int(np.median(weak_degrees))}  max={weak_degrees[-1]}\n"
        f"  Strong drug degree  min={strong_degrees[0]}  "
        f"median={int(np.median(strong_degrees))}  max={strong_degrees[-1]}\n"
        f"\n"
        f"  Pairs  total:       {len(s1_idx):,}\n"
        f"  Pairs  train:       {len(train_idx):,}  (strong only)\n"
        f"  Pairs  val:         {len(val_idx):,}    (weak only)\n"
        f"  Pairs  test:        {len(test_idx):,}   (weak only)\n"
        f"\n"
        f"  Positive cells train: {int(label_matrix[train_idx].sum()):,}\n"
        f"  Positive cells val:   {int(label_matrix[val_idx].sum()):,}\n"
        f"  Positive cells test:  {int(label_matrix[test_idx].sum()):,}\n"
        f"{'─'*60}"
    )


# ── Reuse unchanged helpers from binary script ────────────────────────────────

def make_condition_tensors(
    condition:   str,
    n_drugs:     int,
    drug_raw:    torch.Tensor,
    prot_raw:    torch.Tensor,
    mono_raw:    torch.Tensor | None,
    ppi_raw:     torch.Tensor | None,
    xavier_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    drug_dim = drug_raw.shape[1]
    prot_dim = prot_raw.shape[1]

    if condition == "A":
        return drug_raw, prot_raw, None, None
    elif condition == "B":
        return torch.zeros(n_drugs, drug_dim), prot_raw, None, None
    elif condition == "C":
        return drug_raw, torch.zeros(n_drugs, prot_dim), None, None
    elif condition == "D":
        g_drug = torch.Generator(); g_drug.manual_seed(xavier_seed)
        g_prot = torch.Generator(); g_prot.manual_seed(xavier_seed + 1)
        bound_drug = math.sqrt(3.0 / drug_dim)
        bound_prot = math.sqrt(3.0 / prot_dim)
        rand_drug  = torch.empty(n_drugs, drug_dim).uniform_(-bound_drug, bound_drug, generator=g_drug)
        rand_prot  = torch.empty(n_drugs, prot_dim).uniform_(-bound_prot, bound_prot, generator=g_prot)
        return rand_drug, rand_prot, None, None
    elif condition == "E":
        if mono_raw is None or ppi_raw is None:
            raise ValueError("Condition E requires --drug_raw_mono and --drug_raw_ppi.")
        return drug_raw, prot_raw, mono_raw, ppi_raw
    else:
        raise ValueError(f"Unknown condition: {condition!r}")


def assign_tiers(se_counts: pd.Series, n_tiers: int = 5) -> dict:
    sorted_ses = se_counts.sort_values()
    tier_size  = len(sorted_ses) // n_tiers
    tier_map   = {}
    labels     = [f"T{n_tiers}_rare", f"T{n_tiers-1}", "T3", "T2", "T1_frequent"]
    for i, (se, _) in enumerate(sorted_ses.items()):
        tier_idx      = min(i // tier_size, n_tiers - 1)
        tier_map[se]  = labels[tier_idx]
    return tier_map


def _degree_score_corrs(degrees: np.ndarray, scores: np.ndarray):
    if len(degrees) < 3 or np.std(degrees) == 0 or np.std(scores) == 0:
        return float("nan"), float("nan")
    return (
        float(np.corrcoef(degrees, scores)[0, 1]),
        float(spearmanr(degrees, scores).statistic),
    )


# ── Per-SE positive-weight computation ───────────────────────────────────────

def compute_pos_weights(label_matrix: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Per-SE pos_weight = n_neg / n_pos.

    Passed to F.binary_cross_entropy_with_logits so that rare SEs (T5) receive
    the same total gradient mass as common ones (T1).  Clipped to [1, 1000] to
    prevent numerical instability for extremely rare SEs.
    """
    n     = label_matrix.shape[0]
    n_pos = label_matrix.sum(axis=0).clip(min=1)
    n_neg = n - n_pos
    pos_w = (n_neg / n_pos).clip(1.0, 1000.0)
    return torch.tensor(pos_w, dtype=torch.float32, device=device)


# ── Training & evaluation ─────────────────────────────────────────────────────

def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _forward(model: MultiTaskAblationNet, batch: dict) -> torch.Tensor:
    return model(
        batch["drug_a"], batch["prot_a"],
        batch["drug_b"], batch["prot_b"],
        mono_a=batch.get("mono_a"), mono_b=batch.get("mono_b"),
        ppi_a =batch.get("ppi_a"),  ppi_b =batch.get("ppi_b"),
    )


def train_one_epoch(
    model:      MultiTaskAblationNet,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scheduler:  OneCycleLR,
    device:     torch.device,
    pos_weight: torch.Tensor,   # (n_se,)
) -> float:
    """Returns mean BCE loss (averaged over all cells in the batch)."""
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch  = _batch_to_device(batch, device)
        logits = _forward(model, batch)            # (B, n_se)
        loss   = F.binary_cross_entropy_with_logits(
            logits, batch["label"],
            pos_weight=pos_weight,                  # broadcast over batch dim
            reduction="mean",
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_predictions(
    model:    MultiTaskAblationNet,
    loader:   DataLoader,
    device:   torch.device,
    n_se:     int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      label_matrix : (N, n_se) float32 ground-truth
      score_matrix : (N, n_se) float32 sigmoid probabilities
      degrees      : (N,) float32
    """
    model.eval()
    all_labels  = []
    all_scores  = []
    all_degrees = []

    for batch in loader:
        batch  = _batch_to_device(batch, device)
        logits = _forward(model, batch)            # (B, n_se)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(batch["label"].cpu().numpy())
        all_scores.append(probs)
        all_degrees.extend(batch["degree"].cpu().numpy().tolist())

    label_mat  = np.vstack(all_labels)   # (N, n_se)
    score_mat  = np.vstack(all_scores)   # (N, n_se)
    degrees    = np.array(all_degrees, dtype=np.float32)
    return label_mat, score_mat, degrees


def evaluate_multitask(
    label_mat:  np.ndarray,         # (N, n_se)
    score_mat:  np.ndarray,         # (N, n_se)
    degrees:    np.ndarray,         # (N,)
    se_list:    list[str],          # ordered SE CUIs
    tier_map:   dict,
    condition:  str,
    threshold:  float = 0.5,
    is_weak:    np.ndarray | None = None,  # (N,) bool — weak-split flag
) -> pd.DataFrame:
    """Per-SE AUROC / AUPRC / AP@50 / degree-bias metrics.

    When ``is_weak`` is provided (weak nodes split mode), each SE row is
    additionally annotated with the split group ("weak" or "strong").
    The whole test set is always evaluated together; ``is_weak`` is a
    metadata column only, useful for downstream groupby analysis.
    """
    results = []
    for col_i, se in enumerate(se_list):
        yt = label_mat[:, col_i]
        ys = score_mat[:, col_i]

        if yt.sum() == 0 or yt.sum() == len(yt):
            continue

        auroc  = roc_auc_score(yt, ys)
        auprc  = average_precision_score(yt, ys)
        top50  = np.argsort(ys)[::-1][:50]
        ap50   = yt[top50].sum() / 50.0

        pos_m  = yt == 1
        neg_m  = yt == 0
        pears_pos, spear_pos = _degree_score_corrs(degrees[pos_m], ys[pos_m])
        pears_neg, spear_neg = _degree_score_corrs(degrees[neg_m], ys[neg_m])

        yp = ys >= threshold

        row: dict = {
            "condition":                      condition,
            "se":                             se,
            "tier":                           tier_map.get(se, "unknown"),
            "auroc":                          auroc,
            "auprc":                          auprc,
            "ap50":                           ap50,
            "n_pos":                          int(yt.sum()),
            "n_neg":                          int((yt == 0).sum()),
            "n_total":                        len(yt),
            "threshold":                      threshold,
            "tp":                             int(( yt.astype(bool) &  yp).sum()),
            "fp":                             int((~yt.astype(bool) &  yp).sum()),
            "tn":                             int((~yt.astype(bool) & ~yp).sum()),
            "fn":                             int(( yt.astype(bool) & ~yp).sum()),
            "degree_score_corr_pearson":      pears_pos,
            "degree_score_corr_spearman":     spear_pos,
            "degree_score_corr_pearson_neg":  pears_neg,
            "degree_score_corr_spearman_neg": spear_neg,
        }

        # ── Weak-split annotations ────────────────────────────────────────────
        # In weak mode the test set contains *only* weak pairs, so is_weak is
        # all-True on test.  We annotate the column for clarity and also
        # compute per-group AUROC/AUPRC where both groups have ≥1 positive.
        if is_weak is not None:
            row["split_mode"]   = "weak"
            row["n_weak_pairs"] = int(is_weak.sum())
            # per-group metrics (weak vs strong) — available on val which may
            # contain a mix; on test is_weak is uniform so only one group exists
            for group_name, gmask in [("weak", is_weak), ("strong", ~is_weak)]:
                gyt = yt[gmask]; gys = ys[gmask]
                if gyt.sum() > 0 and gyt.sum() < len(gyt):
                    row[f"auroc_{group_name}"] = roc_auc_score(gyt, gys)
                    row[f"auprc_{group_name}"] = average_precision_score(gyt, gys)
                else:
                    row[f"auroc_{group_name}"] = float("nan")
                    row[f"auprc_{group_name}"] = float("nan")
        else:
            row["split_mode"]   = "random"
            row["n_weak_pairs"] = 0

        results.append(row)

    return pd.DataFrame(results)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-task MLP ablation — one model, all SEs simultaneously"
    )

    # Input tensors
    p.add_argument("--drug_raw_chemberta", required=True)
    p.add_argument("--drug_raw_mono",      default=None)
    p.add_argument("--drug_raw_esm2",      required=True)
    p.add_argument("--drug_raw_ppi",       default=None)
    p.add_argument("--drug_to_idx",        required=True)

    # Data
    p.add_argument("--combo",     default="data/raw/bio-decagon-combo.csv")
    p.add_argument("--min_edges", type=int, default=500)
    p.add_argument("--neg_ratio", type=int, default=0,
                   help="Explicit negative drug pairs per positive pair. "
                        "0 = rely solely on implicit negatives (cells with 0 in label matrix).")
    p.add_argument("--seed",      type=int, default=42)

    # Split mode
    p.add_argument(
        "--split_mode",
        choices=["random", "weak"],
        default="random",
        help=(
            "random: stratified random train/val/test (original behaviour).  "
            "weak: SimVec-style weak-nodes split — the N lowest-degree drugs "
            "are held out of training entirely; their pairs form val+test."
        ),
    )
    p.add_argument(
        "--weak_n",
        type=int,
        default=98,
        help=(
            "Number of lowest-degree drug nodes to treat as 'new/weak drugs' "
            "in --split_mode weak.  Paper default = 98 (≈1/6 of the 645-drug KG). "
            "Increase for a harder evaluation, decrease for a smaller held-out set."
        ),
    )
    p.add_argument(
        "--weak_val_frac",
        type=float,
        default=0.5,
        help=(
            "Fraction of weak triples allocated to *validation* "
            "(remainder → test).  Paper uses 0.5 (M/2 val, M/2 test)."
        ),
    )

    # Conditions
    p.add_argument("--conditions", default="A,B,C,D,E")

    # SE sampling
    p.add_argument("--n_se_sample",  type=int,  default=None)
    p.add_argument("--se_offset",    type=int,  default=0)
    p.add_argument("--only_se_ids",  default=None)

    # Model
    p.add_argument("--embed_dim",    type=int,   default=256)
    p.add_argument("--trunk_hidden", type=int,   default=512,
                   help="Hidden dim of the shared trunk MLP")
    p.add_argument("--pair_repr",    choices=["sym", "concat", "sum"], default="sym")
    p.add_argument("--dropout",      type=float, default=0.1)

    # Training
    p.add_argument("--max_epochs",   type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=512,
                   help="Smaller default than binary because each row is a (n_se,) label vector")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=5)
    p.add_argument("--val_frac",     type=float, default=0.1)
    p.add_argument("--test_frac",    type=float, default=0.2)
    p.add_argument("--num_workers",  type=int,   default=0)

    # Output
    p.add_argument("--output",           default="results/ablation_mlp_mt")
    p.add_argument("--save_checkpoints", action="store_true", default=False)

    # wandb
    p.add_argument("--wandb",         action="store_true", default=False)
    p.add_argument("--wandb_project", default="pse_ablation_mlp_mt")
    p.add_argument("--wandb_entity",  default=None)
    p.add_argument("--wandb_run_name",default=None)
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--wandb_tags",    default=None)

    p.add_argument("--device", default="cpu")
    return p.parse_args()


# ── Data helpers (unchanged from binary script) ───────────────────────────────

def _load_tensor(path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    return torch.load(path, map_location="cpu").float()


def _load_combo(combo_path: str, min_edges: int) -> pd.DataFrame:
    df        = pd.read_csv(combo_path, encoding="latin-1")
    se_counts = df.groupby("Polypharmacy Side Effect").size()
    valid     = se_counts[se_counts >= min_edges].index
    df        = df[df["Polypharmacy Side Effect"].isin(valid)]
    print(
        f"Combo after min_edges={min_edges}: {len(df):,} rows, "
        f"{len(valid)} SEs, "
        f"{pd.concat([df['STITCH 1'], df['STITCH 2']]).nunique()} drugs"
    )
    return df


def _verify_drug_alignment(loaded_idx, combo_idx):
    if loaded_idx != combo_idx:
        extra_l = set(loaded_idx) - set(combo_idx)
        extra_c = set(combo_idx)  - set(loaded_idx)
        raise ValueError(
            "drug_to_idx.json does not match the filtered combo.\n"
            f"  Only in JSON: {len(extra_l)}; only in combo: {len(extra_c)}\n"
            "Re-run precompute_embeddings_ablation_mlp.py with the same flags."
        )


def _parse_se_sample(args, ordered_se):
    if args.only_se_ids:
        requested = {s.strip() for s in args.only_se_ids.replace(",", " ").split() if s.strip()}
        se_sample = [se for se in ordered_se if se in requested]
        if not se_sample:
            raise RuntimeError("None of --only_se_ids CUIs found.")
        return se_sample
    if args.n_se_sample is not None:
        start = max(0, args.se_offset)
        end   = min(start + args.n_se_sample, len(ordered_se))
        return ordered_se[start:end]
    return None


def _append_results(path, df, first):
    df.to_csv(path, mode="w" if first else "a", header=first, index=False)
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)

    print("=" * 60)
    print("Track 1 MLP Multi-Task Ablation")
    print("=" * 60)

    # ── Load tensors ──────────────────────────────────────────────────────────
    drug_raw = _load_tensor(args.drug_raw_chemberta)
    prot_raw = _load_tensor(args.drug_raw_esm2)
    mono_raw = _load_tensor(args.drug_raw_mono)
    ppi_raw  = _load_tensor(args.drug_raw_ppi)

    with open(args.drug_to_idx) as f:
        drug_to_idx: dict[str, int] = json.load(f)

    # ── Load + filter combo ───────────────────────────────────────────────────
    df               = _load_combo(args.combo, args.min_edges)
    all_drugs_sorted = sorted(pd.concat([df["STITCH 1"], df["STITCH 2"]]).unique())
    combo_idx        = {d: i for i, d in enumerate(all_drugs_sorted)}
    _verify_drug_alignment(drug_to_idx, combo_idx)
    n_drugs = len(drug_to_idx)

    se_counts  = df.groupby("Polypharmacy Side Effect").size()
    tier_map   = assign_tiers(se_counts)
    ordered_se = se_counts.sort_values(ascending=False).index.tolist()
    se_sample  = _parse_se_sample(args, ordered_se)

    # Build the SE → column mapping once for the entire run
    active_ses = se_sample if se_sample is not None else ordered_se
    se_to_col  = {se: col for col, se in enumerate(active_ses)}
    n_se       = len(active_ses)
    print(f"Active SEs: {n_se}  (output head size = {n_se})")

    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    conditions = [
        c for c in conditions
        if not (c == "E" and (mono_raw is None or ppi_raw is None))
    ]
    print(f"Conditions: {conditions}")

    # ── Build the pair / label-matrix arrays (shared across conditions) ───────
    print("\nBuilding multi-label pair matrix ...")
    s1, s2, lm, degrees = build_multitask_pairs(
        df, drug_to_idx, se_to_col, seed=args.seed, se_sample=se_sample
    )
    if args.neg_ratio > 0:
        s1, s2, lm, degrees = add_negative_pairs(
            s1, s2, lm, degrees, drug_to_idx, args.neg_ratio, args.seed
        )

    n_pairs = len(s1)
    n_pos   = int(lm.sum())
    n_cells = n_pairs * n_se
    print(
        f"  Pairs: {n_pairs:,}  |  Label-matrix cells: {n_cells:,}  "
        f"|  Positive cells: {n_pos:,}  ({100*n_pos/n_cells:.2f} % fill)"
    )

    # Stratified split on whether a pair has ANY active SE (1) or not (0)
    pair_has_se = (lm.sum(axis=1) > 0).astype(int)
    idx         = np.arange(n_pairs, dtype=np.int64)

    if args.split_mode == "weak":
        # ── SimVec weak-nodes split ───────────────────────────────────────────
        (
            train_idx, val_idx, test_idx,
            is_weak_train, is_weak_val, is_weak_test,
            weak_drug_indices,
        ) = make_weak_nodes_split(
            s1, s2, lm, degrees, drug_to_idx,
            weak_n   = args.weak_n,
            val_frac = args.weak_val_frac,
            seed     = args.seed,
        )
        subtrain_idx = train_idx   # no inner val split; val_idx already set
        log_weak_split_stats(
            s1, s2, lm,
            train_idx, val_idx, test_idx,
            weak_drug_indices, drug_to_idx,
        )
        # Save weak drug mapping for downstream analysis
        idx_to_drug = {v: k for k, v in drug_to_idx.items()}
        weak_drugs_out = [idx_to_drug[int(d)] for d in weak_drug_indices]
        pd.DataFrame({"drug": weak_drugs_out}).to_csv(
            output_dir / "weak_drug_ids.csv", index=False
        )
        print(f"  Weak drug IDs saved to: {output_dir / 'weak_drug_ids.csv'}")

    else:
        # ── Original random split ─────────────────────────────────────────────
        train_idx, test_idx = train_test_split(
            idx, test_size=args.test_frac, stratify=pair_has_se, random_state=args.seed
        )
        subtrain_idx, val_idx = train_test_split(
            train_idx,
            test_size    = args.val_frac,
            stratify     = pair_has_se[train_idx],
            random_state = args.seed,
        )
        is_weak_val  = None
        is_weak_test = None
    print(
        f"  Split ({args.split_mode}): train={len(subtrain_idx):,}  val={len(val_idx):,}  "
        f"test={len(test_idx):,}"
    )

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb_run  = None
    if args.wandb:
        import wandb
        tags = [t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_run_name, tags=tags or None,
            mode="offline" if args.wandb_offline else "online",
            config=vars(args) | {
                "n_se": n_se, "n_pairs": n_pairs,
                "split_mode": args.split_mode,
                "weak_n": args.weak_n if args.split_mode == "weak" else None,
            },
        )

    results_csv = output_dir / f"ablation_mlp_mt_{args.split_mode}_results_per_se.csv"
    curves_csv  = output_dir / f"ablation_mlp_mt_{args.split_mode}_training_curves.csv"
    first_write = True
    first_curve = True
    wandb_step  = 0

    # ── Per-condition loop ────────────────────────────────────────────────────
    for condition in conditions:
        print(f"\n{'─'*40}\nCondition {condition}")

        drug_feat, prot_feat, mono_feat, ppi_feat = make_condition_tensors(
            condition, n_drugs,
            drug_raw, prot_raw, mono_raw, ppi_raw,
            xavier_seed=args.seed,
        )

        def _make_ds(idx_sel):
            return PSEPairMultiTaskDataset(
                s1[idx_sel], s2[idx_sel],
                lm[idx_sel], degrees[idx_sel],
                drug_feat, prot_feat, mono_feat, ppi_feat,
            )

        train_ds = _make_ds(subtrain_idx)
        val_ds   = _make_ds(val_idx)
        test_ds  = _make_ds(test_idx)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader  = DataLoader(val_ds,  batch_size=args.batch_size * 2, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

        # Per-SE positive weights (computed from training split only to avoid leakage)
        pos_weight = compute_pos_weights(lm[subtrain_idx], device)

        model = MultiTaskAblationNet(
            drug_in_dim  = drug_raw.shape[1],
            prot_in_dim  = prot_raw.shape[1],
            n_se         = n_se,
            embed_dim    = args.embed_dim,
            trunk_hidden = args.trunk_hidden,
            pair_repr    = args.pair_repr,
            dropout      = args.dropout,
            mono_in_dim  = mono_raw.shape[1] if (mono_feat is not None and mono_raw is not None) else None,
            has_ppi      = (ppi_feat is not None),
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model: {n_params:,} trainable parameters  |  Output head: {n_se} SEs")

        optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_steps = args.max_epochs * len(train_loader)
        scheduler   = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

        best_val_auroc   = -1.0
        best_epoch       = 0
        patience_counter = 0
        ckpt_path        = ckpt_dir / f"best_{condition}.pt"
        epoch_rows       = []

        print(f"  Training (max {args.max_epochs} epochs, patience {args.patience}) ...")
        for epoch in range(args.max_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, pos_weight)

            lm_v, sm_v, _ = eval_predictions(model, val_loader, device, n_se)
            # Macro-average AUROC across SEs that have both classes in the val split
            aurocs = []
            for col_i in range(n_se):
                yt = lm_v[:, col_i]
                if yt.sum() == 0 or yt.sum() == len(yt):
                    continue
                aurocs.append(roc_auc_score(yt, sm_v[:, col_i]))
            val_auroc = float(np.mean(aurocs)) if aurocs else 0.5

            is_best = val_auroc > best_val_auroc
            print(f"  Epoch {epoch+1:3d}  train_loss={train_loss:.4f}  "
                  f"val_macro_auroc={val_auroc:.4f}{'  *' if is_best else ''}")

            epoch_rows.append({
                "condition": condition, "seed": args.seed, "epoch": epoch + 1,
                "train_loss": train_loss, "val_macro_auroc": val_auroc, "is_best": is_best,
            })

            if is_best:
                best_val_auroc = val_auroc; best_epoch = epoch + 1; patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        first_curve = _append_results(curves_csv, pd.DataFrame(epoch_rows), first_curve)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        lm_t, sm_t, degs_t = eval_predictions(model, test_loader, device, n_se)

        # In weak mode, is_weak_test is all-True (test = weak pairs only).
        # In random mode, is_weak_test is None (no weak annotation).
        results_df = evaluate_multitask(
            lm_t, sm_t, degs_t, active_ses, tier_map, condition,
            is_weak=is_weak_test,
        )
        results_df["best_val_macro_auroc"] = best_val_auroc
        results_df["best_epoch"]           = best_epoch
        first_write = _append_results(results_csv, results_df, first_write)

        overall_auroc = float(results_df["auroc"].mean())
        overall_auprc = float(results_df["auprc"].mean())
        print(f"  Test  macro-AUROC: {overall_auroc:.4f}  macro-AUPRC: {overall_auprc:.4f}")

        if wandb_run is not None:
            import wandb
            wandb.log({
                "overall/macro_auroc": overall_auroc,
                "overall/macro_auprc": overall_auprc,
                "meta/condition":      condition,
            }, step=wandb_step)
            wandb_step += 1

        if not args.save_checkpoints:
            ckpt_path.unlink(missing_ok=True)

        del model, optimizer, scheduler, train_ds, val_ds, test_ds
        del train_loader, val_loader, test_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    if not results_csv.exists():
        print("\nNo results written.")
        return

    all_df  = pd.read_csv(results_csv)
    summary = (
        all_df.groupby(["condition", "tier"])
        .agg(
            median_auroc = ("auroc", "median"),
            median_auprc = ("auprc", "median"),
            median_ap50  = ("ap50",  "median"),
            n_ses        = ("se",    "count"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / f"ablation_mlp_mt_{args.split_mode}_summary.csv", index=False)

    print("\n" + "=" * 60)
    print(f"SUMMARY — median AUPRC by condition and tier  [{args.split_mode} split]")
    print("=" * 60)
    pivot = summary.pivot_table(index="tier", columns="condition", values="median_auprc").round(4)
    print(pivot.to_string())
    print(f"\nResults saved to: {output_dir}")

    if wandb_run is not None:
        import wandb
        wandb.log({"tables/summary": wandb.Table(dataframe=summary)})
        wandb.finish()


if __name__ == "__main__":
    main()
