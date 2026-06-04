#!/usr/bin/env python
"""
ablation_track1_mlp.py
----------------------
Track 1 ablation with TRAINABLE projection MLPs.

Unlike ablation_track1_ml.py (fixed random projection → XGBoost), this script
trains small projection MLP heads end-to-end on the PSE classification task.
The pretrained language models (ChemBERTa, ESM-2) remain FROZEN throughout.
Only the projection heads and the classifier MLP are updated by backpropagation.

Architecture  (per drug pair, per SE)
--------------------------------------
  ChemBERTa features (frozen, 768-d)    ESM-2 drug-level features (frozen, 640-d)
          ↓                                        ↓
  DrugBranchMLP  768→512→256            ProtBranchMLP  640→512→256
  (LayerNorm, GELU, Dropout 0.1)        (LayerNorm, GELU, Dropout 0.1)
          ↓                                        ↓
  drug_proj (256)             ←hstack→  prot_proj (256)
                    fused entity vector  (512)
                           ↓
             pair operator on (e_A, e_B) → (1024) for sym/concat
                           ↓
             ClassifierMLP  1024→256→128→1
                           ↓
             sigmoid(logit) → P(SE | drug_A, drug_B)

Ablation conditions A–E
------------------------
  A  ChemBERTa + ESM-2 (main)
  B  ESM-2 only  (drug branch receives zeros)
  C  ChemBERTa only  (protein branch receives zeros)
  D  Random baseline  (both branches receive fixed Xavier noise)
  E  Full multimodal  (ChemBERTa+mono drug branch, ESM-2+PPI protein branch)

Prerequisites
-------------
  Run scripts/precompute_embeddings_ablation_mlp.py first to save raw feature
  tensors.  All raw tensors are drug-indexed (rows = sorted drug_to_idx).

Usage
-----
  python ablation/ablation_track1_mlp.py \\
      --drug_raw_chemberta  data/cache/ablation_mlp/drug_raw_chemberta.pt \\
      --drug_raw_mono       data/cache/ablation_mlp/drug_raw_mono.pt \\
      --drug_raw_esm2       data/cache/ablation_mlp/drug_raw_esm2_via_targets.pt \\
      --drug_raw_ppi        data/cache/ablation_mlp/drug_raw_ppi_via_targets.pt \\
      --drug_to_idx         data/cache/ablation_mlp/drug_to_idx.json \\
      --combo               data/raw/bio-decagon-combo.csv \\
      --output              results/ablation_mlp \\
      --device              cuda

  # Conditions subset / pair operator
  python ablation/ablation_track1_mlp.py ... --conditions A,B,C --pair_repr concat

See ablation/README_MLP_ablation.md for the full design rationale.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import tempfile
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Model ─────────────────────────────────────────────────────────────────────

class BranchMLP(nn.Module):
    """
    Single trainable projection head: in_dim → hidden_dim → out_dim.

    Architecture mirrors _ablation_mlp_project in src/model.py (same depth,
    same activation) but with trainable weights updated by backpropagation.
    """

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


class AblationNet(nn.Module):
    """
    Full trainable model: two branch MLPs + optional extra heads for condition E
    + symmetric pair operator + classifier MLP.

    Args
    ----
    drug_in_dim : Input dimension of ChemBERTa drug features (default 768).
    prot_in_dim : Input dimension of ESM-2 drug-level features (default 640).
    embed_dim   : Target projection dimension (default 256); each branch projects
                  to embed_dim, so fused entity = 2*embed_dim.
    pair_repr   : "sym" (product+abs-diff), "concat" (e_A||e_B), or "sum".
    dropout     : Dropout in BranchMLP and ClassifierMLP.
    mono_in_dim : If given, adds a mono branch MLP for condition E.
    has_ppi     : If True, adds a PPI branch MLP for condition E.
    """

    def __init__(
        self,
        drug_in_dim: int,
        prot_in_dim: int,
        embed_dim:   int   = 256,
        pair_repr:   str   = "sym",
        dropout:     float = 0.1,
        mono_in_dim: int | None = None,
        has_ppi:     bool  = False,
    ) -> None:
        super().__init__()
        self.embed_dim  = embed_dim
        self.pair_repr  = pair_repr

        self.drug_mlp = BranchMLP(drug_in_dim, embed_dim, dropout)
        self.prot_mlp = BranchMLP(prot_in_dim, embed_dim, dropout)

        # Condition E extras
        self.mono_mlp = BranchMLP(mono_in_dim, embed_dim, dropout) if mono_in_dim else None
        self.ppi_mlp  = BranchMLP(prot_in_dim, embed_dim, dropout) if has_ppi   else None

        fused_dim = 2 * embed_dim   # drug_branch + prot_branch per entity
        if pair_repr in ("sym", "concat"):
            pair_dim = 2 * fused_dim    # 1024 for embed_dim=256
        else:
            pair_dim = fused_dim        # 512 for sum

        # Classifier: pair_dim → pair_dim//4 → pair_dim//8 → 1
        # For embed_dim=256, sym: 1024 → 256 → 128 → 1 (matches README spec)
        self.classifier = nn.Sequential(
            nn.Linear(pair_dim,       pair_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim // 4, pair_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pair_dim // 8, 1),
        )

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
        """Returns raw logits (B,).  Apply sigmoid for probabilities."""
        e_a = self._entity_embed(drug_a, prot_a, mono_a, ppi_a)
        e_b = self._entity_embed(drug_b, prot_b, mono_b, ppi_b)

        if self.pair_repr == "sym":
            pair_feat = torch.cat([e_a * e_b, (e_a - e_b).abs()], dim=1)
        elif self.pair_repr == "concat":
            pair_feat = torch.cat([e_a, e_b], dim=1)
        else:   # sum
            pair_feat = e_a + e_b

        return self.classifier(pair_feat).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PSEPairDataset(Dataset):
    """
    Drug-pair PSE binary classification dataset.

    Stores pre-built index arrays (s1_idx, s2_idx, labels, se_ids, degrees) and
    looks up condition-specific raw feature tensors at __getitem__ time.

    Args
    ----
    s1_idx, s2_idx : (N,) int64 arrays of row indices into the feature tensors.
    labels         : (N,) float32 binary labels.
    se_ids         : (N,) object array of SE CUI strings.
    degrees        : (N,) float32 degree products (bias metric).
    drug_feat      : (n_drugs, drug_dim) float32 tensor.
    prot_feat      : (n_drugs, prot_dim) float32 tensor.
    mono_feat      : (n_drugs, mono_dim) float32 tensor or None (conditions A-D).
    ppi_feat       : (n_drugs, prot_dim) float32 tensor or None (conditions A-D).
    """

    def __init__(
        self,
        s1_idx:    np.ndarray,
        s2_idx:    np.ndarray,
        labels:    np.ndarray,
        se_ids:    np.ndarray,
        degrees:   np.ndarray,
        drug_feat: torch.Tensor,
        prot_feat: torch.Tensor,
        mono_feat: torch.Tensor | None = None,
        ppi_feat:  torch.Tensor | None = None,
    ) -> None:
        self.s1      = s1_idx.astype(np.int64)
        self.s2      = s2_idx.astype(np.int64)
        self.labels  = labels.astype(np.float32)
        self.se_ids  = se_ids
        self.degrees = degrees.astype(np.float32)
        # Keep tensors on CPU; DataLoader workers copy slices to device
        self.drug_feat = drug_feat.float().cpu()
        self.prot_feat = prot_feat.float().cpu()
        self.mono_feat = mono_feat.float().cpu() if mono_feat is not None else None
        self.ppi_feat  = ppi_feat.float().cpu()  if ppi_feat  is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        a, b = int(self.s1[i]), int(self.s2[i])
        item = {
            "drug_a":  self.drug_feat[a],
            "prot_a":  self.prot_feat[a],
            "drug_b":  self.drug_feat[b],
            "prot_b":  self.prot_feat[b],
            "label":   torch.tensor(self.labels[i],  dtype=torch.float32),
            "degree":  torch.tensor(self.degrees[i], dtype=torch.float32),
            "se_id":   self.se_ids[i],
        }
        if self.mono_feat is not None:
            item["mono_a"] = self.mono_feat[a]
            item["mono_b"] = self.mono_feat[b]
        if self.ppi_feat is not None:
            item["ppi_a"] = self.ppi_feat[a]
            item["ppi_b"] = self.ppi_feat[b]
        return item


def _collate(batch: list[dict]) -> dict:
    """Stack tensors; keep se_id as a plain list of strings."""
    keys = batch[0].keys()
    out  = {}
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack([item[k] for item in batch])
        else:
            out[k] = [item[k] for item in batch]
    return out


# ── Pair generation ───────────────────────────────────────────────────────────

def build_pairs(
    df:          pd.DataFrame,
    drug_to_idx: dict[str, int],
    neg_ratio:   int,
    seed:        int,
    se_sample:   list[str] | None = None,
    max_pos_edges: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate (s1_idx, s2_idx, labels, se_ids, degrees) arrays.

    Mirrors the negative-sampling logic in ablation_track1_ml.py: corrupt one
    drug per positive (50 % left, 50 % right) with a random drug from the full
    vocabulary.

    Returns numpy arrays: s1_idx, s2_idx, labels, se_ids, degrees.
    """
    rng          = np.random.RandomState(seed)
    all_drugs    = list(drug_to_idx.keys())
    n_drugs      = len(all_drugs)

    # Degree product (bias metric) computed over the FULL combo before SE filtering
    stacked_all  = pd.concat([df["STITCH 1"], df["STITCH 2"]], ignore_index=True)
    degree       = stacked_all.value_counts().to_dict()
    deg_prod     = lambda x, y: float(degree.get(x, 1) * degree.get(y, 1))

    if se_sample is not None:
        df = df[df["Polypharmacy Side Effect"].isin(se_sample)]

    mask = df["STITCH 1"].isin(drug_to_idx) & df["STITCH 2"].isin(drug_to_idx)
    df   = df.loc[mask]
    if len(df) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, np.float32), np.empty(0, object), np.empty(0, np.float32)

    if max_pos_edges is not None and len(df) > max_pos_edges:
        df = df.sample(n=max_pos_edges, random_state=seed)

    n_pos     = len(df)
    n_samples = n_pos * (1 + neg_ratio)

    s1_a     = df["STITCH 1"].to_numpy()
    s2_a     = df["STITCH 2"].to_numpy()
    se_col   = df["Polypharmacy Side Effect"].to_numpy()

    s1_idx   = np.empty(n_samples, dtype=np.int64)
    s2_idx   = np.empty(n_samples, dtype=np.int64)
    labels   = np.empty(n_samples, dtype=np.float32)
    se_ids   = np.empty(n_samples, dtype=object)
    degrees  = np.empty(n_samples, dtype=np.float32)

    row = 0
    for i in range(n_pos):
        a, b, se = s1_a[i], s2_a[i], se_col[i]
        ia, ib   = drug_to_idx[a], drug_to_idx[b]

        s1_idx[row]  = ia
        s2_idx[row]  = ib
        labels[row]  = 1.0
        se_ids[row]  = se
        degrees[row] = deg_prod(a, b)
        row += 1

        for _ in range(neg_ratio):
            neg = all_drugs[rng.randint(n_drugs)]
            ic  = drug_to_idx[neg]
            if rng.rand() < 0.5:
                s1_idx[row], s2_idx[row] = ic, ib
                degrees[row] = deg_prod(neg, b)
            else:
                s1_idx[row], s2_idx[row] = ia, ic
                degrees[row] = deg_prod(a, neg)
            labels[row]  = 0.0
            se_ids[row]  = se
            row += 1

    assert row == n_samples
    return s1_idx, s2_idx, labels, se_ids, degrees


# ── Condition tensors ─────────────────────────────────────────────────────────

def make_condition_tensors(
    condition:    str,
    n_drugs:      int,
    drug_raw:     torch.Tensor,         # (n_drugs, 768)
    prot_raw:     torch.Tensor,         # (n_drugs, 640)
    mono_raw:     torch.Tensor | None,  # (n_drugs, mono_dim) — None unless E
    ppi_raw:      torch.Tensor | None,  # (n_drugs, 640) — None unless E
    xavier_seed:  int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Return (drug_feat, prot_feat, mono_feat, ppi_feat) for the given condition.

    Conditions B and C zero-out the inactive branch.  Condition D replaces
    both branches with fixed Xavier-uniform noise (same shape, different seed).
    Condition E passes chemberta+mono for drug and esm2+ppi for protein.
    """
    drug_dim = drug_raw.shape[1]
    prot_dim = prot_raw.shape[1]

    if condition == "A":
        return drug_raw, prot_raw, None, None

    elif condition == "B":
        # Protein signal only — drug branch receives zeros
        return torch.zeros(n_drugs, drug_dim), prot_raw, None, None

    elif condition == "C":
        # Drug signal only — protein branch receives zeros
        return drug_raw, torch.zeros(n_drugs, prot_dim), None, None

    elif condition == "D":
        # Random baseline — fixed per-drug Xavier noise, no pretrained signal
        g_drug = torch.Generator()
        g_drug.manual_seed(xavier_seed)
        g_prot = torch.Generator()
        g_prot.manual_seed(xavier_seed + 1)
        bound_drug = math.sqrt(3.0 / drug_dim)
        bound_prot = math.sqrt(3.0 / prot_dim)
        rand_drug  = torch.empty(n_drugs, drug_dim).uniform_(-bound_drug, bound_drug, generator=g_drug)
        rand_prot  = torch.empty(n_drugs, prot_dim).uniform_(-bound_prot, bound_prot, generator=g_prot)
        return rand_drug, rand_prot, None, None

    elif condition == "E":
        if mono_raw is None or ppi_raw is None:
            raise ValueError(
                "Condition E requires --drug_raw_mono and --drug_raw_ppi. "
                "Ensure both tensors were computed by precompute_embeddings_ablation_mlp.py."
            )
        return drug_raw, prot_raw, mono_raw, ppi_raw

    else:
        raise ValueError(f"Unknown condition: {condition!r}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def assign_tiers(se_counts: pd.Series, n_tiers: int = 5) -> dict:
    """Assign frequency tier label to each SE CUI (mirrors ablation_track1_ml.py)."""
    sorted_ses = se_counts.sort_values()
    tier_size  = len(sorted_ses) // n_tiers
    tier_map   = {}
    labels     = [f"T{n_tiers}_rare", f"T{n_tiers-1}", "T3", "T2", "T1_frequent"]
    for i, (se, _) in enumerate(sorted_ses.items()):
        tier_idx      = min(i // tier_size, n_tiers - 1)
        tier_map[se]  = labels[tier_idx]
    return tier_map


def _degree_score_corrs(degrees: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Pearson and Spearman corr(deg_product, score) on test positives for one SE."""
    if len(degrees) < 3:
        return float("nan"), float("nan")
    if np.std(degrees) == 0 or np.std(scores) == 0:
        return float("nan"), float("nan")
    pearson  = float(np.corrcoef(degrees, scores)[0, 1])
    spearman = float(spearmanr(degrees, scores).statistic)
    return pearson, spearman


def _confusion_counts(
    y_true:    np.ndarray,
    y_score:   np.ndarray,
    threshold: float = 0.5,
) -> dict[str, int]:
    """Binary confusion matrix at a fixed score threshold (default 0.5)."""
    yt = y_true.astype(bool)
    yp = y_score >= threshold
    return {
        "tp": int(( yt &  yp).sum()),
        "fp": int((~yt &  yp).sum()),
        "tn": int((~yt & ~yp).sum()),
        "fn": int(( yt & ~yp).sum()),
    }


def evaluate(
    y_true:      np.ndarray,
    y_score:     np.ndarray,
    se_labels:   np.ndarray,
    degrees:     np.ndarray,
    tier_map:    dict,
    condition:   str,
    threshold:   float = 0.5,
) -> pd.DataFrame:
    """
    Per-SE AUROC, AUPRC, AP@50, confusion counts (TP/FP/TN/FN), and degree-score
    correlation (Pearson + Spearman) computed separately on positive and negative
    test pairs.

    Positive-pair correlation (r_pos): high r_pos can reflect real biology —
    hub drugs genuinely co-occur in more SEs, so the model may legitimately
    score them higher.

    Negative-pair correlation (r_neg): negatives are randomly corrupted pairs
    that should carry no true biological signal.  A high r_neg therefore
    indicates genuine degree bias — the model inflates scores for hub pairs
    regardless of biological relevance.

    Confusion counts use ``y_score >= threshold`` (default 0.5).
    """
    results    = []
    unique_ses = np.unique(se_labels)

    for se in unique_ses:
        mask = se_labels == se
        yt   = y_true[mask]
        ys   = y_score[mask]

        if yt.sum() == 0 or yt.sum() == mask.sum():
            continue

        auroc   = roc_auc_score(yt, ys)
        auprc   = average_precision_score(yt, ys)
        top50   = np.argsort(ys)[::-1][:50]
        ap50    = yt[top50].sum() / 50.0

        pos_mask = yt == 1
        neg_mask = yt == 0

        deg_pearson_pos, deg_spearman_pos = _degree_score_corrs(
            degrees[mask][pos_mask], ys[pos_mask]
        )
        deg_pearson_neg, deg_spearman_neg = _degree_score_corrs(
            degrees[mask][neg_mask], ys[neg_mask]
        )

        cm = _confusion_counts(yt, ys, threshold=threshold)

        results.append({
            "condition":                      condition,
            "se":                             se,
            "tier":                           tier_map.get(se, "unknown"),
            "auroc":                          auroc,
            "auprc":                          auprc,
            "ap50":                           ap50,
            "n_pos":                          int(yt.sum()),
            "n_neg":                          int((yt == 0).sum()),
            "n_total":                        int(mask.sum()),
            "threshold":                      threshold,
            "tp":                             cm["tp"],
            "fp":                             cm["fp"],
            "tn":                             cm["tn"],
            "fn":                             cm["fn"],
            # Positive-pair degree correlation (may reflect real biology)
            "degree_score_corr_pearson":      deg_pearson_pos,
            "degree_score_corr_spearman":     deg_spearman_pos,
            # Negative-pair degree correlation (pure bias signal)
            "degree_score_corr_pearson_neg":  deg_pearson_neg,
            "degree_score_corr_spearman_neg": deg_spearman_neg,
        })

    return pd.DataFrame(results)


# ── Training helpers ──────────────────────────────────────────────────────────

def _batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _forward(model: AblationNet, batch: dict) -> torch.Tensor:
    return model(
        batch["drug_a"], batch["prot_a"],
        batch["drug_b"], batch["prot_b"],
        mono_a=batch.get("mono_a"), mono_b=batch.get("mono_b"),
        ppi_a =batch.get("ppi_a"),  ppi_b =batch.get("ppi_b"),
    )


def train_one_epoch(
    model:      AblationNet,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    scheduler:  OneCycleLR,
    device:     torch.device,
    pos_weight: torch.Tensor | None = None,
) -> float:
    """One pass over the training DataLoader.  Returns mean BCE loss."""
    model.train()
    total_loss  = 0.0
    n_batches   = 0

    for batch in loader:
        batch = _batch_to_device(batch, device)
        logits = _forward(model, batch)
        loss   = F.binary_cross_entropy_with_logits(
            logits, batch["label"],
            pos_weight=pos_weight,
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
    model:  AblationNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader.
    Returns (y_true, y_score, se_ids, degrees) as numpy arrays.
    """
    model.eval()
    all_labels  = []
    all_scores  = []
    all_se_ids  = []
    all_degrees = []

    for batch in loader:
        batch  = _batch_to_device(batch, device)
        logits = _forward(model, batch)
        probs  = torch.sigmoid(logits).cpu().numpy()

        all_labels.extend(batch["label"].cpu().numpy().tolist())
        all_scores.extend(probs.tolist())
        all_se_ids.extend(batch["se_id"])
        all_degrees.extend(batch["degree"].cpu().numpy().tolist())

    return (
        np.array(all_labels,  dtype=np.float32),
        np.array(all_scores,  dtype=np.float32),
        np.array(all_se_ids,  dtype=object),
        np.array(all_degrees, dtype=np.float32),
    )


# ── Sampler helpers ───────────────────────────────────────────────────────────

def _se_balanced_weights(se_ids: np.ndarray) -> np.ndarray:
    """
    Per-sample weight = 1 / n_samples_for_this_SE.

    Ensures each SE contributes equally to gradient updates regardless of how
    many pairs it has (T1 SEs have 28K pairs vs T5 SEs with ~500 pairs).
    """
    counts  = defaultdict(int)
    for se in se_ids:
        counts[se] += 1
    weights = np.array([1.0 / counts[se] for se in se_ids], dtype=np.float64)
    weights /= weights.mean()   # scale so mean weight = 1
    return weights.astype(np.float32)


def _append_results(path: Path, df: pd.DataFrame, first: bool) -> bool:
    df.to_csv(path, mode="w" if first else "a", header=first, index=False)
    return False


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track 1 MLP ablation — trainable projection heads"
    )

    # Input tensors
    p.add_argument("--drug_raw_chemberta", required=True,
                   help="Path to drug_raw_chemberta.pt  (n_drugs, 768)")
    p.add_argument("--drug_raw_mono",      default=None,
                   help="Path to drug_raw_mono.pt  (n_drugs, mono_dim).  Required for condition E.")
    p.add_argument("--drug_raw_esm2",      required=True,
                   help="Path to drug_raw_esm2_via_targets.pt  (n_drugs, esm_dim)")
    p.add_argument("--drug_raw_ppi",       default=None,
                   help="Path to drug_raw_ppi_via_targets.pt  (n_drugs, esm_dim).  Required for E.")
    p.add_argument("--drug_to_idx",        required=True,
                   help="Path to drug_to_idx.json  (drug_str → row_index in above tensors)")

    # Data
    p.add_argument("--combo",     default="data/raw/bio-decagon-combo.csv")
    p.add_argument("--min_edges", type=int, default=500,
                   help="Minimum positive pairs per SE (Lloyd threshold)")
    p.add_argument("--neg_ratio", type=int, default=1,
                   help="Negative pairs per positive")
    p.add_argument("--seed",      type=int, default=42)

    # Conditions
    p.add_argument("--conditions", default="A,B,C,D,E",
                   help="Comma-separated conditions to run, e.g. A,B,C")

    # SE sampling (optional — useful for debugging)
    p.add_argument("--n_se_sample",  type=int, default=None,
                   help="Run on only the first N SEs (by decreasing edge count)")
    p.add_argument("--se_offset",    type=int, default=0,
                   help="Start index for --n_se_sample slicing")
    p.add_argument("--only_se_ids",  default=None,
                   help="Comma or space separated SE CUI codes to restrict evaluation")

    # Model
    p.add_argument("--embed_dim",   type=int,   default=256,
                   help="Projection dimension per branch (drug/prot each → embed_dim)")
    p.add_argument("--pair_repr",   choices=["sym", "concat", "sum"], default="sym")
    p.add_argument("--dropout",     type=float, default=0.1)

    # Training
    p.add_argument("--max_epochs",  type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=1024)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=5,
                   help="Early stopping patience (epochs with no val AUROC improvement)")
    p.add_argument("--val_frac",    type=float, default=0.1,
                   help="Fraction of training set to use for early stopping validation")
    p.add_argument("--test_frac",   type=float, default=0.2,
                   help="Fraction of all pairs used as held-out test set")
    p.add_argument("--max_pos_edges", type=int, default=None,
                   help="Cap positive pairs per run (for fast debugging)")
    p.add_argument("--se_weighted_sampler", action="store_true", default=True,
                   help="Use SE-balanced WeightedRandomSampler for training mini-batches")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader worker processes (0 = main process)")

    # Output
    p.add_argument("--output", default="results/ablation_mlp",
                   help="Directory for CSV results and model checkpoints")
    p.add_argument("--save_checkpoints", action="store_true", default=False,
                   help="Keep best-epoch .pt checkpoints after training finishes")

    # Optional wandb logging
    p.add_argument("--wandb",         action="store_true", default=False)
    p.add_argument("--wandb_project", default="pse_ablation_mlp")
    p.add_argument("--wandb_entity",  default=None)
    p.add_argument("--wandb_run_name",default=None)
    p.add_argument("--wandb_offline", action="store_true")
    p.add_argument("--wandb_tags",    default=None)

    p.add_argument("--device", default="cpu")

    return p.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_tensor(path: str | None) -> torch.Tensor | None:
    if path is None:
        return None
    t = torch.load(path, map_location="cpu")
    return t.float()


def _load_combo(combo_path: str, min_edges: int) -> pd.DataFrame:
    df        = pd.read_csv(combo_path, encoding="latin-1")
    se_counts = df.groupby("Polypharmacy Side Effect").size()
    valid     = se_counts[se_counts >= min_edges].index
    df        = df[df["Polypharmacy Side Effect"].isin(valid)]
    print(f"Combo after min_edges={min_edges} filter: {len(df):,} rows, "
          f"{len(valid)} SEs, {pd.concat([df['STITCH 1'], df['STITCH 2']]).nunique()} drugs")
    return df


def _verify_drug_alignment(
    loaded_idx: dict[str, int],
    combo_idx:  dict[str, int],
) -> None:
    """Abort if drug_to_idx.json does not match the combo-derived index."""
    if loaded_idx != combo_idx:
        extra_loaded = set(loaded_idx) - set(combo_idx)
        extra_combo  = set(combo_idx)  - set(loaded_idx)
        raise ValueError(
            "drug_to_idx.json does not match the filtered combo.\n"
            f"  Drugs only in JSON: {len(extra_loaded)} (e.g. {sorted(extra_loaded)[:5]})\n"
            f"  Drugs only in combo: {len(extra_combo)} (e.g. {sorted(extra_combo)[:5]})\n"
            "Re-run precompute_embeddings_ablation_mlp.py with the same --combo and "
            "--min_edges arguments, then retry."
        )


def _parse_se_sample(args: argparse.Namespace, ordered_se: list[str]) -> list[str] | None:
    if args.only_se_ids:
        requested = {s.strip() for s in args.only_se_ids.replace(",", " ").split() if s.strip()}
        se_sample = [se for se in ordered_se if se in requested]
        if not se_sample:
            raise RuntimeError("None of --only_se_ids CUIs found in filtered combo.")
        print(f"Running --only_se_ids: {len(se_sample)} SEs")
        return se_sample
    if args.n_se_sample is not None:
        start = max(0, args.se_offset)
        end   = min(start + args.n_se_sample, len(ordered_se))
        se_sample = ordered_se[start:end]
        if not se_sample:
            raise RuntimeError("Empty SE slice; check --n_se_sample / --se_offset.")
        print(f"SE slice [{start}:{end}] of {len(ordered_se)}: {len(se_sample)} SEs")
        return se_sample
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir   = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    device = torch.device(args.device)

    print("=" * 60)
    print("Track 1 MLP Ablation — trainable projection heads")
    print("=" * 60)

    # ── Load tensors ──────────────────────────────────────────────────────────
    print("\nLoading raw feature tensors ...")
    drug_raw  = _load_tensor(args.drug_raw_chemberta)
    prot_raw  = _load_tensor(args.drug_raw_esm2)
    mono_raw  = _load_tensor(args.drug_raw_mono)
    ppi_raw   = _load_tensor(args.drug_raw_ppi)

    print(f"  ChemBERTa: {tuple(drug_raw.shape)}")
    print(f"  ESM-2:     {tuple(prot_raw.shape)}")
    if mono_raw is not None:
        print(f"  Mono:      {tuple(mono_raw.shape)}")
    if ppi_raw is not None:
        print(f"  PPI:       {tuple(ppi_raw.shape)}")

    with open(args.drug_to_idx) as f:
        drug_to_idx: dict[str, int] = json.load(f)

    # ── Load + filter combo ───────────────────────────────────────────────────
    df          = _load_combo(args.combo, args.min_edges)
    all_drugs_sorted = sorted(pd.concat([df["STITCH 1"], df["STITCH 2"]]).unique())
    combo_idx   = {d: i for i, d in enumerate(all_drugs_sorted)}
    _verify_drug_alignment(drug_to_idx, combo_idx)
    n_drugs     = len(drug_to_idx)

    # SE frequency tiers and sampling
    se_counts   = df.groupby("Polypharmacy Side Effect").size()
    tier_map    = assign_tiers(se_counts)
    ordered_se  = se_counts.sort_values(ascending=False).index.tolist()
    se_sample   = _parse_se_sample(args, ordered_se)

    # Determine conditions to run
    requested_conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    conditions = []
    for c in requested_conditions:
        if c == "E" and (mono_raw is None or ppi_raw is None):
            print(f"Skipping condition E (--drug_raw_mono or --drug_raw_ppi not provided)")
            continue
        conditions.append(c)
    print(f"\nConditions to run: {conditions}")

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb_run  = None
    if args.wandb:
        try:
            import wandb
        except ImportError as e:
            raise ImportError("pip install wandb") from e
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=tags or None,
            mode="offline" if args.wandb_offline else "online",
            config={
                "embed_dim":       args.embed_dim,
                "pair_repr":       args.pair_repr,
                "dropout":         args.dropout,
                "max_epochs":      args.max_epochs,
                "batch_size":      args.batch_size,
                "lr":              args.lr,
                "weight_decay":    args.weight_decay,
                "patience":        args.patience,
                "neg_ratio":       args.neg_ratio,
                "test_frac":       args.test_frac,
                "val_frac":        args.val_frac,
                "min_edges":       args.min_edges,
                "conditions":      conditions,
                "drug_dim":        drug_raw.shape[1],
                "prot_dim":        prot_raw.shape[1],
                "mono_dim":        mono_raw.shape[1] if mono_raw is not None else None,
                "n_drugs":         n_drugs,
            },
        )

    results_csv  = output_dir / "ablation_mlp_results_per_se.csv"
    curves_csv   = output_dir / "ablation_mlp_training_curves.csv"
    first_write  = True
    first_curve  = True
    wandb_step   = 0

    # ── Per-condition training loop ───────────────────────────────────────────
    for condition in conditions:
        print(f"\n{'─'*40}")
        print(f"Condition {condition}")

        # Build condition-specific input tensors
        drug_feat, prot_feat, mono_feat, ppi_feat = make_condition_tensors(
            condition, n_drugs,
            drug_raw, prot_raw, mono_raw, ppi_raw,
            xavier_seed=args.seed,
        )

        # Generate all pairs (fixed negatives for reproducibility)
        print("  Building pairs ...")
        s1, s2, labels, se_ids, degrees = build_pairs(
            df, drug_to_idx,
            neg_ratio    = args.neg_ratio,
            seed         = args.seed,
            se_sample    = se_sample,
            max_pos_edges= args.max_pos_edges,
        )
        print(f"  Pairs: {len(labels):,}  ({int(labels.sum()):,} pos, "
              f"{int((1-labels).sum()):,} neg)")

        # 80 / 20 stratified split — test set is identical to ablation_track1_ml.py
        idx        = np.arange(len(labels), dtype=np.int64)
        train_idx, test_idx = train_test_split(
            idx, test_size=args.test_frac, stratify=labels.astype(int), random_state=args.seed
        )

        # 10 % of train → validation for early stopping
        subtrain_idx, val_idx = train_test_split(
            train_idx,
            test_size   = args.val_frac,
            stratify    = labels[train_idx].astype(int),
            random_state= args.seed,
        )

        def _make_ds(idx_sel: np.ndarray) -> PSEPairDataset:
            return PSEPairDataset(
                s1[idx_sel], s2[idx_sel],
                labels[idx_sel], se_ids[idx_sel], degrees[idx_sel],
                drug_feat, prot_feat, mono_feat, ppi_feat,
            )

        train_ds = _make_ds(subtrain_idx)
        val_ds   = _make_ds(val_idx)
        test_ds  = _make_ds(test_idx)

        # SE-balanced sampler for training
        if args.se_weighted_sampler:
            w        = _se_balanced_weights(se_ids[subtrain_idx])
            sampler  = WeightedRandomSampler(
                weights     = torch.from_numpy(w).double(),
                num_samples = len(subtrain_idx),
                replacement = True,
            )
            train_loader = DataLoader(
                train_ds,
                batch_size  = args.batch_size,
                sampler     = sampler,
                num_workers = args.num_workers,
                collate_fn  = _collate,
                pin_memory  = (device.type == "cuda"),
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, collate_fn=_collate,
                pin_memory=(device.type == "cuda"),
            )

        val_loader  = DataLoader(
            val_ds,  batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, collate_fn=_collate,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, collate_fn=_collate,
        )

        # Instantiate model for this condition
        model = AblationNet(
            drug_in_dim = drug_raw.shape[1],
            prot_in_dim = prot_raw.shape[1],
            embed_dim   = args.embed_dim,
            pair_repr   = args.pair_repr,
            dropout     = args.dropout,
            mono_in_dim = mono_raw.shape[1] if (mono_feat is not None and mono_raw is not None) else None,
            has_ppi     = (ppi_feat is not None),
        ).to(device)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model: {trainable_params:,} trainable parameters")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        total_steps = args.max_epochs * len(train_loader)
        scheduler   = OneCycleLR(
            optimizer,
            max_lr      = args.lr,
            total_steps = total_steps,
            pct_start   = 0.05,     # 5 % warmup
        )

        # Positive-class weight to counteract 1:neg_ratio imbalance
        pos_weight = torch.tensor([float(args.neg_ratio)], device=device)

        # Training loop
        best_val_auroc   = -1.0
        best_epoch       = 0
        patience_counter = 0
        ckpt_path        = ckpt_dir / f"best_{condition}.pt"
        epoch_rows: list[dict] = []

        print(f"  Training (max {args.max_epochs} epochs, patience {args.patience}) ...")
        for epoch in range(args.max_epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, pos_weight
            )
            y_true_v, y_score_v, _, _ = eval_predictions(model, val_loader, device)

            try:
                val_auroc = roc_auc_score(y_true_v, y_score_v)
            except ValueError:
                val_auroc = 0.5

            is_best = val_auroc > best_val_auroc
            print(f"  Epoch {epoch+1:3d}  train_loss={train_loss:.4f}  "
                  f"val_auroc={val_auroc:.4f}{'  *' if is_best else ''}")

            epoch_rows.append({
                "condition":  condition,
                "seed":       args.seed,
                "epoch":      epoch + 1,
                "train_loss": train_loss,
                "val_auroc":  val_auroc,
                "is_best":    is_best,
            })

            if is_best:
                best_val_auroc   = val_auroc
                best_epoch       = epoch + 1
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1} (patience={args.patience})")
                    break

        # Flush per-epoch curve rows to CSV
        curves_df  = pd.DataFrame(epoch_rows)
        first_curve = _append_results(curves_csv, curves_df, first_curve)

        # Load best checkpoint and evaluate on test set
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        y_true_t, y_score_t, se_labels_t, degrees_t = eval_predictions(
            model, test_loader, device
        )

        overall_auroc = roc_auc_score(y_true_t, y_score_t)
        overall_auprc = average_precision_score(y_true_t, y_score_t)
        print(f"  Test — AUROC: {overall_auroc:.4f}  AUPRC: {overall_auprc:.4f}  "
              f"(best val AUROC: {best_val_auroc:.4f})")

        results_df = evaluate(
            y_true_t, y_score_t, se_labels_t, degrees_t, tier_map, condition
        )
        results_df["best_val_auroc"] = best_val_auroc
        results_df["best_epoch"]     = best_epoch
        first_write = _append_results(results_csv, results_df, first_write)

        if wandb_run is not None:
            import wandb
            log_payload = {
                "overall/auroc":            overall_auroc,
                "overall/auprc":            overall_auprc,
                "best_val/auroc":           best_val_auroc,
                "per_se/median_auroc":      float(results_df["auroc"].median()),
                "per_se/median_auprc":      float(results_df["auprc"].median()),
                "bias/median_deg_corr_pearson_pos":  float(
                    results_df["degree_score_corr_pearson"].median()
                ),
                "bias/median_deg_corr_spearman_pos": float(
                    results_df["degree_score_corr_spearman"].median()
                ),
                "bias/median_deg_corr_pearson_neg":  float(
                    results_df["degree_score_corr_pearson_neg"].median()
                ),
                "bias/median_deg_corr_spearman_neg": float(
                    results_df["degree_score_corr_spearman_neg"].median()
                ),
                "meta/condition":           condition,
            }
            for tier, val in results_df.groupby("tier")["auprc"].median().items():
                log_payload[f"tier/median_auprc/{tier}"] = float(val)
            wandb.log(log_payload, step=wandb_step)
            wandb_step += 1

        # Optionally clean up checkpoint to save disk space
        if not args.save_checkpoints:
            ckpt_path.unlink(missing_ok=True)

        del model, optimizer, scheduler
        del train_ds, val_ds, test_ds
        del train_loader, val_loader, test_loader
        del s1, s2, labels, se_ids, degrees
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Aggregate results ─────────────────────────────────────────────────────
    if not results_csv.exists():
        print("\nNo results written (all conditions skipped or failed).")
        return

    all_df  = pd.read_csv(results_csv)
    summary = (
        all_df.groupby(["condition", "tier"])
        .agg(
            median_auroc                    = ("auroc",                          "median"),
            median_auprc                    = ("auprc",                          "median"),
            median_ap50                     = ("ap50",                           "median"),
            median_deg_corr_pearson_pos     = ("degree_score_corr_pearson",      "median"),
            median_deg_corr_spearman_pos    = ("degree_score_corr_spearman",     "median"),
            median_deg_corr_pearson_neg     = ("degree_score_corr_pearson_neg",  "median"),
            median_deg_corr_spearman_neg    = ("degree_score_corr_spearman_neg", "median"),
            mean_best_val_auroc             = ("best_val_auroc",                 "mean"),
            mean_best_epoch                 = ("best_epoch",                     "mean"),
            n_ses                           = ("se",                             "count"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "ablation_mlp_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("SUMMARY — median AUPRC by condition and tier")
    print("=" * 60)
    pivot = summary.pivot_table(
        index="tier", columns="condition", values="median_auprc"
    ).round(4)
    print(pivot.to_string())

    print("\nDegree-score correlation — median across SEs (pos = positive pairs, neg = negative pairs):")
    bias = (
        all_df.groupby("condition")[[
            "degree_score_corr_pearson",
            "degree_score_corr_spearman",
            "degree_score_corr_pearson_neg",
            "degree_score_corr_spearman_neg",
        ]]
        .median().round(4).reset_index()
    )
    print(bias.to_string(index=False))

    print(f"\nResults saved to: {output_dir}")

    if wandb_run is not None:
        import wandb
        wandb.log({
            "tables/ablation_summary":   wandb.Table(dataframe=summary),
            "tables/bias_by_condition":  wandb.Table(dataframe=bias),
        })
        for _, row in summary.iterrows():
            key = f"final/median_auprc/{row['condition']}/{row['tier']}"
            wandb.summary[key] = float(row["median_auprc"])
        wandb.finish()


if __name__ == "__main__":
    main()
