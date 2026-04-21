"""
Stratified evaluation for polypharmacy side effect prediction.

Standard evaluation (AUROC, AUPRC, AP@50) averages over all test drug pairs.
This module additionally stratifies results by the number of known protein
targets per drug, revealing how well a model performs for drugs with sparse
vs. dense protein-interaction coverage.

Stratification bins:
    "none"   — 0 known targets in the Decagon PPI graph
    "low"    — 1-2 known targets
    "medium" — 3-9 known targets
    "high"   — 10+ known targets

This directly tests the hypothesis that our semantically grounded
initialization (ChemBERTa + TF-IDF/CUR) makes predictions more robust to
incomplete PPI coverage, compared to the Non-naive baseline.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BIN_THRESHOLDS = {
    "none":   (0, 0),
    "low":    (1, 2),
    "medium": (3, 9),
    "high":   (10, int(1e9)),
}


def assign_target_bins(
    drug_targets: Dict[str, Set[str]],
    protein_to_id: Dict[str, int],
) -> Dict[str, str]:
    """
    Assign each drug to a DPI (drug-protein interaction) coverage bin.

    Bins measure how many of the drug's known target proteins appear in
    bio-decagon-targets.csv AND are present as graph nodes — i.e., how
    much DPI-derived signal is available to the model for this drug.

    Drugs with 0 DPI entries ('none' bin) can only be predicted via
    molecular structure (ChemBERTa) and monopharmacy side effects
    (TF-IDF/CUR) — entirely independent of the target file.

    Args:
        drug_targets: drug identifier -> set of target protein identifiers,
            from bio-decagon-targets.csv.
        protein_to_id: protein identifiers present in the graph.

    Returns:
        drug identifier -> bin label ('none', 'low', 'medium', 'high').
    """
    bin_map: Dict[str, str] = {}
    for drug, targets in drug_targets.items():
        n_covered = sum(1 for t in targets if t in protein_to_id)
        for bin_name, (lo, hi) in BIN_THRESHOLDS.items():
            if lo <= n_covered <= hi:
                bin_map[drug] = bin_name
                break
        else:
            bin_map[drug] = "high"
    return bin_map


def compute_auroc_auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float]:
    """Compute AUROC and AUPRC for a single side effect type."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan"), float("nan")
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    return auroc, auprc


def ap_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 50) -> float:
    """Average precision at k (AP@k)."""
    sorted_idx = np.argsort(y_score)[::-1][:k]
    hits = y_true[sorted_idx]
    if hits.sum() == 0:
        return 0.0
    precision_at_i = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((precision_at_i * hits).sum() / min(k, y_true.sum()))


def stratified_evaluate(
    model,
    test_triples: np.ndarray,
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    drug_to_id: Dict[str, int],
    drug_targets: Dict[str, Set[str]],
    protein_to_id: Dict[str, int],
    n_negative_samples: int = 100,
    device: str = "cpu",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run stratified evaluation over polypharmacy test triples.

    For each side effect type, evaluates per coverage bin.
    Follows the Decagon evaluation protocol: for each positive test triple
    (drug_i, SE_r, drug_j), sample n_negative_samples negative drug_j
    candidates and rank the positive among them.

    Args:
        model: trained PyKEEN model.
        test_triples: (n, 3) array of (head_label, relation_label, tail_label).
        entity_to_id: entity label -> id.
        relation_to_id: relation label -> id.
        drug_to_id: drug identifier -> integer index (without "drug:" prefix).
        drug_targets: drug identifier -> set of target protein identifiers.
        protein_to_id: protein identifier -> integer index.
        n_negative_samples: negatives per positive.
        device: torch device string.
        random_state: RNG seed for negative sampling.

    Returns:
        DataFrame with columns:
            side_effect, bin, n_pairs, auroc, auprc, ap50
    """
    import torch

    rng = np.random.default_rng(random_state)
    bin_map = assign_target_bins(drug_targets, protein_to_id)

    # Get all drug entity ids
    all_drug_entity_ids = [
        entity_to_id[f"drug:{d}"] for d in drug_to_id if f"drug:{d}" in entity_to_id
    ]
    all_drug_entity_ids = np.array(all_drug_entity_ids)

    # Group test triples by relation (side effect type)
    se_triples: Dict[str, List] = defaultdict(list)
    for triple in test_triples:
        h_label, r_label, t_label = triple
        if r_label.startswith("SE:"):
            se_triples[r_label].append((h_label, t_label))

    results = []
    model.eval()

    for se_label, pairs in se_triples.items():
        se_scores_by_bin: Dict[str, Tuple[List, List]] = {
            b: ([], []) for b in BIN_THRESHOLDS
        }

        for h_label, t_label in pairs:
            h_id = entity_to_id.get(h_label)
            t_id = entity_to_id.get(t_label)
            r_id = relation_to_id.get(se_label)
            if h_id is None or t_id is None or r_id is None:
                continue

            # Determine bin for this pair (use lower-coverage drug)
            h_drug = h_label.replace("drug:", "")
            t_drug = t_label.replace("drug:", "")
            h_bin = bin_map.get(h_drug, "none")
            t_bin = bin_map.get(t_drug, "none")
            # Choose the bin of the drug with fewer targets (harder case)
            pair_bin = h_bin if (
                list(BIN_THRESHOLDS.keys()).index(h_bin) <=
                list(BIN_THRESHOLDS.keys()).index(t_bin)
            ) else t_bin

            # Sample negatives (replace tail)
            neg_ids = rng.choice(all_drug_entity_ids, size=n_negative_samples, replace=False)
            neg_ids = neg_ids[neg_ids != t_id][:n_negative_samples]

            candidate_ids = np.concatenate([[t_id], neg_ids])
            labels = np.zeros(len(candidate_ids))
            labels[0] = 1.0

            h_tensor = torch.tensor([h_id] * len(candidate_ids), dtype=torch.long).to(device)
            r_tensor = torch.tensor([r_id] * len(candidate_ids), dtype=torch.long).to(device)
            t_tensor = torch.tensor(candidate_ids, dtype=torch.long).to(device)

            with torch.no_grad():
                scores = model.score_hrt(
                    torch.stack([h_tensor, r_tensor, t_tensor], dim=1)
                ).squeeze(-1).cpu().numpy()

            se_scores_by_bin[pair_bin][0].extend(labels.tolist())
            se_scores_by_bin[pair_bin][1].extend(scores.tolist())

        # Compute metrics per bin
        for bin_name, (y_true_list, y_score_list) in se_scores_by_bin.items():
            if not y_true_list:
                continue
            y_true = np.array(y_true_list)
            y_score = np.array(y_score_list)
            auroc, auprc = compute_auroc_auprc(y_true, y_score)
            ap50 = ap_at_k(y_true, y_score, k=50)
            results.append({
                "side_effect": se_label,
                "bin": bin_name,
                "n_pairs": int(y_true.sum()),
                "auroc": auroc,
                "auprc": auprc,
                "ap50": ap50,
            })

    df = pd.DataFrame(results)
    return df


def summarize_stratified(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-side-effect results by coverage bin.

    Returns a summary DataFrame with median metrics per bin.
    """
    summary = (
        df.groupby("bin")[["auroc", "auprc", "ap50"]]
        .median()
        .reset_index()
    )
    counts = df.groupby("bin")["n_pairs"].sum().reset_index().rename(
        columns={"n_pairs": "total_pairs"}
    )
    summary = summary.merge(counts, on="bin")

    # Re-order bins by severity
    bin_order = ["none", "low", "medium", "high"]
    summary["bin"] = pd.Categorical(summary["bin"], categories=bin_order, ordered=True)
    summary = summary.sort_values("bin").reset_index(drop=True)
    return summary


def print_stratified_report(summary: pd.DataFrame, model_name: str = "Model"):
    """Print a formatted stratified evaluation report."""
    header = f"\n{'='*60}\n  Stratified Evaluation: {model_name}\n{'='*60}"
    print(header)
    print(f"{'Bin':<10} {'Pairs':>8} {'AUROC':>8} {'AUPRC':>8} {'AP@50':>8}")
    print("-" * 46)
    for _, row in summary.iterrows():
        print(
            f"{row['bin']:<10} {row['total_pairs']:>8} "
            f"{row['auroc']:>8.3f} {row['auprc']:>8.3f} {row['ap50']:>8.3f}"
        )
    print("=" * 60)
