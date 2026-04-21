"""
Evaluation protocols for polypharmacy side effect prediction.

Two protocols are implemented, both computing AUROC, AUPRC, AP@50
per side effect type and reporting the median across all SE types.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROTOCOL 1 — Decagon / Lloyd et al. ("false edge") protocol
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exactly matches the evaluation used in:
    - Zitnik et al. (2018) Decagon paper
    - Lloyd et al. (2024) TF-Decagon paper

Construction:
    For every SE type r:
        positives = held-out test edges for r
        For each positive (drug_i, r, drug_j), generate ONE false edge
        (drug_i, r, drug_k) where (drug_i, r, drug_k) is not a known
        true positive — sampled uniformly at random from all drugs.

    This gives a balanced 1:1 positive:negative ratio per SE type.

    Metrics (AUROC, AUPRC, AP@50) are computed per SE type, then the
    median is reported across all SE types.

This is the protocol to use when reporting results comparable to the
existing literature.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROTOCOL 2 — Sampled negatives ("N-per-positive") protocol
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A widely used approximation where each positive is ranked against a
fixed pool of N sampled negatives (e.g., N=100).

Construction:
    For every SE type r:
        For each positive (drug_i, r, drug_j):
            Sample N false edges (drug_i, r, drug_k) where k ≠ j
            and (drug_i, SE_r, drug_k) is not a known true positive.
            Score all N+1 candidates (1 positive + N negatives).
            Record labels and scores.

    Metrics are computed over the pooled (positive, negatives) set
    per SE type, then median reported.

Use this for fast ablation studies; results are not directly comparable
to Protocol 1 because the harder-negative ratio inflates AUROC/AUPRC.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATIFIED VARIANT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Both protocols support stratified breakdown by drug PPI coverage
(none / low / medium / high known protein targets) to test robustness
to missing biological data. See evaluate_stratified().
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Coverage bin thresholds (# of PPI-covered protein targets)
BIN_THRESHOLDS = {
    "none":   (0,  0),
    "low":    (1,  2),
    "medium": (3,  9),
    "high":   (10, int(1e9)),
}
BIN_ORDER = ["none", "low", "medium", "high"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _auroc_auprc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan"), float("nan")
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)


def _ap_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 50) -> float:
    """AP@k using the original Decagon rank_metrics.apk logic."""
    sorted_idx = np.argsort(y_score)[::-1]
    sorted_labels = y_true[sorted_idx]
    score, hits = 0.0, 0.0
    for i in range(min(k, len(sorted_labels))):
        if sorted_labels[i] == 1:
            hits += 1.0
            score += hits / (i + 1.0)
    n_pos = min(int(y_true.sum()), k)
    return score / n_pos if n_pos > 0 else 0.0


def _score_candidates(
    model,
    h_id: int,
    r_id: int,
    candidate_ids: np.ndarray,
    device: str,
) -> np.ndarray:
    """Score a batch of (h, r, candidate_t) triples with the model."""
    import torch
    n = len(candidate_ids)
    h_t = torch.tensor([h_id] * n, dtype=torch.long, device=device)
    r_t = torch.tensor([r_id] * n, dtype=torch.long, device=device)
    t_t = torch.tensor(candidate_ids, dtype=torch.long, device=device)
    hrt = torch.stack([h_t, r_t, t_t], dim=1)
    with torch.no_grad():
        scores = model.score_hrt(hrt).squeeze(-1).cpu().numpy()
    return scores


# ---------------------------------------------------------------------------
# Protocol 1: Decagon false-edge protocol
# ---------------------------------------------------------------------------

def evaluate_false_edge_protocol(
    model,
    test_triples: List[Tuple[str, str, str]],
    false_triples: List[Tuple[str, str, str]],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Protocol 1 — Decagon / Lloyd et al. false-edge evaluation.

    Uses pre-generated false edges (1 false per positive, same head and
    relation type, random tail filtered to exclude true positives).

    This is the evaluation that produces results directly comparable to
    numbers reported in the Decagon and TF-Decagon papers.

    Args:
        model: trained PyKEEN model (must implement .score_hrt()).
        test_triples: held-out positive (h_label, r_label, t_label) triples.
        false_triples: pre-generated false edges from generate_false_edges()
            with n_false_per_positive=1. Must have same length as test_triples.
        entity_to_id: entity label -> integer id.
        relation_to_id: relation label -> integer id.
        device: torch device string.

    Returns:
        DataFrame with columns: side_effect, n_pos, auroc, auprc, ap50
    """
    assert len(test_triples) == len(false_triples), (
        f"test ({len(test_triples)}) and false ({len(false_triples)}) "
        f"lists must have the same length"
    )

    import torch
    model.eval()

    # Group by SE type
    by_se: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    # (h_id, r_id, pos_t_id, neg_t_id)

    for (h_pos, r_pos, t_pos), (h_neg, r_neg, t_neg) in zip(test_triples, false_triples):
        r_id = relation_to_id.get(r_pos)
        h_id = entity_to_id.get(h_pos)
        t_id = entity_to_id.get(t_pos)
        neg_t_id = entity_to_id.get(t_neg)
        if None in (r_id, h_id, t_id, neg_t_id):
            continue
        by_se[r_pos].append((h_id, r_id, t_id, neg_t_id))

    results = []
    for se_label, quad_list in by_se.items():
        # Score all positives and negatives for this SE type in one pass
        h_ids = np.array([q[0] for q in quad_list])
        r_ids = np.array([q[1] for q in quad_list])
        pos_t_ids = np.array([q[2] for q in quad_list])
        neg_t_ids = np.array([q[3] for q in quad_list])

        all_h = np.concatenate([h_ids, h_ids])
        all_r = np.concatenate([r_ids, r_ids])
        all_t = np.concatenate([pos_t_ids, neg_t_ids])
        all_labels = np.concatenate([
            np.ones(len(pos_t_ids)),
            np.zeros(len(neg_t_ids)),
        ])

        import torch
        hrt = torch.tensor(
            np.stack([all_h, all_r, all_t], axis=1),
            dtype=torch.long, device=device
        )
        with torch.no_grad():
            scores = model.score_hrt(hrt).squeeze(-1).cpu().numpy()

        auroc, auprc = _auroc_auprc(all_labels, scores)
        ap50 = _ap_at_k(all_labels, scores, k=50)

        results.append({
            "side_effect": se_label,
            "n_pos": int(len(pos_t_ids)),
            "auroc": auroc,
            "auprc": auprc,
            "ap50": ap50,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Protocol 2: Sampled negatives protocol
# ---------------------------------------------------------------------------

def evaluate_sampled_negatives_protocol(
    model,
    test_triples: List[Tuple[str, str, str]],
    all_drug_labels: List[str],
    true_edge_lookup: Dict[str, Set[Tuple[str, str]]],
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    n_negatives: int = 100,
    device: str = "cpu",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Protocol 2 — Sampled negatives (N-per-positive) evaluation.

    For each positive test triple, scores it against N randomly sampled
    negatives (filtered to exclude true positives). Suitable for fast
    ablation runs, but produces inflated AUROC/AUPRC compared to Protocol 1.

    Args:
        model: trained PyKEEN model.
        test_triples: held-out positive (h, r, t) triples.
        all_drug_labels: list of all drug entity labels eligible as negatives.
        true_edge_lookup: output of build_true_edge_set() — all known true pairs.
        entity_to_id: entity label -> integer id.
        relation_to_id: relation label -> integer id.
        n_negatives: number of sampled negatives per positive (default 100).
        device: torch device string.
        random_state: RNG seed.

    Returns:
        DataFrame with columns: side_effect, n_pos, auroc, auprc, ap50
    """
    import torch
    rng = np.random.default_rng(random_state)
    drugs_arr = np.array(all_drug_labels)
    all_drug_ids = np.array([
        entity_to_id[d] for d in drugs_arr if d in entity_to_id
    ])
    drug_label_to_id = {d: entity_to_id[d] for d in drugs_arr if d in entity_to_id}

    model.eval()

    by_se: Dict[str, List[Tuple]] = defaultdict(list)
    for triple in test_triples:
        by_se[triple[1]].append(triple)

    results = []
    for se_label, triples in by_se.items():
        r_id = relation_to_id.get(se_label)
        if r_id is None:
            continue
        true_for_r = true_edge_lookup.get(se_label, set())

        y_true_all, y_score_all = [], []

        for h_label, _, t_label in triples:
            h_id = entity_to_id.get(h_label)
            t_id = entity_to_id.get(t_label)
            if h_id is None or t_id is None:
                continue

            # Sample N negatives: random drugs, filtered to exclude true positives
            neg_ids = []
            attempts = 0
            perm = rng.permutation(len(all_drug_ids))
            for idx in perm:
                fake_label = drugs_arr[idx] if idx < len(drugs_arr) else drugs_arr[0]
                if (
                    (h_label, fake_label) not in true_for_r
                    and (fake_label, h_label) not in true_for_r
                ):
                    neg_ids.append(all_drug_ids[idx])
                    if len(neg_ids) == n_negatives:
                        break

            if not neg_ids:
                continue

            # Score positive + negatives
            candidate_ids = np.array([t_id] + neg_ids)
            labels = np.zeros(len(candidate_ids))
            labels[0] = 1.0

            scores = _score_candidates(model, h_id, r_id, candidate_ids, device)
            y_true_all.extend(labels.tolist())
            y_score_all.extend(scores.tolist())

        if not y_true_all:
            continue

        y_true = np.array(y_true_all)
        y_score = np.array(y_score_all)
        auroc, auprc = _auroc_auprc(y_true, y_score)
        ap50 = _ap_at_k(y_true, y_score, k=50)

        results.append({
            "side_effect": se_label,
            "n_pos": int(y_true.sum()),
            "auroc": auroc,
            "auprc": auprc,
            "ap50": ap50,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Stratified wrapper (works with either protocol)
# ---------------------------------------------------------------------------

def assign_coverage_bins(
    drug_targets: Dict[str, Set[str]],
    protein_to_id: Dict[str, int],
) -> Dict[str, str]:
    """
    Assign each drug to a DPI (drug-protein interaction) coverage bin.

    Bins are based on how many of the drug's known protein targets appear in
    bio-decagon-targets.csv AND are present as nodes in the Decagon graph.

    This specifically measures DPI data availability — NOT PPI connectivity.
    A drug in the 'none' bin has zero entries in the DPI target file; the model
    can only predict its interactions via molecular structure (ChemBERTa) and
    monopharmacy side effects (TF-IDF/CUR). A drug in 'high' has rich DPI data
    and additionally benefits from protein neighbourhood signal.

    This stratification tests the core thesis hypothesis:
        Semantically grounded initialization (ChemBERTa + TF-IDF/CUR) is robust
        to missing DPI data because these modalities are entirely independent of
        the target file. The baseline degrades in the 'none'/'low' bins because
        without DPI edges it has no structural signal to fall back on.

    Args:
        drug_targets: drug identifier -> set of target protein identifiers,
            loaded from bio-decagon-targets.csv.
        protein_to_id: protein identifiers present as graph nodes.

    Returns:
        Dict mapping drug identifier -> bin name ('none'/'low'/'medium'/'high').
        Drugs absent from drug_targets entirely are assigned 'none'.
    """
    bin_map: Dict[str, str] = {}
    for drug, targets in drug_targets.items():
        # Count targets that are both in the DPI file and present in the graph
        n_covered = sum(1 for t in targets if t in protein_to_id)
        for bin_name, (lo, hi) in BIN_THRESHOLDS.items():
            if lo <= n_covered <= hi:
                bin_map[drug] = bin_name
                break
    return bin_map


def evaluate_stratified(
    results_df: pd.DataFrame,
    test_triples: List[Tuple[str, str, str]],
    false_triples: Optional[List[Tuple[str, str, str]]],
    drug_targets: Dict[str, Set[str]],
    protein_to_id: Dict[str, int],
    model,
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    protocol: str = "false_edge",
    all_drug_labels: Optional[List[str]] = None,
    true_edge_lookup: Optional[Dict[str, Set[Tuple[str, str]]]] = None,
    n_negatives: int = 100,
    device: str = "cpu",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Stratified evaluation by drug PPI coverage bin.

    Runs evaluation separately for each coverage bin (none/low/medium/high)
    using the selected protocol.

    Args:
        results_df: optional overall results from evaluate_false_edge_protocol()
            or evaluate_sampled_negatives_protocol() — if provided, per-bin
            results are computed from scratch by sub-selecting test triples.
        test_triples: all held-out positive triples.
        false_triples: pre-generated false edges (Protocol 1 only).
        drug_targets: drug label -> set of target protein labels.
        protein_to_id: protein label -> integer id (PPI graph proteins).
        model: trained PyKEEN model.
        entity_to_id: entity label -> integer id.
        relation_to_id: relation label -> integer id.
        protocol: 'false_edge' or 'sampled_negatives'.
        all_drug_labels: list of all drug entity labels (Protocol 2 only).
        true_edge_lookup: known true pairs per SE type (Protocol 2 only).
        n_negatives: negatives per positive (Protocol 2 only).
        device: torch device string.
        random_state: RNG seed.

    Returns:
        DataFrame with columns: bin, n_se_types, n_pos, auroc, auprc, ap50
        (median metrics per coverage bin).
    """
    bin_map = assign_coverage_bins(drug_targets, protein_to_id)

    # Build (positive, false) pairs per bin
    bin_results = {}
    for bin_name in BIN_ORDER:
        # Select test triples where the head drug is in this bin
        bin_test = [
            (h, r, t) for h, r, t in test_triples
            if bin_map.get(h.replace("drug:", ""), "none") == bin_name
        ]
        if not bin_test:
            continue

        if protocol == "false_edge":
            assert false_triples is not None, "false_triples required for false_edge protocol"
            # Need to align false triples — build index
            test_set = set((h, r, t) for h, r, t in bin_test)
            # Re-generate false edges for this bin's subset
            from src.data.splitting import generate_false_edges, build_true_edge_set
            all_poly = test_triples  # use full set for true-edge lookup
            lookup = true_edge_lookup or build_true_edge_set(all_poly)
            all_drugs = all_drug_labels or [
                lbl for lbl in entity_to_id if lbl.startswith("drug:")
            ]
            bin_false = generate_false_edges(
                bin_test, all_drugs, lookup,
                n_false_per_positive=1, random_state=random_state
            )
            df_bin = evaluate_false_edge_protocol(
                model, bin_test, bin_false,
                entity_to_id, relation_to_id, device=device
            )

        elif protocol == "sampled_negatives":
            assert all_drug_labels is not None and true_edge_lookup is not None
            df_bin = evaluate_sampled_negatives_protocol(
                model, bin_test, all_drug_labels, true_edge_lookup,
                entity_to_id, relation_to_id,
                n_negatives=n_negatives, device=device, random_state=random_state
            )
        else:
            raise ValueError(f"Unknown protocol: {protocol!r}")

        bin_results[bin_name] = df_bin

    # Aggregate: median per bin
    rows = []
    for bin_name in BIN_ORDER:
        if bin_name not in bin_results:
            continue
        df = bin_results[bin_name].dropna(subset=["auroc", "auprc", "ap50"])
        rows.append({
            "bin":        bin_name,
            "n_se_types": len(df),
            "n_pos":      int(df["n_pos"].sum()),
            "auroc":      df["auroc"].median(),
            "auprc":      df["auprc"].median(),
            "ap50":       df["ap50"].median(),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def summarise(df: pd.DataFrame, model_name: str = "Model", protocol: str = "") -> None:
    """Print a summary table of evaluation results (median over SE types)."""
    df_clean = df.dropna(subset=["auroc", "auprc", "ap50"])
    tag = f" [{protocol}]" if protocol else ""
    print(f"\n{'='*56}")
    print(f"  {model_name}{tag}")
    print(f"  SE types evaluated : {len(df_clean)}")
    print(f"  Median AUROC       : {df_clean['auroc'].median():.4f}")
    print(f"  Median AUPRC       : {df_clean['auprc'].median():.4f}")
    print(f"  Median AP@50       : {df_clean['ap50'].median():.4f}")
    print(f"{'='*56}\n")


def summarise_stratified(df: pd.DataFrame, model_name: str = "Model", protocol: str = "") -> None:
    """Print stratified results table."""
    tag = f" [{protocol}]" if protocol else ""
    print(f"\n{'='*60}")
    print(f"  Stratified Evaluation: {model_name}{tag}")
    print(f"{'='*60}")
    print(f"  {'Bin':<10} {'SE types':>9} {'Pos pairs':>10} {'AUROC':>8} {'AUPRC':>8} {'AP@50':>8}")
    print(f"  {'-'*56}")
    for _, row in df.iterrows():
        print(
            f"  {row['bin']:<10} {int(row['n_se_types']):>9} {int(row['n_pos']):>10} "
            f"{row['auroc']:>8.4f} {row['auprc']:>8.4f} {row['ap50']:>8.4f}"
        )
    print(f"{'='*60}\n")
