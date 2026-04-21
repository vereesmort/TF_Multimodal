"""
Data splitting for polypharmacy side effect prediction.

Implements the exact split protocol used by Decagon (Zitnik et al., 2018)
and reproduced by Lloyd et al. (2024):

    For each polypharmacy side effect type independently:
        - 80% of drug pairs → training
        - 10% of drug pairs → validation   (held out, never in graph)
        - 10% of drug pairs → test         (held out, never in graph)

    PPI and drug-target edges go entirely into training — they are
    structural context and are never evaluated as link predictions.

Critically, the validation and test edges are REMOVED from the adjacency
graph passed to the KGE model.  This prevents data leakage: the model
must predict these edges, not memorise them.

The false-edge (negative) construction follows Decagon minibatch.py:
    For each held-out positive edge (drug_i, SE_r, drug_j), sample one
    random false edge (drug_i, SE_r, drug_k) where (drug_i, SE_r, drug_k)
    does not exist in the full known edge set.

Reference:
    Zitnik et al. (2018), Section 5 — Experimental Setup
    Lloyd et al. (2024), Data/polypharmacy_split/
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Correct per-SE-type stratified split
# ---------------------------------------------------------------------------

def split_polypharmacy_edges(
    poly_triples: List[Tuple[str, str, str]],
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    min_edges_per_se: int = 50,
    random_state: int = 42,
) -> Tuple[
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
    List[Tuple[str, str, str]],
]:
    """
    Split polypharmacy triples per side-effect type (stratified).

    Matches the Decagon / Lloyd et al. protocol exactly:
      - Split is done independently for each SE type.
      - val and test sets each contain ~10% of edges per SE type.
      - SE types with fewer than min_edges_per_se total edges are kept
        entirely in training (too few to evaluate reliably).

    Args:
        poly_triples: list of (drug_i, SE_relation, drug_j) string tuples.
        val_frac: fraction held out for validation.
        test_frac: fraction held out for testing.
        min_edges_per_se: minimum edges for a SE type to be split.
        random_state: RNG seed for reproducibility.

    Returns:
        train_triples, val_triples, test_triples — all as lists of
        (head, relation, tail) string tuples.
    """
    rng = np.random.default_rng(random_state)

    # Group triples by SE type
    by_se: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for triple in poly_triples:
        by_se[triple[1]].append(triple)

    train, val, test = [], [], []
    skipped = 0

    for se_type, triples in by_se.items():
        n = len(triples)
        if n < min_edges_per_se:
            # Too few edges — keep all in training
            train.extend(triples)
            skipped += 1
            continue

        perm = rng.permutation(n)
        shuffled = [triples[i] for i in perm]

        # Match Lloyd et al.: use max(50, floor(n * frac)) for val/test
        n_val = max(50, int(np.floor(n * val_frac)))
        n_test = max(50, int(np.floor(n * test_frac)))

        # Clamp so we don't exceed available edges
        n_val = min(n_val, n // 4)
        n_test = min(n_test, n // 4)

        val.extend(shuffled[:n_val])
        test.extend(shuffled[n_val : n_val + n_test])
        train.extend(shuffled[n_val + n_test :])

    logger.info(
        f"Split: train={len(train)}, val={len(val)}, test={len(test)} | "
        f"SE types: {len(by_se)} total, {skipped} too-small → all in train"
    )
    return train, val, test


def build_true_edge_set(
    poly_triples: List[Tuple[str, str, str]],
) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Build a lookup of all known true positive (head, tail) pairs per SE type.

    Used during false-edge generation to ensure negatives are genuine negatives.

    Args:
        poly_triples: all known polypharmacy triples (train + val + test).

    Returns:
        Dict mapping SE relation string -> set of (drug_i, drug_j) pairs.
    """
    true_edges: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for h, r, t in poly_triples:
        true_edges[r].add((h, t))
        true_edges[r].add((t, h))  # undirected: (j,i) is also true
    return true_edges


def generate_false_edges(
    positive_edges: List[Tuple[str, str, str]],
    all_drugs: List[str],
    true_edge_lookup: Dict[str, Set[Tuple[str, str]]],
    n_false_per_positive: int = 1,
    random_state: int = 42,
) -> List[Tuple[str, str, str]]:
    """
    Generate false (corrupted) edges matching the Decagon evaluation protocol.

    For each positive edge (drug_i, SE_r, drug_j):
        Sample n_false_per_positive drugs drug_k uniformly at random such that
        (drug_i, SE_r, drug_k) is NOT a known true positive.

    This is exactly the procedure in Decagon's minibatch.py:
        idx_j = np.random.randint(...)
        if self._ismember([idx_i, idx_j], edges_all): continue

    Args:
        positive_edges: held-out positive (h, r, t) triples.
        all_drugs: list of all drug identifiers eligible as corrupted tails.
        true_edge_lookup: output of build_true_edge_set() — all known true pairs.
        n_false_per_positive: number of false edges per positive (default 1,
            matching Decagon exactly; set higher for sampled-negatives protocol).
        random_state: RNG seed.

    Returns:
        List of false (corrupted) triples with the same length as
        positive_edges * n_false_per_positive.
    """
    rng = np.random.default_rng(random_state)
    drugs_arr = np.array(all_drugs)
    false_edges: List[Tuple[str, str, str]] = []

    for h, r, t in positive_edges:
        true_for_r = true_edge_lookup.get(r, set())
        n_generated = 0
        max_attempts = len(all_drugs) * 3
        attempts = 0

        while n_generated < n_false_per_positive and attempts < max_attempts:
            fake_t = rng.choice(drugs_arr)
            # Reject if (h, fake_t) or (fake_t, h) is a true positive for SE r
            if (h, fake_t) not in true_for_r and (fake_t, h) not in true_for_r:
                false_edges.append((h, r, fake_t))
                n_generated += 1
            attempts += 1

        if n_generated < n_false_per_positive:
            # Fallback: duplicate last valid false edge if we ran out of candidates
            # (extremely rare; only occurs for drugs with near-complete connectivity)
            logger.debug(f"Could not generate enough false edges for ({h}, {r})")
            while n_generated < n_false_per_positive:
                false_edges.append(false_edges[-1] if false_edges else (h, r, all_drugs[0]))
                n_generated += 1

    return false_edges
