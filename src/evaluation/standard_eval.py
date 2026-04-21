"""
Standard evaluation metrics matching the Decagon paper.

Computes AUROC, AUPRC, and AP@50 per polypharmacy side effect type,
then reports median across all 963 side effects.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_pse(
    model,
    test_triples: np.ndarray,
    entity_to_id: Dict[str, int],
    relation_to_id: Dict[str, int],
    n_negative_samples: int = 100,
    device: str = "cpu",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Standard polypharmacy side effect evaluation.

    For each PSE type, computes AUROC, AUPRC, AP@50 using sampled negatives
    (following the Decagon evaluation protocol).

    Returns:
        DataFrame with columns: side_effect, n_positives, auroc, auprc, ap50
    """
    import torch
    from .stratified_eval import compute_auroc_auprc, ap_at_k
    from collections import defaultdict

    rng = np.random.default_rng(random_state)
    all_drug_ids = np.array([
        eid for label, eid in entity_to_id.items() if label.startswith("drug:")
    ])

    se_triples: Dict[str, List] = defaultdict(list)
    for triple in test_triples:
        h, r, t = triple
        if r.startswith("SE:"):
            se_triples[r].append((h, t))

    results = []
    model.eval()

    for se_label, pairs in se_triples.items():
        y_true_all, y_score_all = [], []
        r_id = relation_to_id.get(se_label)
        if r_id is None:
            continue

        for h_label, t_label in pairs:
            h_id = entity_to_id.get(h_label)
            t_id = entity_to_id.get(t_label)
            if h_id is None or t_id is None:
                continue

            neg_ids = rng.choice(all_drug_ids, size=n_negative_samples + 1, replace=False)
            neg_ids = neg_ids[neg_ids != t_id][:n_negative_samples]
            candidate_ids = np.concatenate([[t_id], neg_ids])
            labels = np.zeros(len(candidate_ids))
            labels[0] = 1.0

            h_t = torch.tensor([h_id] * len(candidate_ids), dtype=torch.long).to(device)
            r_t = torch.tensor([r_id] * len(candidate_ids), dtype=torch.long).to(device)
            t_t = torch.tensor(candidate_ids, dtype=torch.long).to(device)

            with torch.no_grad():
                scores = model.score_hrt(
                    torch.stack([h_t, r_t, t_t], dim=1)
                ).squeeze(-1).cpu().numpy()

            y_true_all.extend(labels.tolist())
            y_score_all.extend(scores.tolist())

        y_true = np.array(y_true_all)
        y_score = np.array(y_score_all)
        auroc, auprc = compute_auroc_auprc(y_true, y_score)
        ap50 = ap_at_k(y_true, y_score, k=50)
        results.append({
            "side_effect": se_label,
            "n_positives": int(y_true.sum()),
            "auroc": auroc,
            "auprc": auprc,
            "ap50": ap50,
        })

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame, model_name: str = "Model"):
    """Print median metrics across all side effects."""
    print(f"\n{'='*50}")
    print(f"  {model_name} — overall evaluation (median)")
    print(f"{'='*50}")
    print(f"  Side effects evaluated : {len(df)}")
    print(f"  Median AUROC           : {df['auroc'].median():.4f}")
    print(f"  Median AUPRC           : {df['auprc'].median():.4f}")
    print(f"  Median AP@50           : {df['ap50'].median():.4f}")
    print(f"{'='*50}\n")
