#!/usr/bin/env python
"""
Interpretability analysis for monopharmacy side effect encodings.

Demonstrates the interpretability advantage of TF-IDF and CUR over raw
PCA (as used in the Non-naive baseline):

  1. TF-IDF: shows top discriminative side effects per drug
  2. CUR:    shows which real side effects were selected as basis columns

Usage:
    python scripts/interpretability_analysis.py \
        --raw_dir data/raw \
        --output_dir outputs/interpretability \
        --mono_method tfidf

Output:
    - CSV of top TF-IDF side effects per drug
    - CSV of CUR-selected side effects
    - Plots: TF-IDF weight distributions, SE frequency heatmap
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("interpretability")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="outputs/interpretability")
    parser.add_argument("--mono_method", choices=["tfidf", "cur", "both"], default="both")
    parser.add_argument("--top_k_ses", type=int, default=10,
                        help="Top-k side effects to show per drug (TF-IDF only)")
    parser.add_argument("--n_example_drugs", type=int, default=20,
                        help="Number of example drugs to analyse")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.data import load_decagon
    import numpy as np
    import pandas as pd

    logger.info("Loading Decagon data ...")
    data = load_decagon(raw_dir=args.raw_dir)

    if data.mono_se_matrix is None:
        logger.error("No monopharmacy matrix found. Check that mono CSV was downloaded.")
        return

    mono_mat = data.mono_se_matrix
    id_to_drug = {v: k for k, v in data.drug_to_id.items()}
    id_to_se_mono = {v: k for k, v in data.se_mono_to_id.items()}

    # ------------------------------------------------------------------ #
    # TF-IDF analysis
    # ------------------------------------------------------------------ #
    if args.mono_method in ("tfidf", "both"):
        from src.encoders.drug_encoder import TFIDFMonoEncoder

        logger.info("Fitting TF-IDF encoder ...")
        tfidf_enc = TFIDFMonoEncoder(n_components=None)  # no reduction for interpretability
        tfidf_enc.fit(mono_mat)

        # Collect top-k SEs for a sample of drugs
        records = []
        drug_indices = list(range(min(args.n_example_drugs, mono_mat.shape[0])))

        for drug_idx in drug_indices:
            drug_id = id_to_drug.get(drug_idx, f"drug_{drug_idx}")
            top_ses = tfidf_enc.top_side_effects(
                drug_idx=drug_idx,
                mono_matrix=mono_mat,
                se_id_to_name=id_to_se_mono,
                k=args.top_k_ses,
            )
            for rank, (se_name, weight) in enumerate(top_ses, 1):
                records.append({
                    "drug": drug_id,
                    "rank": rank,
                    "side_effect": se_name,
                    "tfidf_weight": weight,
                })

        df_tfidf = pd.DataFrame(records)
        out_path = out_dir / "tfidf_top_ses_per_drug.csv"
        df_tfidf.to_csv(out_path, index=False)
        logger.info(f"TF-IDF analysis saved to {out_path}")

        # IDF values — which SEs are most discriminative overall
        idf_df = pd.DataFrame({
            "side_effect": [id_to_se_mono.get(i, str(i)) for i in range(len(tfidf_enc._idf))],
            "idf": tfidf_enc._idf,
            "doc_freq": (mono_mat > 0).sum(axis=0),
        }).sort_values("idf", ascending=False)
        idf_out = out_dir / "se_idf_ranking.csv"
        idf_df.to_csv(idf_out, index=False)
        logger.info(f"IDF ranking saved to {idf_out}")

        print("\n--- Top-20 most discriminative side effects (highest IDF) ---")
        print(idf_df.head(20).to_string(index=False))

        print("\n--- Top-20 most common side effects (lowest IDF) ---")
        print(idf_df.tail(20).to_string(index=False))

    # ------------------------------------------------------------------ #
    # CUR analysis
    # ------------------------------------------------------------------ #
    if args.mono_method in ("cur", "both"):
        from src.encoders.drug_encoder import CURMonoEncoder

        logger.info("Fitting CUR encoder ...")
        cur_enc = CURMonoEncoder(n_components=128, leverage_sampling=True)
        cur_enc.fit(mono_mat)

        selected = cur_enc.selected_side_effect_indices
        selected_names = [id_to_se_mono.get(int(i), str(i)) for i in selected]
        col_norms = (mono_mat ** 2).sum(axis=0)

        cur_df = pd.DataFrame({
            "col_index": selected,
            "side_effect": selected_names,
            "column_norm_sq": col_norms[selected],
            "drug_freq": (mono_mat[:, selected] > 0).sum(axis=0),
        }).sort_values("column_norm_sq", ascending=False)

        cur_out = out_dir / "cur_selected_side_effects.csv"
        cur_df.to_csv(cur_out, index=False)
        logger.info(f"CUR selected side effects saved to {cur_out}")

        print("\n--- CUR-selected side effects (top-20 by column norm) ---")
        print(cur_df.head(20).to_string(index=False))

    logger.info(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
