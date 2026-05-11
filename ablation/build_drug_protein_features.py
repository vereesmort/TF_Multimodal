"""
build_drug_protein_features.py
-------------------------------
Aggregate ESM-2 protein embeddings per drug via drug→target associations.

Why this is needed
------------------
The combo file has drug-drug pairs. The ESM-2 file has protein embeddings.
They do not connect directly. This script bridges them:

  drug A → [target protein 1, target protein 2, ...] → mean(ESM-2 vectors)

The result is a "drug-via-targets" embedding: a 256-dim vector per drug
that summarises the sequence/structural properties of that drug's target proteins.

This is used in the ablation conditions B and E:
  Condition B (0 + ESM-2):   drug features = zeros, protein feature = drug-via-targets
  Condition E (full model):  drug features = ChemBERTa + mono,
                             protein feature = drug-via-targets (or PPI-pooled version)

For drugs with NO target annotations (361/645 in Decagon):
  - Default: use a zero vector (unknown target profile)
  - Alternative: use the global mean of all protein embeddings (mean imputation)
  Both are implemented; zero is the default because mean imputation would
  artificially inflate the similarity between unannotated drugs.

Output
------
  <out_tensor_name>.pt      — (n_drugs, 256) tensor, ordered by drug_to_idx
  target_coverage_report.csv — per-drug annotation coverage
  drug_to_idx.json           — drug string → row index used for tensor rows

Usage
-----
  # ESM branch (default output name)
  python build_drug_protein_features.py \
    --targets   bio-decagon-targets.csv \
    --combo     bio-decagon-combo.csv \
    --prot_emb  protein_init_dim256_esm2_t30_150M_UR50D.pt \
    --output    ./

  # PPI branch with custom output tensor filename
  python build_drug_protein_features.py \
    --targets          bio-decagon-targets.csv \
    --combo            bio-decagon-combo.csv \
    --prot_emb         protein_emb_ppi_neighbour_dim256_esm2_t30_150M_UR50D.pt \
    --out_tensor_name  drug_via_mean_ppi.pt \
    --output           ./
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets",   required=True, help="bio-decagon-targets.csv")
    p.add_argument("--combo",     required=True, help="bio-decagon-combo.csv")
    p.add_argument("--prot_emb",  required=True, help="(n_proteins, 256) ESM-2 tensor")
    p.add_argument("--prot_idx",  default=None,
                   help="JSON mapping gene_id_str → row index in prot_emb. "
                        "If None, assumes rows are ordered by sorted gene ID.")
    p.add_argument("--output",    default="./")
    p.add_argument("--out_tensor_name", default="drug_via_targets_256.pt",
                   help="Output filename for drug-level tensor (e.g. drug_via_mean_ppi.pt)")
    p.add_argument("--impute",    default="zero",
                   choices=["zero", "mean"],
                   help="How to handle drugs with no target annotations")
    p.add_argument("--min_edges", type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load files ─────────────────────────────────────────────────────────
    targets = pd.read_csv(args.targets)
    targets["STITCH"] = targets["STITCH"].astype(str)
    targets["Gene"]   = targets["Gene"].astype(str)

    combo = pd.read_csv(args.combo)
    se_counts = combo.groupby("Polypharmacy Side Effect").size()
    valid_ses = se_counts[se_counts >= args.min_edges].index
    combo = combo[combo["Polypharmacy Side Effect"].isin(valid_ses)]

    all_drugs = sorted(pd.concat([combo["STITCH 1"], combo["STITCH 2"]]).unique())
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    n_drugs = len(all_drugs)
    print(f"Drugs in 963-SE graph: {n_drugs}")

    # ── Load protein embeddings ────────────────────────────────────────────
    prot_emb = torch.load(args.prot_emb, map_location="cpu").float()
    dim = prot_emb.shape[1]
    print(f"Protein embedding shape: {prot_emb.shape}")

    # Build gene → row index mapping
    if args.prot_idx is not None:
        with open(args.prot_idx) as f:
            gene_to_row = json.load(f)
        gene_to_row = {str(k): int(v) for k, v in gene_to_row.items()}
    else:
        # Assume rows correspond to genes in sorted order
        # This must match how you built the ESM-2 tensor
        # Adjust if your ordering is different
        all_genes_in_targets = sorted(targets["Gene"].unique())
        gene_to_row = {g: i for i, g in enumerate(all_genes_in_targets)}
        print("WARNING: assuming protein rows are ordered by sorted gene ID. "
              "Pass --prot_idx if this is incorrect.")

    # ── Build drug→targets→mean ESM-2 vector ──────────────────────────────
    drug_targets = defaultdict(list)
    for _, row in targets.iterrows():
        drug_targets[row["STITCH"]].append(row["Gene"])

    # Imputation vector for drugs with no targets
    if args.impute == "zero":
        fallback = torch.zeros(dim)
    else:
        fallback = prot_emb.mean(dim=0)

    drug_via_targets = torch.zeros(n_drugs, dim)
    coverage = []

    for drug, idx in drug_to_idx.items():
        gene_list = drug_targets.get(drug, [])
        known_genes = [g for g in gene_list if g in gene_to_row]

        if known_genes:
            rows = [gene_to_row[g] for g in known_genes]
            vecs = prot_emb[rows]           # (n_targets, dim)
            drug_via_targets[idx] = vecs.mean(dim=0)
        else:
            drug_via_targets[idx] = fallback

        coverage.append({
            "drug": drug,
            "n_targets_in_file": len(gene_list),
            "n_targets_with_embedding": len(known_genes),
            "has_target_signal": len(known_genes) > 0,
        })

    cov_df = pd.DataFrame(coverage)
    n_covered = cov_df["has_target_signal"].sum()
    print(f"\nDrugs with target signal: {n_covered} / {n_drugs} "
          f"({n_covered/n_drugs*100:.1f}%)")
    print(f"Drugs using {args.impute} imputation: {n_drugs - n_covered}")

    # ── Save ───────────────────────────────────────────────────────────────
    out_tensor = output_dir / args.out_tensor_name
    out_cov    = output_dir / "target_coverage_report.csv"
    out_idx    = output_dir / "drug_to_idx.json"

    torch.save(drug_via_targets, out_tensor)
    cov_df.to_csv(out_cov, index=False)
    with open(out_idx, "w") as f:
        json.dump(drug_to_idx, f)

    print(f"\nSaved:")
    print(f"  {out_tensor}  — shape {drug_via_targets.shape}")
    print(f"  {out_cov}")
    print(f"  {out_idx}")
    print(f"\nUse {args.out_tensor_name} as --prot_emb_esm2 or --prot_emb_ppi in")
    print("ablation_track1_ml.py / hpo_track1_optuna.py depending on branch.")


if __name__ == "__main__":
    main()
