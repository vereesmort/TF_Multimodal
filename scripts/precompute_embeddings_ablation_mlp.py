#!/usr/bin/env python
"""
precompute_embeddings_ablation_mlp.py
--------------------------------------
Precompute RAW (un-projected) per-modality feature tensors for
ablation_track1_mlp.py.

Unlike precompute_embeddings_ablation.py — which applies a fixed random
_ablation_mlp_project MLP to reduce every modality to --embedding_dim — this
script saves the FULL-DIMENSIONAL outputs of the frozen pretrained models.
The learnable projection to embedding_dim happens inside AblationNet during
training in ablation_track1_mlp.py.

Key differences from precompute_embeddings_ablation.py
-------------------------------------------------------
  - No _ablation_mlp_project call.
  - Drug tensor ordering follows sorted(drugs_in_filtered_combo), matching the
    drug_to_idx produced by make_entity_index(df) in the training script.
    A drug_to_idx.json is saved alongside each tensor for explicit alignment.
  - Protein→drug aggregation (mean over target ESM-2 vectors) is done here so
    training never needs to load the large protein-level tensors.
  - Both ESM-2 and PPI-aggregated protein embeddings are stored at drug level.

Outputs  (under --output_dir)
-------------------------------
  drug_raw_chemberta.pt             (n_drugs, 768)      — ChemBERTa mean-pool
  drug_raw_mono.pt                  (n_drugs, mono_dim) — TF-IDF or CUR
  drug_raw_esm2_via_targets.pt      (n_drugs, esm_dim)  — ESM-2 mean over targets
  drug_raw_ppi_via_targets.pt       (n_drugs, esm_dim)  — PPI-agg mean over targets
  drug_to_idx.json                  {drug_str: row_index}
  metadata.json                     shapes, model names, esm_dim

Intermediate heavy caches  (under --cache_dir, reused across runs)
-------------------------------------------------------------------
  protein_raw_esm2_{tag}.pt         (n_proteins, esm_dim)
  protein_raw_ppi_{tag}_h{hops}.pt  (n_proteins, esm_dim)

Usage
-----
  python scripts/precompute_embeddings_ablation_mlp.py \\
      --raw_dir    data/raw \\
      --combo      data/raw/bio-decagon-combo.csv \\
      --targets    data/raw/bio-decagon-targets.csv \\
      --cache_dir  data/cache \\
      --output_dir data/cache/ablation_mlp \\
      --esm2_model facebook/esm2_t30_150M_UR50D \\
      --device     cuda

  # Skip drug or protein steps if tensors already exist
  python scripts/precompute_embeddings_ablation_mlp.py ... --skip_drugs
  python scripts/precompute_embeddings_ablation_mlp.py ... --skip_proteins
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("precompute_ablation_mlp")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_sorted_drug_to_idx(combo_path: str, min_edges: int) -> dict[str, int]:
    """
    Build drug→row_index mapping ordered by sorted(drug_strs) over the
    filtered combo file.  Matches make_entity_index(df) in the training script.
    """
    df = pd.read_csv(combo_path)
    se_counts = df.groupby("Polypharmacy Side Effect").size()
    valid_ses = se_counts[se_counts >= min_edges].index
    df = df[df["Polypharmacy Side Effect"].isin(valid_ses)]
    all_drugs = sorted(
        pd.concat([df["STITCH 1"], df["STITCH 2"]]).unique()
    )
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    logger.info(f"Drugs in filtered combo (min_edges={min_edges}): {len(drug_to_idx)}")
    return drug_to_idx


def _build_sorted_gene_to_row(targets_path: str) -> dict[str, int]:
    """
    Build gene→row_index mapping ordered by sorted(unique_genes) from the
    targets CSV.  Matches the resolve_gene_to_row default in ablation_track1_ml.py.
    """
    df = pd.read_csv(targets_path)
    genes = sorted(df["Gene"].astype(str).unique())
    gene_to_row = {g: i for i, g in enumerate(genes)}
    logger.info(f"Unique proteins (sorted gene order): {len(gene_to_row)}")
    return gene_to_row


def _build_reordered_mono_matrix(
    mono_se_matrix: np.ndarray,
    data_drug_to_id: dict[str, int],
    drug_to_idx: dict[str, int],
) -> np.ndarray:
    """
    Reorder mono_se_matrix rows from data.drug_to_id (insertion order) to
    drug_to_idx (sorted order for the filtered combo subset).

    data.drug_to_id may have more drugs than drug_to_idx (it includes drugs
    outside the filtered combo). Only drugs present in drug_to_idx are kept.
    """
    n_drugs  = len(drug_to_idx)
    n_mono   = mono_se_matrix.shape[1]
    out      = np.zeros((n_drugs, n_mono), dtype=np.float32)
    missing  = 0
    for drug, sorted_idx in drug_to_idx.items():
        ins_idx = data_drug_to_id.get(drug)
        if ins_idx is not None and ins_idx < mono_se_matrix.shape[0]:
            out[sorted_idx] = mono_se_matrix[ins_idx]
        else:
            missing += 1
    if missing:
        logger.warning(
            f"{missing} drugs in drug_to_idx not found in mono_se_matrix; "
            "those rows are zero (drug has no mono-SE annotations)."
        )
    return out


def _aggregate_protein_to_drug(
    prot_emb: torch.Tensor,
    targets_df: pd.DataFrame,
    drug_to_idx: dict[str, int],
    gene_to_row: dict[str, int],
    impute: str = "zero",
) -> tuple[torch.Tensor, pd.DataFrame]:
    """
    Mean-pool protein embeddings over a drug's known target genes.
    Returns (drug_level_tensor, coverage_df).
    """
    dim      = prot_emb.shape[1]
    n_drugs  = len(drug_to_idx)
    fallback = torch.zeros(dim) if impute == "zero" else prot_emb.mean(dim=0)

    drug_targets: dict[str, list[str]] = defaultdict(list)
    for _, row in targets_df.iterrows():
        drug_targets[str(row["STITCH"])].append(str(row["Gene"]))

    out      = torch.zeros(n_drugs, dim, dtype=prot_emb.dtype)
    coverage = []
    for drug, idx in drug_to_idx.items():
        genes   = drug_targets.get(drug, [])
        ok      = [g for g in genes if g in gene_to_row]
        if ok:
            rows        = [gene_to_row[g] for g in ok]
            out[idx]    = prot_emb[rows].mean(dim=0)
        else:
            out[idx]    = fallback
        coverage.append({
            "drug":                     drug,
            "n_targets_in_file":        len(genes),
            "n_targets_with_embedding": len(ok),
            "has_target_signal":        len(ok) > 0,
        })

    cov_df    = pd.DataFrame(coverage)
    n_covered = cov_df["has_target_signal"].sum()
    logger.info(
        f"Drug-level protein aggregation: {n_covered}/{n_drugs} drugs have target signal"
    )
    return out, cov_df


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Save raw (un-projected) pretrained features for ablation_track1_mlp.py"
    )
    p.add_argument("--raw_dir",      default="data/raw",
                   help="Directory containing bio-decagon-*.csv and protein_sequences.json")
    p.add_argument("--combo",        default=None,
                   help="Path to bio-decagon-combo.csv. Defaults to raw_dir/bio-decagon-combo.csv")
    p.add_argument("--targets",      default=None,
                   help="Path to bio-decagon-targets.csv. Defaults to raw_dir/bio-decagon-targets.csv")
    p.add_argument("--cache_dir",    default="data/cache",
                   help="Directory for intermediate heavy caches (ESM-2 protein tensors)")
    p.add_argument("--output_dir",   default="data/cache/ablation_mlp",
                   help="Directory for final drug-level raw feature tensors")
    p.add_argument("--min_edges",    type=int, default=500,
                   help="Lloyd threshold — filters SEs before building drug vocabulary")
    p.add_argument("--mono_method",  choices=["tfidf", "cur"], default="tfidf")
    p.add_argument("--mono_components", type=int, default=128,
                   help="SVD / CUR components for mono encoding (output dim of mono tensor)")
    p.add_argument("--chemberta_model", default="seyonec/ChemBERTa-zinc-base-v1")
    p.add_argument("--esm2_model",   default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--n_hops",       type=int, default=1,
                   help="PPI neighbourhood hops for aggregation")
    p.add_argument("--target_impute", choices=["zero", "mean"], default="zero",
                   help="Imputation for drugs with no target annotations")
    p.add_argument("--device",       default="cpu")
    p.add_argument("--skip_drugs",   action="store_true",
                   help="Skip ChemBERTa and mono encoding (re-use existing .pt files)")
    p.add_argument("--skip_proteins", action="store_true",
                   help="Skip ESM-2 encoding and PPI aggregation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir    = Path(args.raw_dir)
    cache_dir  = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    combo_path   = Path(args.combo)   if args.combo   else raw_dir / "bio-decagon-combo.csv"
    targets_path = Path(args.targets) if args.targets else raw_dir / "bio-decagon-targets.csv"

    # ── Build entity indexes ──────────────────────────────────────────────────
    logger.info("Building sorted drug and gene indexes ...")
    drug_to_idx = _build_sorted_drug_to_idx(str(combo_path), args.min_edges)
    gene_to_row = _build_sorted_gene_to_row(str(targets_path))
    n_drugs     = len(drug_to_idx)

    # Load Decagon data (needed for SMILES strings, PPI edges, mono matrix)
    from src.data import load_decagon
    logger.info("Loading Decagon data (SMILES, PPI, mono) ...")
    data = load_decagon(raw_dir=str(raw_dir))

    # Build sorted mono matrix aligned to drug_to_idx
    reordered_mono = _build_reordered_mono_matrix(
        data.mono_se_matrix, data.drug_to_id, drug_to_idx
    ) if data.mono_se_matrix is not None else None

    esm_tag  = args.esm2_model.split("/")[-1]

    # ── Drug features ─────────────────────────────────────────────────────────
    if not args.skip_drugs:
        from src.encoders.drug_encoder import (
            ChemBERTaEncoder, TFIDFMonoEncoder, CURMonoEncoder,
        )

        # ChemBERTa — pass drug_to_idx so encoder iterates in sorted order
        cb_cache = cache_dir / f"drug_raw_chemberta_{n_drugs}_{args.chemberta_model.replace('/', '_')}.pt"
        if cb_cache.exists():
            logger.info(f"  ChemBERTa cache hit: {cb_cache}")
            drug_raw_cb = torch.load(cb_cache)
        else:
            logger.info("  Encoding SMILES with ChemBERTa ...")
            encoder = ChemBERTaEncoder(
                model_name=args.chemberta_model, device=args.device, freeze=True
            )
            drug_raw_cb = encoder.encode_drugs(drug_to_idx, data.drug_smiles)
            torch.save(drug_raw_cb, cb_cache)
            logger.info(f"  Cached ChemBERTa features → {cb_cache}")

        out_cb = output_dir / "drug_raw_chemberta.pt"
        torch.save(drug_raw_cb.float(), out_cb)
        logger.info(f"Saved {out_cb}  shape={tuple(drug_raw_cb.shape)}")

        # Mono (TF-IDF / CUR)
        if reordered_mono is None:
            logger.warning("No mono_se_matrix in loaded data; skipping mono tensor.")
        else:
            mono_cache = cache_dir / (
                f"drug_raw_mono_{n_drugs}_{args.mono_method}_c{args.mono_components}.pt"
            )
            if mono_cache.exists():
                logger.info(f"  Mono cache hit: {mono_cache}")
                drug_raw_mono = torch.load(mono_cache)
            else:
                logger.info(f"  Encoding mono SEs with {args.mono_method.upper()} ...")
                if args.mono_method == "tfidf":
                    enc = TFIDFMonoEncoder(n_components=args.mono_components)
                else:
                    enc = CURMonoEncoder(n_components=args.mono_components)
                mono_feats_np = enc.fit_transform(reordered_mono)
                drug_raw_mono = torch.tensor(mono_feats_np, dtype=torch.float32)
                torch.save(drug_raw_mono, mono_cache)
                logger.info(f"  Cached mono features → {mono_cache}")

            out_mono = output_dir / "drug_raw_mono.pt"
            torch.save(drug_raw_mono.float(), out_mono)
            logger.info(f"Saved {out_mono}  shape={tuple(drug_raw_mono.shape)}")
    else:
        logger.info("Skipping drug encoding (--skip_drugs).")

    # ── Protein features → drug-level aggregation ─────────────────────────────
    if not args.skip_proteins:
        from src.encoders.protein_encoder import ESM2Encoder, PPINeighbourhoodAggregator, ESM2_DIMS

        protein_sequences: dict[str, str] = {}
        seq_file = raw_dir / "protein_sequences.json"
        if seq_file.exists():
            with open(seq_file, encoding="utf-8") as f:
                protein_sequences = json.load(f)
            logger.info(f"Loaded {len(protein_sequences)} protein sequences")
        else:
            logger.warning(
                f"No protein_sequences.json in {raw_dir}; "
                "proteins without sequences use ESM-2 fallback ('M')."
            )

        esm_dim = ESM2_DIMS.get(args.esm2_model, 320)

        # ESM-2 raw per-protein
        esm_raw_cache = cache_dir / f"protein_raw_esm2_{esm_tag}.pt"
        if esm_raw_cache.exists():
            logger.info(f"  ESM-2 cache hit: {esm_raw_cache}")
            prot_raw_esm = torch.load(esm_raw_cache)
        else:
            logger.info(f"  Encoding protein sequences with ESM-2 ({args.esm2_model}) ...")
            esm_encoder = ESM2Encoder(
                model_name=args.esm2_model,
                device=args.device,
                freeze=True,
            )
            # Pass gene_to_row as protein_to_id → tensor rows are sorted-gene order
            prot_raw_esm = esm_encoder.encode_proteins(gene_to_row, protein_sequences)
            torch.save(prot_raw_esm, esm_raw_cache)
            logger.info(f"  Cached ESM-2 protein tensor → {esm_raw_cache}")

        # PPI-aggregated per-protein
        ppi_raw_cache = cache_dir / f"protein_raw_ppi_{esm_tag}_h{args.n_hops}.pt"
        if ppi_raw_cache.exists():
            logger.info(f"  PPI cache hit: {ppi_raw_cache}")
            prot_raw_ppi = torch.load(ppi_raw_cache)
        else:
            logger.info(f"  PPI neighbourhood aggregation (n_hops={args.n_hops}) ...")
            agg = PPINeighbourhoodAggregator(n_hops=args.n_hops)
            # gene_to_row maps gene strings to sorted row indices; ppi_edges use gene strings
            prot_raw_ppi = agg.aggregate(prot_raw_esm, data.ppi_edges, gene_to_row)
            torch.save(prot_raw_ppi, ppi_raw_cache)
            logger.info(f"  Cached PPI protein tensor → {ppi_raw_cache}")

        # Save protein-level tensors (useful for future re-aggregation with different targets)
        torch.save(prot_raw_esm.float(), output_dir / "protein_raw_esm2.pt")
        torch.save(prot_raw_ppi.float(), output_dir / "protein_raw_ppi.pt")
        logger.info(
            f"Saved protein-level tensors  esm={tuple(prot_raw_esm.shape)}  "
            f"ppi={tuple(prot_raw_ppi.shape)}"
        )

        # Drug-level aggregation: mean over target proteins
        logger.info("Aggregating ESM-2 embeddings to drug level (via bio-decagon-targets) ...")
        targets_df = pd.read_csv(targets_path)
        targets_df["STITCH"] = targets_df["STITCH"].astype(str)
        targets_df["Gene"]   = targets_df["Gene"].astype(str)

        drug_esm_via, cov_esm = _aggregate_protein_to_drug(
            prot_raw_esm, targets_df, drug_to_idx, gene_to_row, args.target_impute
        )
        drug_ppi_via, cov_ppi = _aggregate_protein_to_drug(
            prot_raw_ppi, targets_df, drug_to_idx, gene_to_row, args.target_impute
        )

        out_esm = output_dir / "drug_raw_esm2_via_targets.pt"
        out_ppi = output_dir / "drug_raw_ppi_via_targets.pt"
        torch.save(drug_esm_via.float(), out_esm)
        torch.save(drug_ppi_via.float(), out_ppi)
        cov_esm.to_csv(output_dir / "target_coverage_esm2.csv",  index=False)
        cov_ppi.to_csv(output_dir / "target_coverage_ppi.csv",   index=False)
        logger.info(f"Saved {out_esm}  shape={tuple(drug_esm_via.shape)}")
        logger.info(f"Saved {out_ppi}  shape={tuple(drug_ppi_via.shape)}")
    else:
        logger.info("Skipping protein encoding (--skip_proteins).")
        esm_dim = None

    # ── Save index and metadata ───────────────────────────────────────────────
    idx_path = output_dir / "drug_to_idx.json"
    with open(idx_path, "w") as f:
        json.dump(drug_to_idx, f)
    logger.info(f"Saved {idx_path}")

    # Collect actual tensor shapes from disk
    def _shape(pt_path: Path) -> list[int] | None:
        if pt_path.exists():
            t = torch.load(pt_path, map_location="cpu")
            return list(t.shape)
        return None

    meta = {
        "n_drugs":            n_drugs,
        "min_edges":          args.min_edges,
        "chemberta_model":    args.chemberta_model,
        "esm2_model":         args.esm2_model,
        "esm_dim":            esm_dim,
        "n_hops":             args.n_hops,
        "mono_method":        args.mono_method,
        "mono_components":    args.mono_components,
        "target_impute":      args.target_impute,
        "shapes": {
            "drug_raw_chemberta":          _shape(output_dir / "drug_raw_chemberta.pt"),
            "drug_raw_mono":               _shape(output_dir / "drug_raw_mono.pt"),
            "drug_raw_esm2_via_targets":   _shape(output_dir / "drug_raw_esm2_via_targets.pt"),
            "drug_raw_ppi_via_targets":    _shape(output_dir / "drug_raw_ppi_via_targets.pt"),
            "protein_raw_esm2":            _shape(output_dir / "protein_raw_esm2.pt"),
            "protein_raw_ppi":             _shape(output_dir / "protein_raw_ppi.pt"),
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved metadata.json")

    logger.info("=" * 60)
    logger.info("Done.  Pass these paths to ablation/ablation_track1_mlp.py:")
    logger.info(f"  --drug_raw_chemberta  {output_dir / 'drug_raw_chemberta.pt'}")
    logger.info(f"  --drug_raw_mono       {output_dir / 'drug_raw_mono.pt'}")
    logger.info(f"  --drug_raw_esm2       {output_dir / 'drug_raw_esm2_via_targets.pt'}")
    logger.info(f"  --drug_raw_ppi        {output_dir / 'drug_raw_ppi_via_targets.pt'}")
    logger.info(f"  --drug_to_idx         {output_dir / 'drug_to_idx.json'}")


if __name__ == "__main__":
    main()
