#!/usr/bin/env python
"""
Precompute per-modality tensors for ablation_track1_ml.py.

Unlike scripts/precompute_embeddings.py (fused ChemBERTa+mono drugs and fused
ESM-2+PPI proteins for KGE training), this script writes *separate* matrices so
you can pass them to:

  --drug_emb_chemberta   ChemBERTa → 256-d (no monopharmacy SE branch)
  --drug_emb_mono        TF-IDF/CUR mono → 256-d (no ChemBERTa)
  --prot_emb_esm2        ESM-2 sequence branch only → 256-d
  --prot_emb_ppi         PPI neighbourhood branch only → 256-d

Intermediate heavy steps are cached under --cache_dir; final files are written
to --output_dir with stable names for the ablation CLI.

Prerequisite:
    bash data/raw/download_unpack.sh

Example:
    python scripts/precompute_embeddings_ablation.py \\
        --raw_dir data/raw \\
        --cache_dir data/cache \\
        --output_dir data/cache/ablation \\
        --embedding_dim 256 \\
        --esm2_model facebook/esm2_t30_150M_UR50D \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("precompute_ablation")


def main():
    p = argparse.ArgumentParser(description="Ablation embedding tensors for ablation_track1_ml.py")
    p.add_argument("--raw_dir", type=str, default="data/raw")
    p.add_argument("--cache_dir", type=str, default="data/cache",
                   help="Intermediate caches (ChemBERTa / mono / ESM-2 compute)")
    p.add_argument("--output_dir", type=str, default="data/cache/ablation",
                   help="Directory for final drug_emb_*.pt and protein_emb_*.pt")
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--mono_method", choices=["tfidf", "cur"], default="tfidf")
    p.add_argument("--mono_components", type=int, default=128)
    p.add_argument("--chemberta_model", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    p.add_argument("--esm2_model", type=str, default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--n_hops", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--esm_projection_seed", type=int, default=42)
    p.add_argument("--ppi_projection_seed", type=int, default=43)
    p.add_argument("--mono_projection_seed", type=int, default=42)
    p.add_argument("--chemberta_projection_seed", type=int, default=42)
    p.add_argument("--skip_drugs", action="store_true")
    p.add_argument("--skip_proteins", action="store_true")
    args = p.parse_args()

    from src.data import load_decagon
    from src.model import (
        build_drug_chemberta_only_tensor,
        build_drug_mono_only_tensor,
        build_protein_esm_ppi_ablation_tensors,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Decagon data ...")
    data = load_decagon(raw_dir=args.raw_dir)

    D = args.embedding_dim

    if not args.skip_drugs:
        logger.info("Drug: ChemBERTa-only ...")
        drug_cb = build_drug_chemberta_only_tensor(
            drug_to_id=data.drug_to_id,
            drug_smiles=data.drug_smiles,
            embedding_dim=D,
            chemberta_model=args.chemberta_model,
            device=args.device,
            cache_dir=args.cache_dir,
            projection_seed=args.chemberta_projection_seed,
        )
        path_cb = out / f"drug_emb_chemberta_{D}.pt"
        torch.save(drug_cb, path_cb)
        logger.info(f"Saved {path_cb} shape {tuple(drug_cb.shape)}")

        logger.info("Drug: mono-only ...")
        drug_mono = build_drug_mono_only_tensor(
            mono_se_matrix=data.mono_se_matrix,
            embedding_dim=D,
            mono_method=args.mono_method,
            mono_components=args.mono_components,
            cache_dir=args.cache_dir,
            projection_seed=args.mono_projection_seed,
        )
        path_mono = out / f"drug_emb_mono_{D}.pt"
        torch.save(drug_mono, path_mono)
        logger.info(f"Saved {path_mono} shape {tuple(drug_mono.shape)}")

    if not args.skip_proteins:
        protein_seq_file = Path(args.raw_dir) / "protein_sequences.json"
        if protein_seq_file.exists():
            with open(protein_seq_file, encoding="utf-8") as f:
                protein_sequences = json.load(f)
            logger.info(f"Loaded {len(protein_sequences)} protein sequences from {protein_seq_file}")
        else:
            protein_sequences = {}
            logger.warning(
                f"No {protein_seq_file}; proteins without sequences use ESM-2 fallback (see precompute_embeddings.py doc)."
            )

        esm_tag = args.esm2_model.split("/")[-1]
        logger.info("Protein: ESM-2 self vs PPI-neighbour branches ...")
        esm_only, ppi_only = build_protein_esm_ppi_ablation_tensors(
            protein_to_id=data.protein_to_id,
            ppi_edges=data.ppi_edges,
            protein_sequences=protein_sequences,
            embedding_dim=D,
            esm2_model=args.esm2_model,
            n_hops=args.n_hops,
            device=args.device,
            cache_dir=args.cache_dir,
            esm_projection_seed=args.esm_projection_seed,
            ppi_projection_seed=args.ppi_projection_seed,
        )
        path_esm = out / f"protein_emb_esm2_only_dim{D}_{esm_tag}.pt"
        path_ppi = out / f"protein_emb_ppi_neighbour_dim{D}_{esm_tag}.pt"

        torch.save(esm_only, path_esm)
        torch.save(ppi_only, path_ppi)
        logger.info(f"Saved {path_esm} shape {tuple(esm_only.shape)}")
        logger.info(f"Saved {path_ppi} shape {tuple(ppi_only.shape)}")

    logger.info("Done. Pass these paths to ablation/ablation_track1_ml.py")


if __name__ == "__main__":
    main()
