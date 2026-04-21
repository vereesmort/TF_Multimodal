#!/usr/bin/env python
"""
Precompute and cache drug + protein embeddings before training.

This is recommended when running training multiple times (hyperparameter
sweeps, ablations) because encoding with ChemBERTa and ESM-2 is the
most expensive step and does not change between runs.

Usage:
    python scripts/precompute_embeddings.py \
        --raw_dir data/raw \
        --cache_dir data/cache \
        --embedding_dim 256 \
        --mono_method tfidf \
        --esm2_model facebook/esm2_t6_8M_UR50D \
        --device cpu
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("precompute")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--mono_method", choices=["tfidf", "cur"], default="tfidf")
    parser.add_argument("--mono_components", type=int, default=128)
    parser.add_argument("--esm2_model", type=str, default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--n_hops", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    from src.data import download_decagon, load_decagon
    from src.model import build_drug_init_tensor, build_protein_init_tensor

    # Download if needed
    download_decagon(raw_dir=args.raw_dir)

    # Load data
    logger.info("Loading Decagon data ...")
    data = load_decagon(raw_dir=args.raw_dir)

    # Drug embeddings
    logger.info("Precomputing drug embeddings ...")
    drug_init = build_drug_init_tensor(
        drug_to_id=data.drug_to_id,
        drug_smiles=data.drug_smiles,
        mono_se_matrix=data.mono_se_matrix,
        embedding_dim=args.embedding_dim,
        mono_method=args.mono_method,
        mono_components=args.mono_components,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    logger.info(f"Drug init tensor: {drug_init.shape}")

    # Protein embeddings
    logger.info("Precomputing protein embeddings ...")
    protein_init = build_protein_init_tensor(
        protein_to_id=data.protein_to_id,
        ppi_edges=data.ppi_edges,
        drug_targets=data.drug_targets,
        drug_to_id=data.drug_to_id,
        protein_sequences={},  # Provide your own FASTA-derived sequences
        embedding_dim=args.embedding_dim,
        esm2_model=args.esm2_model,
        n_hops=args.n_hops,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    logger.info(f"Protein init tensor: {protein_init.shape}")
    logger.info(f"All embeddings cached to: {args.cache_dir}")


if __name__ == "__main__":
    main()
