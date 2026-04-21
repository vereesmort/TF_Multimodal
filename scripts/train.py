#!/usr/bin/env python
"""
Training script for TF-Decagon Improved.

Implements the exact Decagon / Lloyd et al. data split:
    - Polypharmacy edges split PER SIDE-EFFECT TYPE: 80/10/10
    - PPI and drug-target edges go entirely into training
    - Val/test edges are removed from the KGE training graph

After training the following artifacts are saved to --output_dir:
    model.pt              final model weights
    best_model.pt         weights at the epoch with lowest training loss
    train_tf.pt           PyKEEN TriplesFactory used during training
    entity_to_id.json     entity label → integer ID mapping
    relation_to_id.json   relation label → integer ID mapping
    test_edges.tsv        held-out test polypharmacy triples (TSV, no header)
    config.yaml           fully resolved run configuration

Next steps after training:
    python scripts/generate_negatives.py --dataset_dir <output_dir>
    python scripts/evaluate.py --checkpoint <output_dir>/best_model.pt \\
        --dataset_dir <output_dir> --out_dir <output_dir>/results/

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train improved TF-Decagon with semantic entity initialization"
    )
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--raw_dir", type=str, default="data/raw")
    p.add_argument("--cache_dir", type=str, default="data/cache")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--mono_method", choices=["tfidf", "cur"], default=None)
    p.add_argument("--mono_components", type=int, default=None,
                   help="Dimensionality of monopharmacy SE features (default: 128)")
    p.add_argument("--chemberta_model", type=str, default=None,
                   help="HuggingFace ChemBERTa model ID for drug SMILES encoding "
                        "(default: seyonec/ChemBERTa-zinc-base-v1)")
    p.add_argument("--esm2_model", type=str, default=None,
                   help="HuggingFace ESM-2 model ID for protein sequence encoding "
                        "(default: facebook/esm2_t6_8M_UR50D). "
                        "Options: esm2_t6_8M_UR50D | esm2_t12_35M_UR50D | "
                        "esm2_t30_150M_UR50D | esm2_t33_650M_UR50D")
    p.add_argument("--interaction", choices=["SimplE", "ComplEx", "DistMult"], default=None)
    p.add_argument("--embedding_dim", type=int, default=None)
    p.add_argument("--n_epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ckpt_frequency", type=int, default=30,
                   help="Save a PyKEEN checkpoint every N minutes (default: 30). "
                        "Set to 0 to checkpoint after every epoch.")
    p.add_argument("--patience", type=int, default=None,
                   help="Early stopping patience in epochs on validation loss. "
                        "Omit to disable early stopping.")
    p.add_argument("--skip_download", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    if Path(path).exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    cfg.setdefault("raw_dir", args.raw_dir)
    cfg.setdefault("cache_dir", args.cache_dir)
    cfg.setdefault("output_dir", args.output_dir)
    cfg.setdefault("mono_method", "tfidf")
    cfg.setdefault("interaction", "SimplE")
    cfg.setdefault("embedding_dim", 256)
    cfg.setdefault("n_epochs", 200)
    cfg.setdefault("lr", 0.001)
    cfg.setdefault("batch_size", 512)
    cfg.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
    cfg.setdefault("mono_components", 128)
    cfg.setdefault("chemberta_model", "seyonec/ChemBERTa-zinc-base-v1")
    cfg.setdefault("esm2_model", "facebook/esm2_t6_8M_UR50D")
    cfg.setdefault("n_hops", 1)
    cfg.setdefault("val_frac", 0.10)
    cfg.setdefault("test_frac", 0.10)

    for key in ["mono_method", "mono_components", "chemberta_model", "esm2_model",
                "interaction", "embedding_dim", "n_epochs", "lr", "batch_size", "device"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Config: {cfg}")

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    from src.data import (
        download_decagon, load_decagon,
        split_polypharmacy_edges,
    )

    if not args.skip_download:
        download_decagon(raw_dir=cfg["raw_dir"])

    logger.info("Loading Decagon data ...")
    data = load_decagon(raw_dir=cfg["raw_dir"])

    # ------------------------------------------------------------------ #
    # 2. Correct per-SE-type stratified split
    # ------------------------------------------------------------------ #
    logger.info("Splitting polypharmacy edges per SE type (80/10/10) ...")
    train_poly, val_poly, test_poly = split_polypharmacy_edges(
        poly_triples=data.poly_triples,
        val_frac=cfg["val_frac"],
        test_frac=cfg["test_frac"],
        random_state=42,
    )

    # ------------------------------------------------------------------ #
    # 3. Record labeled test edges (saved to disk after training)
    # ------------------------------------------------------------------ #
    test_labeled = [(f"drug:{h}", r, f"drug:{t}") for h, r, t in test_poly]

    # ------------------------------------------------------------------ #
    # 4. Build PyKEEN training triple factory
    #    Val/test polypharmacy edges are EXCLUDED from the training graph
    # ------------------------------------------------------------------ #
    import numpy as np
    from pykeen.triples import TriplesFactory

    entity_to_id: dict = {}
    for drug, idx in data.drug_to_id.items():
        entity_to_id[f"drug:{drug}"] = idx
    offset = len(data.drug_to_id)
    for prot, idx in data.protein_to_id.items():
        entity_to_id[f"protein:{prot}"] = offset + idx

    relation_to_id: dict = {f"SE:{se}": idx for se, idx in data.se_pair_to_id.items()}
    relation_to_id["drug_targets"] = len(relation_to_id)
    relation_to_id["ppi"] = len(relation_to_id)

    train_triples_raw = (
        [(f"drug:{h}", r, f"drug:{t}") for h, r, t in train_poly]
        + [(f"drug:{d}", "drug_targets", f"protein:{p}") for d, p in data.drug_target_edges]
        + [(f"protein:{p1}", "ppi", f"protein:{p2}") for p1, p2 in data.ppi_edges]
    )

    train_tf = TriplesFactory.from_labeled_triples(
        triples=np.array(train_triples_raw),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    logger.info(
        f"Training graph: {len(train_triples_raw)} triples "
        f"(poly: {len(train_poly)}, structural: "
        f"{len(data.drug_target_edges) + len(data.ppi_edges)})"
    )

    # Validation factory — polypharmacy edges only, same entity/relation mapping
    val_triples_raw = [(f"drug:{h}", r, f"drug:{t}") for h, r, t in val_poly]
    val_tf = TriplesFactory.from_labeled_triples(
        triples=np.array(val_triples_raw),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
    )
    logger.info(f"Validation graph: {len(val_triples_raw)} polypharmacy triples")

    # ------------------------------------------------------------------ #
    # 5. Entity initialization
    # ------------------------------------------------------------------ #
    from src.model import build_drug_init_tensor, build_protein_init_tensor, build_pykeen_model

    drug_init = build_drug_init_tensor(
        drug_to_id=data.drug_to_id,
        drug_smiles=data.drug_smiles,
        mono_se_matrix=data.mono_se_matrix,
        embedding_dim=cfg["embedding_dim"],
        mono_method=cfg["mono_method"],
        mono_components=cfg["mono_components"],
        device=cfg["device"],
        cache_dir=cfg["cache_dir"],
    )

    protein_init = build_protein_init_tensor(
        protein_to_id=data.protein_to_id,
        ppi_edges=data.ppi_edges,
        drug_targets=data.drug_targets,
        drug_to_id=data.drug_to_id,
        protein_sequences={},  # provide FASTA-derived sequences here
        embedding_dim=cfg["embedding_dim"],
        esm2_model=cfg["esm2_model"],
        n_hops=cfg["n_hops"],
        device=cfg["device"],
        cache_dir=cfg["cache_dir"],
    )

    # ------------------------------------------------------------------ #
    # 6. Build and train model
    # ------------------------------------------------------------------ #
    model = build_pykeen_model(
        train_tf=train_tf,
        entity_to_id=entity_to_id,
        drug_init=drug_init,
        protein_init=protein_init,
        drug_to_id=data.drug_to_id,
        protein_to_id=data.protein_to_id,
        interaction=cfg["interaction"],
        embedding_dim=cfg["embedding_dim"],
    )
    model = model.to(cfg["device"])

    from pykeen.training import SLCWATrainingLoop
    from pykeen.stoppers import EarlyStopper
    from torch.optim import Adam

    optimizer = Adam(model.parameters(), lr=cfg["lr"])

    best_model_path = out_dir / "best_model.pt"

    stopper = None
    if args.patience:
        stopper = EarlyStopper(
            model=model,
            training_triples_factory=train_tf,
            validation_triples_factory=val_tf,
            frequency=1,               # evaluate every epoch
            patience=args.patience,    # stop if no improvement for N evals
            metric="hits@10",          # monitored metric
            relative_delta=0.002,      # min improvement to count (0.2%)
            result_tracker=None,
        )
        logger.info(
            f"Early stopping enabled: patience={args.patience} epochs, "
            f"metric=hits@10, min_delta=0.2%"
        )

    training_loop = SLCWATrainingLoop(
        model=model,
        triples_factory=train_tf,
        optimizer=optimizer,
    )

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"{cfg['interaction']}_{cfg['mono_method']}_dim{cfg['embedding_dim']}.pt"
    n_epochs  = cfg["n_epochs"]

    # PyKEEN checkpoints store absolute epoch counts, so num_epochs must be
    # the TOTAL target epochs for the full run — not a per-block step size.
    # A chunked while-loop is incompatible with PyKEEN checkpointing because
    # each call would resume from the checkpoint and immediately exit
    # (already at the requested epoch count).
    logger.info(
        f"Training {cfg['interaction']} for {n_epochs} epochs | "
        f"checkpoints every {args.ckpt_frequency} min → {ckpt_dir / ckpt_name} | "
        f"best model → {best_model_path}"
    )

    losses = training_loop.train(
        triples_factory=train_tf,
        num_epochs=n_epochs,
        batch_size=cfg["batch_size"],
        checkpoint_name=ckpt_name,
        checkpoint_directory=ckpt_dir,
        checkpoint_frequency=args.ckpt_frequency,
        stopper=stopper,
    )

    # Save the final model weights.
    # If EarlyStopper triggered, PyKEEN restores the best weights before
    # returning, so model.state_dict() is already the best checkpoint.
    torch.save(model.state_dict(), out_dir / "model.pt")
    torch.save(model.state_dict(), best_model_path)
    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    logger.info(
        f"Training complete — mean loss over last epoch block: {mean_loss:.4f}"
    )
    logger.info(f"Model saved → {out_dir / 'model.pt'}  |  {best_model_path}")

    # ------------------------------------------------------------------ #
    # 7. Save artifacts for standalone evaluation scripts
    # ------------------------------------------------------------------ #
    import json
    import pandas as pd
    import yaml

    (out_dir / "entity_to_id.json").write_text(
        json.dumps(entity_to_id, indent=2)
    )
    (out_dir / "relation_to_id.json").write_text(
        json.dumps(relation_to_id, indent=2)
    )
    torch.save(train_tf, out_dir / "train_tf.pt")

    pd.DataFrame(test_labeled).to_csv(
        out_dir / "test_edges.tsv", header=False, index=False, sep="\t"
    )
    (out_dir / "config.yaml").write_text(yaml.dump(cfg))

    logger.info(
        f"Artifacts saved to {out_dir}/\n"
        f"  Next steps:\n"
        f"    python scripts/generate_negatives.py --dataset_dir {out_dir}\n"
        f"    python scripts/evaluate.py "
        f"--checkpoint {best_model_path} "
        f"--dataset_dir {out_dir} "
        f"--out_dir {out_dir / 'results'}"
    )


if __name__ == "__main__":
    main()
