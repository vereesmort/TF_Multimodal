#!/usr/bin/env python
"""
End-to-end training script for TF-Decagon Improved.

Implements the exact Decagon / Lloyd et al. data split:
    - Polypharmacy edges split PER SIDE-EFFECT TYPE: 80/10/10
    - PPI and drug-target edges go entirely into training
    - Val/test edges are removed from the KGE training graph

Runs both evaluation protocols after training:
    Protocol 1 (--eval_protocol false_edge):
        1 false edge per positive, directly comparable to published results.
    Protocol 2 (--eval_protocol sampled_negatives):
        N sampled negatives per positive, faster for ablations.

Usage:
    python scripts/train.py --config configs/default.yaml

    # Protocol 1 only (comparable to Lloyd et al. paper numbers)
    python scripts/train.py --eval_protocol false_edge

    # Fast ablation with sampled negatives
    python scripts/train.py --eval_protocol sampled_negatives --n_negatives 100

    # Run both protocols (default)
    python scripts/train.py --eval_protocol both
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
    p.add_argument("--interaction", choices=["SimplE", "ComplEx", "DistMult"], default=None)
    p.add_argument("--embedding_dim", type=int, default=None)
    p.add_argument("--n_epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--eval_protocol",
        choices=["false_edge", "sampled_negatives", "both"],
        default="both",
        help=(
            "false_edge: Decagon/Lloyd protocol (1 false per positive) — "
            "use for comparisons with published results. "
            "sampled_negatives: N negatives per positive — faster, for ablations. "
            "both: run both (default)."
        ),
    )
    p.add_argument(
        "--n_negatives",
        type=int,
        default=100,
        help="Negatives per positive for sampled_negatives protocol.",
    )
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
    cfg.setdefault("esm2_model", "facebook/esm2_t6_8M_UR50D")
    cfg.setdefault("n_hops", 1)
    cfg.setdefault("val_frac", 0.10)
    cfg.setdefault("test_frac", 0.10)

    for key in ["mono_method", "interaction", "embedding_dim",
                "n_epochs", "lr", "batch_size", "device"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    eval_protocol = args.eval_protocol
    n_negatives = args.n_negatives

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Config: {cfg}")

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    from src.data import (
        download_decagon, load_decagon,
        split_polypharmacy_edges, build_true_edge_set, generate_false_edges,
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

    # True-edge lookup over ALL known triples (used for filtering negatives)
    true_edge_lookup = build_true_edge_set(data.poly_triples)
    all_drug_labels = [f"drug:{d}" for d in data.drug_to_id]

    # ------------------------------------------------------------------ #
    # 3. Pre-generate false test edges (Protocol 1)
    # ------------------------------------------------------------------ #
    false_test_labeled = None
    test_labeled = [(f"drug:{h}", r, f"drug:{t}") for h, r, t in test_poly]
    true_lookup_labeled = {
        r: {(f"drug:{h}", f"drug:{t}") for h, t in pairs}
        for r, pairs in true_edge_lookup.items()
    }

    if eval_protocol in ("false_edge", "both"):
        logger.info("Generating false test edges (Protocol 1) ...")
        false_test_labeled = generate_false_edges(
            positive_edges=test_labeled,
            all_drugs=all_drug_labels,
            true_edge_lookup=true_lookup_labeled,
            n_false_per_positive=1,
            random_state=42,
        )
        logger.info(
            f"False test edges: {len(false_test_labeled)} "
            f"(must equal {len(test_labeled)})"
        )

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
    from torch.optim import Adam

    optimizer = Adam(model.parameters(), lr=cfg["lr"])
    training_loop = SLCWATrainingLoop(
        model=model, triples_factory=train_tf, optimizer=optimizer,
    )
    logger.info(f"Training {cfg['interaction']} for {cfg['n_epochs']} epochs ...")
    training_loop.train(
        triples_factory=train_tf,
        num_epochs=cfg["n_epochs"],
        batch_size=cfg["batch_size"],
    )

    torch.save(model.state_dict(), out_dir / "model.pt")
    logger.info(f"Model saved to {out_dir / 'model.pt'}")

    # ------------------------------------------------------------------ #
    # 7. Evaluation
    # ------------------------------------------------------------------ #
    from src.evaluation.protocols import (
        evaluate_false_edge_protocol,
        evaluate_sampled_negatives_protocol,
        evaluate_stratified,
        summarise,
        summarise_stratified,
    )

    model_tag = f"{cfg['interaction']}-{cfg['mono_method'].upper()}"

    # Protocol 1: False-edge (Decagon / Lloyd et al.)
    if eval_protocol in ("false_edge", "both"):
        logger.info("Evaluating — Protocol 1: false-edge (Decagon protocol) ...")
        df_fe = evaluate_false_edge_protocol(
            model=model,
            test_triples=test_labeled,
            false_triples=false_test_labeled,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            device=cfg["device"],
        )
        summarise(df_fe, model_name=model_tag, protocol="false_edge")
        df_fe.to_csv(out_dir / "eval_false_edge.csv", index=False)

        df_strat_fe = evaluate_stratified(
            results_df=df_fe,
            test_triples=test_labeled,
            false_triples=false_test_labeled,
            drug_targets=data.drug_targets,
            protein_to_id=data.protein_to_id,
            model=model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            protocol="false_edge",
            all_drug_labels=all_drug_labels,
            true_edge_lookup=true_lookup_labeled,
            device=cfg["device"],
        )
        summarise_stratified(df_strat_fe, model_name=model_tag, protocol="false_edge")
        df_strat_fe.to_csv(out_dir / "eval_false_edge_stratified.csv", index=False)

    # Protocol 2: Sampled negatives
    if eval_protocol in ("sampled_negatives", "both"):
        logger.info(f"Evaluating — Protocol 2: sampled negatives (N={n_negatives}) ...")
        df_sn = evaluate_sampled_negatives_protocol(
            model=model,
            test_triples=test_labeled,
            all_drug_labels=all_drug_labels,
            true_edge_lookup=true_lookup_labeled,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            n_negatives=n_negatives,
            device=cfg["device"],
        )
        summarise(df_sn, model_name=model_tag, protocol=f"sampled_neg_{n_negatives}")
        df_sn.to_csv(out_dir / "eval_sampled_negatives.csv", index=False)

        df_strat_sn = evaluate_stratified(
            results_df=df_sn,
            test_triples=test_labeled,
            false_triples=None,
            drug_targets=data.drug_targets,
            protein_to_id=data.protein_to_id,
            model=model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            protocol="sampled_negatives",
            all_drug_labels=all_drug_labels,
            true_edge_lookup=true_lookup_labeled,
            n_negatives=n_negatives,
            device=cfg["device"],
        )
        summarise_stratified(
            df_strat_sn, model_name=model_tag,
            protocol=f"sampled_neg_{n_negatives}"
        )
        df_strat_sn.to_csv(out_dir / "eval_sampled_negatives_stratified.csv", index=False)

    logger.info(f"Done. All outputs in: {out_dir}")


if __name__ == "__main__":
    main()
