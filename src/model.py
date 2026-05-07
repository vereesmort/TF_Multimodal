"""
Model builder.

Assembles a PyKEEN SimplE (or ComplEx) model with semantically grounded
entity embedding initialization:

  - Drug entities initialized from:
      ChemBERTa (molecular structure) + TF-IDF/CUR (monopharmacy SEs)

  - Protein entities initialized from:
      ESM-2 (sequence) + PPI neighbourhood aggregation

The initialization vectors are computed once, projected to embedding_dim,
and passed to PyKEEN via PretrainedInitializer.  Embeddings are then
fine-tuned jointly with the TF decoder during training.

Usage:
    See scripts/train.py for end-to-end training pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def build_drug_init_tensor(
    drug_to_id: Dict[str, int],
    drug_smiles: Dict[str, str],
    mono_se_matrix: np.ndarray,
    embedding_dim: int = 256,
    mono_method: Literal["tfidf", "cur"] = "tfidf",
    mono_components: int = 128,
    chemberta_model: str = "seyonec/ChemBERTa-zinc-base-v1",
    device: str = "cpu",
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute drug initialization tensor of shape (n_drugs, embedding_dim).

    Steps:
      1. Encode SMILES with ChemBERTa -> (n_drugs, 768)
      2. Encode mono-SEs with TF-IDF or CUR -> (n_drugs, mono_components)
      3. Fuse: concat -> linear proj -> (n_drugs, embedding_dim)

    Args:
        drug_to_id: drug identifier -> integer index.
        drug_smiles: drug identifier -> SMILES string.
        mono_se_matrix: (n_drugs, n_mono_ses) binary float array.
        embedding_dim: target embedding dimensionality.
        mono_method: 'tfidf' or 'cur'.
        mono_components: number of dimensions for mono-SE encoding.
        chemberta_model: HuggingFace model name.
        device: torch device string.
        cache_dir: if provided, cache/load encoded tensors from disk.

    Returns:
        (n_drugs, embedding_dim) float tensor.
    """
    from src.encoders.drug_encoder import (
        ChemBERTaEncoder, TFIDFMonoEncoder, CURMonoEncoder, DrugFusionEncoder
    )

    cache_path = Path(cache_dir) / f"drug_init_dim{embedding_dim}_{mono_method}.pt" if cache_dir else None
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached drug init from {cache_path}")
        return torch.load(cache_path)

    # -- ChemBERTa --
    logger.info("Encoding drug SMILES with ChemBERTa ...")
    cb_encoder = ChemBERTaEncoder(model_name=chemberta_model, device=device)
    chemberta_feats = cb_encoder.encode_drugs(drug_to_id, drug_smiles)  # (n, 768)

    # -- Mono SE --
    logger.info(f"Encoding mono side effects with {mono_method.upper()} ...")
    if mono_method == "tfidf":
        mono_enc = TFIDFMonoEncoder(n_components=mono_components)
    elif mono_method == "cur":
        mono_enc = CURMonoEncoder(n_components=mono_components)
    else:
        raise ValueError(f"Unknown mono_method: {mono_method!r}")

    mono_feats_np = mono_enc.fit_transform(mono_se_matrix)  # (n, mono_components)
    mono_feats = torch.tensor(mono_feats_np, dtype=torch.float32)

    # -- Fusion --
    logger.info("Fusing ChemBERTa + mono features ...")
    fusion = DrugFusionEncoder(
        embedding_dim=embedding_dim,
        chemberta_dim=chemberta_feats.shape[1],
        mono_dim=mono_feats.shape[1],
    )
    with torch.no_grad():
        drug_init = fusion(chemberta_feats, mono_feats)  # (n, embedding_dim)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(drug_init, cache_path)
        logger.info(f"Cached drug init to {cache_path}")

    return drug_init


def _sanitize_hf_id(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def _ablation_mlp_project(
    feats: torch.Tensor,
    in_dim: int,
    embedding_dim: int,
    seed: int,
) -> torch.Tensor:
    """Project modality features to embedding_dim (same depth as fusion MLPs)."""
    torch.manual_seed(seed)
    m = nn.Sequential(
        nn.Linear(in_dim, embedding_dim * 2),
        nn.LayerNorm(embedding_dim * 2),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(embedding_dim * 2, embedding_dim),
    )
    m.eval()
    with torch.no_grad():
        return m(feats)


def build_drug_chemberta_only_tensor(
    drug_to_id: Dict[str, int],
    drug_smiles: Dict[str, str],
    embedding_dim: int = 256,
    chemberta_model: str = "seyonec/ChemBERTa-zinc-base-v1",
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    projection_seed: int = 42,
) -> torch.Tensor:
    """
    ChemBERTa SMILES encoding only, projected to (n_drugs, embedding_dim).

    For ablations that need drug structure signal without monopharmacy SEs.
    """
    from src.encoders.drug_encoder import ChemBERTaEncoder

    tag = _sanitize_hf_id(chemberta_model)
    cache_path = (
        Path(cache_dir) / f"drug_ablation_chemberta_only_dim{embedding_dim}_{tag}.pt"
        if cache_dir
        else None
    )
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached ChemBERTa-only drug init from {cache_path}")
        return torch.load(cache_path)

    logger.info("Encoding drug SMILES with ChemBERTa (mono-free ablation) ...")
    cb_encoder = ChemBERTaEncoder(model_name=chemberta_model, device=device)
    chemberta_feats = cb_encoder.encode_drugs(drug_to_id, drug_smiles)
    in_dim = chemberta_feats.shape[1]
    drug_init = _ablation_mlp_project(
        chemberta_feats, in_dim, embedding_dim, projection_seed
    )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(drug_init, cache_path)
        logger.info(f"Cached ChemBERTa-only drug init to {cache_path}")

    return drug_init


def build_drug_mono_only_tensor(
    mono_se_matrix: np.ndarray,
    embedding_dim: int = 256,
    mono_method: Literal["tfidf", "cur"] = "tfidf",
    mono_components: int = 128,
    cache_dir: Optional[str] = None,
    projection_seed: int = 42,
) -> torch.Tensor:
    """
    Monopharmacy SE (TF-IDF or CUR) only, projected to (n_drugs, embedding_dim).

    For ablations that need pharmacological SE signal without ChemBERTa.
    """
    from src.encoders.drug_encoder import TFIDFMonoEncoder, CURMonoEncoder

    cache_path = (
        Path(cache_dir)
        / f"drug_ablation_mono_only_dim{embedding_dim}_{mono_method}_c{mono_components}.pt"
        if cache_dir
        else None
    )
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached mono-only drug init from {cache_path}")
        return torch.load(cache_path)

    logger.info(f"Encoding mono side effects with {mono_method.upper()} (ChemBERTa-free) ...")
    if mono_method == "tfidf":
        mono_enc = TFIDFMonoEncoder(n_components=mono_components)
    elif mono_method == "cur":
        mono_enc = CURMonoEncoder(n_components=mono_components)
    else:
        raise ValueError(f"Unknown mono_method: {mono_method!r}")

    mono_feats_np = mono_enc.fit_transform(mono_se_matrix)
    mono_feats = torch.tensor(mono_feats_np, dtype=torch.float32)
    in_dim = mono_feats.shape[1]
    drug_init = _ablation_mlp_project(mono_feats, in_dim, embedding_dim, projection_seed)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(drug_init, cache_path)
        logger.info(f"Cached mono-only drug init to {cache_path}")

    return drug_init


def build_protein_esm_ppi_ablation_tensors(
    protein_to_id: Dict[str, int],
    ppi_edges,
    protein_sequences: Dict[str, str],
    embedding_dim: int = 256,
    esm2_model: str = "facebook/esm2_t6_8M_UR50D",
    n_hops: int = 1,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    esm_projection_seed: int = 42,
    ppi_projection_seed: int = 43,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two protein matrices for ablation_track1_ml condition E:

      - esm_only:  ESM-2 sequence embedding per protein, projected to embedding_dim
      - ppi_only:  PPI neighbourhood mean-pool of neighbour ESM-2 vectors (same
                   esm_dim), projected with a *different* MLP seed so averaging
                   with esm_only is meaningful.

    Conditions A/B should pass ``esm_only`` as ``--prot_emb_esm2``; pass
    ``ppi_only`` as ``--prot_emb_ppi`` for condition E.

    Returns:
        (esm_only, ppi_neighbour_only), each (n_proteins, embedding_dim).
    """
    from src.encoders.protein_encoder import ESM2Encoder, PPINeighbourhoodAggregator
    from src.encoders.protein_encoder import ESM2_DIMS

    esm_tag = esm2_model.split("/")[-1]
    cache_esm = (
        Path(cache_dir)
        / f"protein_ablation_esmself_dim{embedding_dim}_{esm_tag}_h{n_hops}.pt"
        if cache_dir
        else None
    )
    cache_ppi = (
        Path(cache_dir)
        / f"protein_ablation_ppineighbour_dim{embedding_dim}_{esm_tag}_h{n_hops}.pt"
        if cache_dir
        else None
    )
    if cache_esm and cache_ppi and cache_esm.exists() and cache_ppi.exists():
        logger.info(f"Loading cached protein ablation tensors from {cache_dir}")
        return torch.load(cache_esm), torch.load(cache_ppi)

    logger.info("Encoding protein sequences with ESM-2 (ablation branches) ...")
    esm_encoder = ESM2Encoder(model_name=esm2_model, device=device)
    esm_self = esm_encoder.encode_proteins(protein_to_id, protein_sequences)
    esm_dim = ESM2_DIMS.get(esm2_model, 320)

    logger.info(f"Aggregating PPI neighbourhood (n_hops={n_hops}) ...")
    agg = PPINeighbourhoodAggregator(n_hops=n_hops)
    esm_neighbourhood = agg.aggregate(esm_self, ppi_edges, protein_to_id)

    esm_only = _ablation_mlp_project(
        esm_self, esm_dim, embedding_dim, esm_projection_seed
    )
    ppi_only = _ablation_mlp_project(
        esm_neighbourhood, esm_dim, embedding_dim, ppi_projection_seed
    )

    if cache_esm and cache_ppi:
        cache_esm.parent.mkdir(parents=True, exist_ok=True)
        torch.save(esm_only, cache_esm)
        torch.save(ppi_only, cache_ppi)
        logger.info(f"Cached protein ablation tensors to {cache_esm} and {cache_ppi}")

    return esm_only, ppi_only


def build_protein_init_tensor(
    protein_to_id: Dict[str, int],
    ppi_edges,
    drug_targets,
    drug_to_id: Dict[str, int],
    protein_sequences: Dict[str, str],
    embedding_dim: int = 256,
    esm2_model: str = "facebook/esm2_t6_8M_UR50D",
    n_hops: int = 1,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """
    Compute protein initialization tensor of shape (n_proteins, embedding_dim).

    Steps:
      1. Encode sequences with ESM-2 -> (n_proteins, esm_dim)
      2. Aggregate over PPI neighbourhood -> (n_proteins, esm_dim)
      3. Fuse: concat self + neighbourhood -> linear proj -> (n_proteins, embedding_dim)

    Args:
        protein_to_id: protein identifier -> integer index.
        ppi_edges: list of (protein_a, protein_b) string tuples.
        drug_targets: drug identifier -> set of target protein identifiers.
        drug_to_id: drug identifier -> integer index.
        protein_sequences: protein identifier -> amino acid sequence.
        embedding_dim: target embedding dimensionality.
        esm2_model: HuggingFace ESM-2 model name.
        n_hops: PPI neighbourhood aggregation hops.
        device: torch device string.
        cache_dir: if provided, cache/load encoded tensors from disk.

    Returns:
        (n_proteins, embedding_dim) float tensor.
    """
    from src.encoders.protein_encoder import (
        ESM2Encoder, PPINeighbourhoodAggregator, ProteinFusionEncoder
    )
    from src.encoders.protein_encoder import ESM2_DIMS

    cache_path = Path(cache_dir) / f"protein_init_dim{embedding_dim}_{esm2_model.split('/')[-1]}.pt" if cache_dir else None
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached protein init from {cache_path}")
        return torch.load(cache_path)

    # -- ESM-2 --
    logger.info("Encoding protein sequences with ESM-2 ...")
    esm_encoder = ESM2Encoder(model_name=esm2_model, device=device)
    esm_self = esm_encoder.encode_proteins(protein_to_id, protein_sequences)
    esm_dim = ESM2_DIMS.get(esm2_model, 320)

    # -- PPI neighbourhood aggregation --
    logger.info(f"Aggregating PPI neighbourhood (n_hops={n_hops}) ...")
    agg = PPINeighbourhoodAggregator(n_hops=n_hops)
    esm_neighbourhood = agg.aggregate(esm_self, ppi_edges, protein_to_id)

    # -- Fusion --
    logger.info("Fusing ESM-2 self + neighbourhood features ...")
    fusion = ProteinFusionEncoder(
        embedding_dim=embedding_dim,
        esm_dim=esm_dim,
    )
    with torch.no_grad():
        protein_init = fusion(esm_self, esm_neighbourhood)  # (n, embedding_dim)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(protein_init, cache_path)
        logger.info(f"Cached protein init to {cache_path}")

    return protein_init


def build_pykeen_model(
    train_tf,
    entity_to_id: Dict[str, int],
    drug_init: torch.Tensor,
    protein_init: torch.Tensor,
    drug_to_id: Dict[str, int],
    protein_to_id: Dict[str, int],
    interaction: Literal["SimplE", "ComplEx", "DistMult"] = "SimplE",
    embedding_dim: int = 256,
    loss: str = "CrossEntropyLoss",
    regularization_weight: float = 1e-3,
    entity_dropout: Optional[float] = 0.0,
    relation_dropout: Optional[float] = 0.0,
    random_seed: int = 42,
):
    """
    Build a PyKEEN model with pretrained entity initializations.

    Uses PyKEEN's dedicated model classes (ComplEx, SimplE, DistMult) rather
    than ERModel directly, so relation representations are configured
    automatically for each interaction type.

    The pretrained initializer is applied to the primary entity embedding
    (real part for ComplEx/SimplE, single embedding for DistMult).  Additional
    embeddings required by each interaction (imaginary parts, inverse relations)
    use PyKEEN's default xavier_uniform_ initializer.

    The merged entity embedding matrix has:
        rows [0 .. n_drugs-1]               <- drug_init rows
        rows [n_drugs .. n_drugs+n_prot-1]  <- protein_init rows

    This ordering must match the entity_to_id mapping produced by
    decagon_loader.py:build_pykeen_triples().

    Args:
        train_tf: PyKEEN TriplesFactory.
        entity_to_id: merged entity -> id mapping.
        drug_init: (n_drugs, embedding_dim) drug initialization tensor.
        protein_init: (n_proteins, embedding_dim) protein initialization tensor.
        drug_to_id: drug identifier -> integer index.
        protein_to_id: protein identifier -> integer index.
        interaction: KGE interaction function name.
        embedding_dim: must match drug_init / protein_init dim.
        loss: loss function name — CrossEntropyLoss | BCEWithLogitsLoss |
              SoftplusLoss | MarginRankingLoss.
        regularization_weight: L3 regularization weight on entity/relation
              embeddings. Set to 0 to disable. (default 1e-3)
        entity_dropout: Dropout on entity embeddings (0 = off). Lloyd best: 0.068.
        relation_dropout: Dropout on relation embeddings (0 = off). Lloyd best: 0.125.
        random_seed: for reproducibility.

    Returns:
        PyKEEN model ready for training.
    """
    from pykeen.models import ComplEx, DistMult, SimplE
    from pykeen.nn.init import PretrainedInitializer
    import pykeen.losses as pykeen_losses
    from pykeen.regularizers import LpRegularizer

    n_entities = len(entity_to_id)
    n_drugs = len(drug_to_id)
    n_proteins = len(protein_to_id)

    # Build merged init tensor: drugs first, then proteins
    entity_init = torch.zeros(n_entities, embedding_dim)
    entity_init[:n_drugs] = drug_init
    entity_init[n_drugs : n_drugs + n_proteins] = protein_init

    logger.info(
        f"Building {interaction} model | "
        f"entities: {n_entities} (drugs: {n_drugs}, proteins: {n_proteins}) | "
        f"embedding_dim: {embedding_dim}"
    )

    # Loss function
    loss_map = {
        "CrossEntropyLoss":    pykeen_losses.CrossEntropyLoss,
        "BCEWithLogitsLoss":   pykeen_losses.BCEWithLogitsLoss,
        "SoftplusLoss":        pykeen_losses.SoftplusLoss,
        "MarginRankingLoss":   pykeen_losses.MarginRankingLoss,
    }
    if loss not in loss_map:
        raise ValueError(
            f"Unknown loss {loss!r}. Choose from: {list(loss_map)}"
        )
    loss_instance = loss_map[loss]()
    logger.info(f"Loss: {loss}")

    # L3 regularization (same as Lloyd et al. SimplE default)
    regularizer = None
    if regularization_weight > 0:
        regularizer = LpRegularizer(p=3, weight=regularization_weight)
        logger.info(f"L3 regularization weight: {regularization_weight}")

    common = dict(
        triples_factory=train_tf,
        embedding_dim=embedding_dim,
        loss=loss_instance,
        random_seed=random_seed,
    )
    if regularizer is not None:
        common["entity_regularizer"] = regularizer
        common["relation_regularizer"] = regularizer

    # PyKEEN passes these to the underlying Embedding representations.
    repr_kw: Dict[str, Any] = {}
    ed = 0.0 if entity_dropout is None else float(entity_dropout)
    rd = 0.0 if relation_dropout is None else float(relation_dropout)
    if ed > 0:
        repr_kw["entity_representations_kwargs"] = {"dropout": ed}
        logger.info(f"Entity embedding dropout: {ed}")
    if rd > 0:
        repr_kw["relation_representations_kwargs"] = {"dropout": rd}
        logger.info(f"Relation embedding dropout: {rd}")

    if interaction == "ComplEx":
        # PyKEEN's ComplEx stores (real, imag) in a single tensor of shape
        # (n_entities, embedding_dim, 2).  Initialise real part from our
        # pretrained vectors; imaginary part starts at zero.
        entity_init_cpx = torch.stack(
            [entity_init, torch.zeros_like(entity_init)], dim=-1
        )  # (n, d, 2)
        model = ComplEx(
            **common,
            entity_initializer=PretrainedInitializer(tensor=entity_init_cpx),
            **repr_kw,
        )
    elif interaction == "SimplE":
        # SimplE has two separate (n, d) embeddings — standard shape works.
        model = SimplE(
            **common,
            entity_initializer=PretrainedInitializer(tensor=entity_init),
            **repr_kw,
        )
    elif interaction == "DistMult":
        model = DistMult(
            **common,
            entity_initializer=PretrainedInitializer(tensor=entity_init),
            **repr_kw,
        )
    else:
        raise ValueError(
            f"Unknown interaction {interaction!r}. "
            "Choose from: 'SimplE', 'ComplEx', 'DistMult'."
        )

    return model
