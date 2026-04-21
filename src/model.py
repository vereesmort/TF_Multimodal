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
from typing import Dict, Literal, Optional, Tuple

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
):
    """
    Build a PyKEEN ERModel with pretrained entity initializations.

    The merged entity embedding matrix has:
        rows [0 .. n_drugs-1]          <- drug_init rows (in drug_to_id order)
        rows [n_drugs .. n_drugs+n_prot-1] <- protein_init rows

    This ordering must match the entity_to_id mapping produced by
    data/decagon_loader.py:build_pykeen_triples().

    Args:
        train_tf: PyKEEN TriplesFactory.
        entity_to_id: merged entity -> id mapping.
        drug_init: (n_drugs, embedding_dim) drug initialization tensor.
        protein_init: (n_proteins, embedding_dim) protein initialization tensor.
        drug_to_id: drug identifier -> integer index.
        protein_to_id: protein identifier -> integer index.
        interaction: KGE interaction function name.
        embedding_dim: must match drug_init / protein_init dim.

    Returns:
        PyKEEN model ready for training.
    """
    from pykeen.models import ERModel
    from pykeen.nn import Embedding
    from pykeen.nn.init import PretrainedInitializer

    n_entities = len(entity_to_id)
    n_drugs = len(drug_to_id)
    n_proteins = len(protein_to_id)

    # Build merged init tensor in entity_to_id order
    # drug entities have ids 0..n_drugs-1, proteins n_drugs..n_drugs+n_proteins-1
    entity_init = torch.zeros(n_entities, embedding_dim)
    entity_init[:n_drugs] = drug_init
    entity_init[n_drugs : n_drugs + n_proteins] = protein_init

    logger.info(
        f"Building {interaction} model | "
        f"entities: {n_entities} (drugs: {n_drugs}, proteins: {n_proteins}) | "
        f"embedding_dim: {embedding_dim}"
    )

    model = ERModel(
        triples_factory=train_tf,
        interaction=interaction,
        entity_representations=Embedding(
            max_id=n_entities,
            embedding_dim=embedding_dim,
            initializer=PretrainedInitializer(tensor=entity_init),
        ),
        entity_representations_kwargs=None,
    )
    return model
