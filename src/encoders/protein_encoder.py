"""
Protein entity encoder.

Two-stage encoding:

Stage 1 — ESM-2 sequence embeddings
    Encodes each protein's amino acid sequence using ESM-2 (facebook/esm2_t6_8M_UR50D
    by default — the smallest ESM-2 variant).  Larger variants (t12, t30, t33, t36, t48)
    trade speed for quality.

Stage 2 — PPI neighbourhood aggregation
    For each protein, mean-pools the ESM-2 embeddings of its PPI neighbours.
    This explicitly encodes the local network context of a protein, directly
    addressing the reviewer's criticism that the original paper's TF models
    only implicitly learn neighbourhood structure through the factorization
    objective.

Fusion:
    [esm2 | ppi_neighbourhood] -> linear -> embedding_dim
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ESM2_MODELS = {
    "small":  "facebook/esm2_t6_8M_UR50D",    # 8M params,  320-d
    "medium": "facebook/esm2_t12_35M_UR50D",   # 35M params, 480-d
    "large":  "facebook/esm2_t30_150M_UR50D",  # 150M params, 640-d
    "xlarge": "facebook/esm2_t33_650M_UR50D",  # 650M params, 1280-d
}

ESM2_DIMS = {
    "facebook/esm2_t6_8M_UR50D":     320,
    "facebook/esm2_t12_35M_UR50D":   480,
    "facebook/esm2_t30_150M_UR50D":  640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D":   2560,
    "facebook/esm2_t48_15B_UR50D":  5120,
}


class ESM2Encoder(nn.Module):
    """
    Encode protein sequences using ESM-2.

    Args:
        model_name: HuggingFace ESM-2 model string.
        batch_size: sequences per forward pass.
        freeze: freeze ESM-2 weights (recommended: True to save memory).
        max_length: max token length (ESM-2 supports up to 1022 residues).
        device: torch device string.
    """

    def __init__(
        self,
        model_name: str = ESM2_MODELS["small"],
        batch_size: int = 32,
        freeze: bool = True,
        max_length: int = 512,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.freeze = freeze
        self.max_length = max_length
        self.device = torch.device(device)
        self.output_dim = ESM2_DIMS.get(model_name, 320)
        self._model = None
        self._tokenizer = None

    def _load(self):
        try:
            from transformers import EsmTokenizer, EsmModel
        except ImportError:
            raise ImportError(
                "transformers>=4.20 is required for ESM-2. "
                "Install with: pip install transformers"
            )
        logger.info(f"Loading ESM-2: {self.model_name}")
        self._tokenizer = EsmTokenizer.from_pretrained(self.model_name)
        self._model = EsmModel.from_pretrained(self.model_name).to(self.device)
        if self.freeze:
            for p in self._model.parameters():
                p.requires_grad_(False)
            self._model.eval()

    @torch.no_grad()
    def encode(self, sequences: List[str]) -> torch.Tensor:
        """
        Encode amino acid sequences.

        Args:
            sequences: list of amino acid strings (single-letter code).

        Returns:
            (n, esm_dim) tensor on CPU.
        """
        if self._model is None:
            self._load()

        all_embeddings = []
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.set_grad_enabled(not self.freeze):
                outputs = self._model(**encoded)
            # Mean pool (masked)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode_proteins(
        self,
        protein_to_id: Dict[str, int],
        protein_sequences: Dict[str, str],
        fallback_seq: str = "M",
    ) -> torch.Tensor:
        """
        Produce a (n_proteins, esm_dim) tensor in protein_to_id order.

        Proteins without a known sequence use a single methionine as fallback.

        Args:
            protein_to_id: protein identifier -> integer index.
            protein_sequences: protein identifier -> amino acid sequence.
            fallback_seq: amino acid sequence for proteins without data.

        Returns:
            (n_proteins, esm_dim) tensor.
        """
        n = len(protein_to_id)
        seqs_ordered = []
        for prot, _ in sorted(protein_to_id.items(), key=lambda x: x[1]):
            seqs_ordered.append(protein_sequences.get(prot, fallback_seq))

        missing = sum(1 for p in protein_to_id if p not in protein_sequences)
        if missing:
            logger.warning(
                f"{missing}/{n} proteins have no sequence; using fallback"
            )

        return self.encode(seqs_ordered)


class PPINeighbourhoodAggregator:
    """
    Aggregate ESM-2 protein embeddings over the PPI graph neighbourhood.

    For each protein node p, this computes:

        ppi_agg(p) = mean{ esm2(q) | q ∈ N(p) }

    where N(p) are the direct PPI neighbours of p.

    This produces an explicit, graph-informed representation of each
    protein's local context — addressing the reviewer's criticism that
    pure TF models do not encode protein neighbourhood structure.

    For downstream drug representation, a drug's target proteins' aggregated
    embeddings are further mean-pooled to produce a drug-level protein context.

    Args:
        n_hops: number of neighbourhood hops (1 = direct neighbours only,
                2 = 2-hop neighbourhood, etc.).  Recommended: 1 or 2.
    """

    def __init__(self, n_hops: int = 1):
        self.n_hops = n_hops

    def aggregate(
        self,
        esm2_embeddings: torch.Tensor,
        ppi_edges: List[Tuple[str, str]],
        protein_to_id: Dict[str, int],
    ) -> torch.Tensor:
        """
        Compute neighbourhood-aggregated embeddings for all proteins.

        Args:
            esm2_embeddings: (n_proteins, esm_dim) base ESM-2 embeddings.
            ppi_edges: list of (protein_a, protein_b) string identifier pairs.
            protein_to_id: mapping from protein identifier to row index.

        Returns:
            (n_proteins, esm_dim) neighbourhood-aggregated embeddings.
            Proteins with no PPI neighbours retain their own ESM-2 embedding.
        """
        n = len(protein_to_id)
        esm_dim = esm2_embeddings.shape[1]

        # Build adjacency list
        adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}
        for p1, p2 in ppi_edges:
            i1 = protein_to_id.get(p1)
            i2 = protein_to_id.get(p2)
            if i1 is not None and i2 is not None:
                adjacency[i1].append(i2)
                adjacency[i2].append(i1)

        # Aggregate over n_hops
        current = esm2_embeddings.clone()
        for hop in range(self.n_hops):
            aggregated = torch.zeros_like(current)
            for node_idx, neighbours in adjacency.items():
                if neighbours:
                    nb_embs = current[neighbours]  # (k, esm_dim)
                    aggregated[node_idx] = nb_embs.mean(dim=0)
                else:
                    # Isolated proteins keep their own embedding
                    aggregated[node_idx] = current[node_idx]
            current = aggregated
            logger.info(f"PPI aggregation: hop {hop+1}/{self.n_hops} done")

        return current

    def drug_protein_context(
        self,
        ppi_aggregated: torch.Tensor,
        drug_targets: Dict[str, Set[str]],
        drug_to_id: Dict[str, int],
        protein_to_id: Dict[str, int],
    ) -> torch.Tensor:
        """
        Compute drug-level protein neighbourhood context.

        For each drug, mean-pools the PPI-aggregated embeddings of all its
        known target proteins.

        Args:
            ppi_aggregated: (n_proteins, esm_dim) from aggregate().
            drug_targets: drug identifier -> set of target protein identifiers.
            drug_to_id: drug identifier -> integer index.
            protein_to_id: protein identifier -> integer index.

        Returns:
            (n_drugs, esm_dim) tensor.  Drugs with no known targets get a
            zero vector (signal explicitly absent, not disguised).
        """
        n_drugs = len(drug_to_id)
        esm_dim = ppi_aggregated.shape[1]
        drug_context = torch.zeros(n_drugs, esm_dim)

        no_targets = 0
        for drug, targets in drug_targets.items():
            drug_idx = drug_to_id.get(drug)
            if drug_idx is None:
                continue
            target_indices = [
                protein_to_id[t] for t in targets if t in protein_to_id
            ]
            if target_indices:
                drug_context[drug_idx] = ppi_aggregated[target_indices].mean(dim=0)
            else:
                no_targets += 1

        if no_targets:
            logger.warning(
                f"{no_targets} drugs have no PPI-covered targets; "
                f"their protein context is a zero vector"
            )
        return drug_context


class ProteinFusionEncoder(nn.Module):
    """
    Fuse ESM-2 per-protein embeddings with PPI-neighbourhood context.

    Architecture:
        [esm2_self (esm_dim) | esm2_neighbourhood (esm_dim)]
            -> linear -> embedding_dim

    Args:
        embedding_dim: target embedding dimension.
        esm_dim: ESM-2 output dimension (depends on model variant).
        dropout: dropout on projection layer.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        esm_dim: int = 320,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = esm_dim * 2  # self + neighbourhood
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(
        self,
        esm2_self: torch.Tensor,
        esm2_neighbourhood: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            esm2_self: (n_proteins, esm_dim) base ESM-2 embeddings.
            esm2_neighbourhood: (n_proteins, esm_dim) PPI-aggregated embeddings.

        Returns:
            (n_proteins, embedding_dim) fused protein embeddings.
        """
        x = torch.cat([esm2_self, esm2_neighbourhood], dim=-1)
        return self.proj(x)
