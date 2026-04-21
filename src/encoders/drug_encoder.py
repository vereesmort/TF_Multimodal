"""
Drug entity encoders.

Provides two complementary drug representations:

1. ChemBERTa encoder
   - Encodes drug SMILES strings using a RoBERTa model pre-trained on SMILES
     (seyonec/ChemBERTa-zinc-base-v1 from HuggingFace).
   - Produces a 768-d embedding per drug capturing molecular structure.

2. Monopharmacy side effect encoder
   - Takes the binary drug x monopharmacy-SE matrix.
   - Offers two featurization strategies:
       a) TF-IDF: downweights ubiquitous side effects (high doc freq)
          and upweights discriminative ones.  Interpretable — the top
          TF-IDF terms for a drug identify its pharmacological signature.
       b) CUR decomposition: selects actual columns (real SEs) and rows
          (real drugs) from the matrix, making the reduced representation
          fully interpretable in terms of original side effects.

Both outputs are projected to a common embedding_dim via a learned
linear layer during model initialization.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChemBERTa encoder
# ---------------------------------------------------------------------------

class ChemBERTaEncoder(nn.Module):
    """
    Encode drug SMILES using ChemBERTa (RoBERTa trained on SMILES).

    Model: seyonec/ChemBERTa-zinc-base-v1  (HuggingFace)
    Output: mean-pooled last hidden state, shape (n_drugs, 768)

    Args:
        model_name: HuggingFace model identifier.
        batch_size: SMILES strings processed per forward pass.
        freeze: If True, ChemBERTa weights are frozen (faster, less memory).
                If False, fine-tuned end-to-end — only recommended with small
                embedding dims and sufficient VRAM.
        device: torch device string.
    """

    DEFAULT_MODEL = "seyonec/ChemBERTa-zinc-base-v1"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        freeze: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.freeze = freeze
        self.device = torch.device(device)
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy-load to avoid import-time HuggingFace dependency."""
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "transformers is required for ChemBERTa encoding. "
                "Install with: pip install transformers"
            )
        logger.info(f"Loading ChemBERTa: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        if self.freeze:
            for p in self._model.parameters():
                p.requires_grad_(False)
            self._model.eval()

    @torch.no_grad()
    def encode(self, smiles_list: List[str]) -> torch.Tensor:
        """
        Encode a list of SMILES strings.

        Args:
            smiles_list: SMILES strings, one per drug.

        Returns:
            Tensor of shape (n, 768) on CPU.
        """
        if not smiles_list:
            raise ValueError(
                "smiles_list is empty. Make sure the Decagon data files are present "
                "in raw_dir and were loaded correctly (run: bash data/raw/download_unpack.sh)."
            )

        if self._model is None:
            self._load()

        all_embeddings = []
        for i in range(0, len(smiles_list), self.batch_size):
            batch = smiles_list[i : i + self.batch_size]
            encoded = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.set_grad_enabled(not self.freeze):
                outputs = self._model(**encoded)
            # Mean pool over token dimension (ignore padding via attention_mask)
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            token_embeddings = outputs.last_hidden_state
            sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
            count = attention_mask.sum(dim=1).clamp(min=1e-9)
            mean_embeddings = sum_embeddings / count
            all_embeddings.append(mean_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode_drugs(
        self,
        drug_to_id: Dict[str, int],
        drug_smiles: Dict[str, str],
        fallback: str = "C",
    ) -> torch.Tensor:
        """
        Produce a (n_drugs, 768) tensor in drug_to_id order.

        Drugs without a known SMILES are encoded with the fallback string
        (single carbon atom by default — better than a zero vector).

        Args:
            drug_to_id: mapping from drug identifier to integer index.
            drug_smiles: mapping from drug identifier to SMILES string.
            fallback: SMILES used for drugs without known structure.

        Returns:
            Tensor (n_drugs, 768).
        """
        n = len(drug_to_id)
        smiles_ordered = []
        for drug, _ in sorted(drug_to_id.items(), key=lambda x: x[1]):
            smiles_ordered.append(drug_smiles.get(drug, fallback))

        missing = sum(1 for d in drug_to_id if d not in drug_smiles)
        if missing > 0:
            logger.warning(
                f"{missing}/{n} drugs have no SMILES; using fallback '{fallback}'"
            )

        return self.encode(smiles_ordered)


# ---------------------------------------------------------------------------
# Monopharmacy side effect encoders
# ---------------------------------------------------------------------------

class TFIDFMonoEncoder:
    """
    Encode monopharmacy side effects using TF-IDF weighting.

    Rationale:
        The raw binary mono-SE matrix treats all side effects equally.
        TF-IDF downweights side effects that appear in nearly every drug
        (e.g. nausea, headache) and upweights those that are drug-specific,
        making the resulting feature vector more discriminative.

    Interpretability note:
        The top-k TF-IDF dimensions for a drug correspond to real, named
        side effects — they can be inspected directly as pharmacological
        fingerprints.

    Args:
        n_components: target dimensionality after SVD compression.
                      Set to None to keep the full TF-IDF vector.
        smooth_idf: add 1 to document frequencies to avoid zero division.
    """

    def __init__(self, n_components: Optional[int] = 128, smooth_idf: bool = True):
        self.n_components = n_components
        self.smooth_idf = smooth_idf
        self._idf: Optional[np.ndarray] = None
        self._svd_components: Optional[np.ndarray] = None  # (n_components, n_features)
        self._is_fit = False

    def fit(self, mono_matrix: np.ndarray) -> "TFIDFMonoEncoder":
        """
        Compute IDF weights from mono_matrix.

        Args:
            mono_matrix: (n_drugs, n_mono_ses) binary float array.
        """
        n_docs = mono_matrix.shape[0]
        # Document frequency: how many drugs have each SE
        df = (mono_matrix > 0).sum(axis=0).astype(float)
        if self.smooth_idf:
            self._idf = np.log((n_docs + 1) / (df + 1)) + 1.0
        else:
            # Mask zero-df SEs
            df = np.where(df == 0, 1, df)
            self._idf = np.log(n_docs / df) + 1.0

        tfidf = mono_matrix * self._idf[np.newaxis, :]

        if self.n_components is not None and self.n_components < tfidf.shape[1]:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=self.n_components, random_state=42)
            svd.fit(tfidf)
            self._svd_components = svd.components_  # (n_components, n_features)
            explained = svd.explained_variance_ratio_.sum()
            logger.info(
                f"TF-IDF SVD: {self.n_components} components explain "
                f"{explained:.1%} of variance"
            )

        self._is_fit = True
        return self

    def transform(self, mono_matrix: np.ndarray) -> np.ndarray:
        """
        Apply TF-IDF weighting (and optional SVD reduction).

        Args:
            mono_matrix: (n_drugs, n_mono_ses) binary array.

        Returns:
            (n_drugs, n_components) or (n_drugs, n_mono_ses) float array.
        """
        assert self._is_fit, "Call fit() before transform()"
        tfidf = mono_matrix * self._idf[np.newaxis, :]

        if self._svd_components is not None:
            return tfidf @ self._svd_components.T
        return tfidf

    def fit_transform(self, mono_matrix: np.ndarray) -> np.ndarray:
        return self.fit(mono_matrix).transform(mono_matrix)

    def top_side_effects(
        self,
        drug_idx: int,
        mono_matrix: np.ndarray,
        se_id_to_name: Dict[int, str],
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Return the top-k most discriminative side effects for a drug.

        Useful for interpretability analysis in the thesis.

        Args:
            drug_idx: row index in mono_matrix.
            mono_matrix: (n_drugs, n_mono_ses) binary array.
            se_id_to_name: mapping from SE column index to SE name.
            k: number of top side effects to return.

        Returns:
            List of (se_name, tfidf_weight) sorted descending.
        """
        assert self._is_fit
        tfidf_row = mono_matrix[drug_idx] * self._idf
        top_k_idx = np.argsort(tfidf_row)[::-1][:k]
        return [(se_id_to_name.get(i, str(i)), float(tfidf_row[i])) for i in top_k_idx]


class CURMonoEncoder:
    """
    Encode monopharmacy side effects using CUR matrix decomposition.

    CUR is an interpretable alternative to SVD/PCA.  Instead of computing
    abstract linear combinations of all side effects, it selects a subset of
    actual columns (real side effects, C) and actual rows (real drugs, R) from
    the original matrix, forming a low-rank approximation.

    This means the reduced representation is directly interpretable:
    each dimension corresponds to a specific real side effect (or drug),
    not a linear combination.

    Reference:
        Mahoney & Drineas (2009). CUR matrix decompositions for improved
        data analysis. PNAS.

    Args:
        n_components: number of columns (side effects) to select.
        leverage_sampling: if True, sample columns proportional to their
                           column norms (statistical leverage scores).
                           If False, selects top-norm columns deterministically.
    """

    def __init__(self, n_components: int = 128, leverage_sampling: bool = True):
        self.n_components = n_components
        self.leverage_sampling = leverage_sampling
        self._selected_cols: Optional[np.ndarray] = None  # indices
        self._U: Optional[np.ndarray] = None  # pseudo-inverse of intersect
        self._is_fit = False

    def fit(self, mono_matrix: np.ndarray, random_state: int = 42) -> "CURMonoEncoder":
        """
        Select columns and precompute CUR factorization.

        Args:
            mono_matrix: (n_drugs, n_ses) binary float array.
        """
        n, m = mono_matrix.shape
        k = min(self.n_components, m)

        # Column norms as importance scores
        col_norms_sq = (mono_matrix ** 2).sum(axis=0)
        col_probs = col_norms_sq / col_norms_sq.sum()

        rng = np.random.default_rng(random_state)
        if self.leverage_sampling:
            self._selected_cols = rng.choice(
                m, size=k, replace=False, p=col_probs
            )
        else:
            self._selected_cols = np.argsort(col_norms_sq)[::-1][:k]

        logger.info(
            f"CUR: selected {k} columns (side effects) out of {m}"
        )
        self._is_fit = True
        return self

    def transform(self, mono_matrix: np.ndarray) -> np.ndarray:
        """
        Project mono_matrix onto selected side effect columns.

        Args:
            mono_matrix: (n_drugs, n_ses) array.

        Returns:
            (n_drugs, n_components) array — values correspond to membership
            in selected side effects, interpretable as a side-effect profile.
        """
        assert self._is_fit, "Call fit() before transform()"
        return mono_matrix[:, self._selected_cols]

    def fit_transform(self, mono_matrix: np.ndarray) -> np.ndarray:
        return self.fit(mono_matrix).transform(mono_matrix)

    @property
    def selected_side_effect_indices(self) -> Optional[np.ndarray]:
        """Integer indices of the selected columns (side effects)."""
        return self._selected_cols


# ---------------------------------------------------------------------------
# Drug fusion: combines ChemBERTa + mono-SE features
# ---------------------------------------------------------------------------

class DrugFusionEncoder(nn.Module):
    """
    Fuse ChemBERTa molecular embeddings with monopharmacy SE features.

    Architecture:
        [chemberta (768) | mono_se (n_components)] -> linear -> embedding_dim

    The fusion MLP is trained jointly with the TF decoder during KGE training.

    Args:
        embedding_dim: target embedding dimension (must match KGE model dim).
        chemberta_dim: output dim of ChemBERTa (768 by default).
        mono_dim: output dim of monopharmacy encoder.
        dropout: dropout on the projection layer.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        chemberta_dim: int = 768,
        mono_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = chemberta_dim + mono_dim
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
        chemberta_feats: torch.Tensor,
        mono_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            chemberta_feats: (n_drugs, 768)
            mono_feats: (n_drugs, mono_dim)

        Returns:
            (n_drugs, embedding_dim) fused drug embeddings.
        """
        x = torch.cat([chemberta_feats, mono_feats], dim=-1)
        return self.proj(x)
