"""
Unit tests for TF-Decagon Improved.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pytest
import torch


# -----------------------------------------------------------------------
# Drug encoder tests
# -----------------------------------------------------------------------

class TestTFIDFMonoEncoder:
    def _make_matrix(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 2, size=(50, 200)).astype(np.float32)

    def test_fit_transform_shape(self):
        from src.encoders.drug_encoder import TFIDFMonoEncoder
        mat = self._make_matrix()
        enc = TFIDFMonoEncoder(n_components=32)
        out = enc.fit_transform(mat)
        assert out.shape == (50, 32), f"Expected (50, 32), got {out.shape}"

    def test_no_reduction(self):
        from src.encoders.drug_encoder import TFIDFMonoEncoder
        mat = self._make_matrix()
        enc = TFIDFMonoEncoder(n_components=None)
        out = enc.fit_transform(mat)
        assert out.shape == (50, 200)

    def test_idf_weights_nonnegative(self):
        from src.encoders.drug_encoder import TFIDFMonoEncoder
        mat = self._make_matrix()
        enc = TFIDFMonoEncoder(n_components=None)
        enc.fit(mat)
        assert (enc._idf >= 0).all()

    def test_top_side_effects_returns_k(self):
        from src.encoders.drug_encoder import TFIDFMonoEncoder
        mat = self._make_matrix()
        enc = TFIDFMonoEncoder(n_components=None)
        enc.fit(mat)
        se_names = {i: f"SE_{i}" for i in range(200)}
        top = enc.top_side_effects(0, mat, se_names, k=5)
        assert len(top) == 5
        # Check sorted descending by weight
        weights = [w for _, w in top]
        assert weights == sorted(weights, reverse=True)

    def test_common_se_has_lower_weight(self):
        """A side effect present in all drugs should get lower IDF weight."""
        from src.encoders.drug_encoder import TFIDFMonoEncoder
        mat = np.zeros((20, 10), dtype=np.float32)
        mat[:, 0] = 1.0   # SE 0 in every drug (low IDF)
        mat[0, 1] = 1.0   # SE 1 in only one drug (high IDF)
        enc = TFIDFMonoEncoder(n_components=None, smooth_idf=False)
        enc.fit(mat)
        assert enc._idf[1] > enc._idf[0], "Rare SE should have higher IDF"


class TestCURMonoEncoder:
    def _make_matrix(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 2, size=(50, 200)).astype(np.float32)

    def test_fit_transform_shape(self):
        from src.encoders.drug_encoder import CURMonoEncoder
        mat = self._make_matrix()
        enc = CURMonoEncoder(n_components=30)
        out = enc.fit_transform(mat)
        assert out.shape == (50, 30)

    def test_selected_cols_are_valid_indices(self):
        from src.encoders.drug_encoder import CURMonoEncoder
        mat = self._make_matrix()
        enc = CURMonoEncoder(n_components=30)
        enc.fit(mat)
        sel = enc.selected_side_effect_indices
        assert len(sel) == 30
        assert (sel >= 0).all() and (sel < 200).all()

    def test_selected_cols_no_duplicates(self):
        from src.encoders.drug_encoder import CURMonoEncoder
        mat = self._make_matrix()
        enc = CURMonoEncoder(n_components=30, leverage_sampling=False)
        enc.fit(mat)
        assert len(set(enc.selected_side_effect_indices.tolist())) == 30

    def test_transform_values_match_original(self):
        """CUR output should be a column subset of the input."""
        from src.encoders.drug_encoder import CURMonoEncoder
        mat = self._make_matrix()
        enc = CURMonoEncoder(n_components=30, leverage_sampling=False)
        out = enc.fit_transform(mat)
        sel = enc.selected_side_effect_indices
        expected = mat[:, sel]
        np.testing.assert_array_equal(out, expected)


class TestDrugFusionEncoder:
    def test_output_shape(self):
        from src.encoders.drug_encoder import DrugFusionEncoder
        fusion = DrugFusionEncoder(embedding_dim=64, chemberta_dim=768, mono_dim=128)
        chem = torch.randn(10, 768)
        mono = torch.randn(10, 128)
        out = fusion(chem, mono)
        assert out.shape == (10, 64)

    def test_output_dim_attribute(self):
        from src.encoders.drug_encoder import DrugFusionEncoder
        fusion = DrugFusionEncoder(embedding_dim=128)
        assert fusion.embedding_dim == 128


# -----------------------------------------------------------------------
# Protein encoder tests
# -----------------------------------------------------------------------

class TestPPINeighbourhoodAggregator:
    def test_aggregation_shape_preserved(self):
        from src.encoders.protein_encoder import PPINeighbourhoodAggregator
        n_proteins, dim = 20, 32
        embeddings = torch.randn(n_proteins, dim)
        protein_to_id = {f"p{i}": i for i in range(n_proteins)}
        ppi_edges = [(f"p{i}", f"p{(i+1) % n_proteins}") for i in range(n_proteins)]

        agg = PPINeighbourhoodAggregator(n_hops=1)
        out = agg.aggregate(embeddings, ppi_edges, protein_to_id)
        assert out.shape == (n_proteins, dim)

    def test_isolated_protein_keeps_own_embedding(self):
        """Proteins with no PPI neighbours should keep their own embedding."""
        from src.encoders.protein_encoder import PPINeighbourhoodAggregator
        n, dim = 5, 8
        embeddings = torch.randn(n, dim)
        protein_to_id = {f"p{i}": i for i in range(n)}
        ppi_edges = [("p0", "p1")]  # only p0 and p1 connected; p2,p3,p4 isolated

        agg = PPINeighbourhoodAggregator(n_hops=1)
        out = agg.aggregate(embeddings, ppi_edges, protein_to_id)

        for isolated_idx in [2, 3, 4]:
            torch.testing.assert_close(out[isolated_idx], embeddings[isolated_idx])

    def test_drug_protein_context_zero_for_no_targets(self):
        from src.encoders.protein_encoder import PPINeighbourhoodAggregator
        n_proteins, dim = 10, 8
        ppi_emb = torch.randn(n_proteins, dim)
        protein_to_id = {f"p{i}": i for i in range(n_proteins)}
        drug_to_id = {"drug_no_targets": 0}
        drug_targets = {}  # drug has no targets

        agg = PPINeighbourhoodAggregator()
        ctx = agg.drug_protein_context(ppi_emb, drug_targets, drug_to_id, protein_to_id)
        assert ctx.shape == (1, dim)
        torch.testing.assert_close(ctx[0], torch.zeros(dim))

    def test_drug_protein_context_correct_mean(self):
        from src.encoders.protein_encoder import PPINeighbourhoodAggregator
        n_proteins, dim = 4, 6
        ppi_emb = torch.randn(n_proteins, dim)
        protein_to_id = {f"p{i}": i for i in range(n_proteins)}
        drug_to_id = {"drugA": 0}
        drug_targets = {"drugA": {"p0", "p1", "p2"}}

        agg = PPINeighbourhoodAggregator()
        ctx = agg.drug_protein_context(ppi_emb, drug_targets, drug_to_id, protein_to_id)
        expected = ppi_emb[:3].mean(dim=0)
        torch.testing.assert_close(ctx[0], expected, atol=1e-5, rtol=1e-5)


class TestProteinFusionEncoder:
    def test_output_shape(self):
        from src.encoders.protein_encoder import ProteinFusionEncoder
        fusion = ProteinFusionEncoder(embedding_dim=64, esm_dim=320)
        self_emb = torch.randn(15, 320)
        nb_emb = torch.randn(15, 320)
        out = fusion(self_emb, nb_emb)
        assert out.shape == (15, 64)


# -----------------------------------------------------------------------
# Stratified evaluation tests
# -----------------------------------------------------------------------

class TestAssignTargetBins:
    def test_no_targets(self):
        from src.evaluation.stratified_eval import assign_target_bins
        drug_targets = {"drugA": set()}
        protein_to_id = {"p1": 0}
        bins = assign_target_bins(drug_targets, protein_to_id)
        assert bins["drugA"] == "none"

    def test_low_targets(self):
        from src.evaluation.stratified_eval import assign_target_bins
        drug_targets = {"drugA": {"p1", "p2"}}
        protein_to_id = {"p1": 0, "p2": 1}
        bins = assign_target_bins(drug_targets, protein_to_id)
        assert bins["drugA"] == "low"

    def test_high_targets(self):
        from src.evaluation.stratified_eval import assign_target_bins
        drug_targets = {"drugA": {f"p{i}" for i in range(15)}}
        protein_to_id = {f"p{i}": i for i in range(15)}
        bins = assign_target_bins(drug_targets, protein_to_id)
        assert bins["drugA"] == "high"

    def test_uncovered_targets_do_not_count(self):
        from src.evaluation.stratified_eval import assign_target_bins
        drug_targets = {"drugA": {"p1", "p2", "p_unknown"}}
        protein_to_id = {"p1": 0}  # only p1 in the PPI graph
        bins = assign_target_bins(drug_targets, protein_to_id)
        assert bins["drugA"] == "low"  # only 1 covered target


class TestAPAtK:
    def test_perfect_ranking(self):
        from src.evaluation.stratified_eval import ap_at_k
        y_true = np.array([1, 0, 0, 0, 0])
        y_score = np.array([0.9, 0.5, 0.4, 0.3, 0.2])
        result = ap_at_k(y_true, y_score, k=50)
        assert result == pytest.approx(1.0)

    def test_worst_ranking(self):
        from src.evaluation.stratified_eval import ap_at_k
        y_true = np.array([1, 0, 0, 0, 0])
        y_score = np.array([0.1, 0.9, 0.8, 0.7, 0.6])
        result = ap_at_k(y_true, y_score, k=50)
        # positive is ranked last among 5 — AP = 1/5
        assert result == pytest.approx(0.2)

    def test_no_positives(self):
        from src.evaluation.stratified_eval import ap_at_k
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.9, 0.5, 0.2])
        result = ap_at_k(y_true, y_score, k=50)
        assert result == 0.0
