"""
Tests for data splitting and evaluation protocols.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from collections import defaultdict


# ---------------------------------------------------------------------------
# Splitting tests
# ---------------------------------------------------------------------------

class TestSplitPolypharmacyEdges:

    def _make_triples(self, n_se=5, edges_per_se=100):
        """Generate synthetic polypharmacy triples."""
        triples = []
        for se_idx in range(n_se):
            se = f"C{se_idx:04d}"
            for i in range(edges_per_se):
                triples.append((f"drug_{i}", se, f"drug_{i+1}"))
        return triples

    def test_split_produces_three_sets(self):
        from src.data.splitting import split_polypharmacy_edges
        triples = self._make_triples()
        train, val, test = split_polypharmacy_edges(triples)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_total_equals_input(self):
        from src.data.splitting import split_polypharmacy_edges
        triples = self._make_triples()
        train, val, test = split_polypharmacy_edges(triples)
        assert len(train) + len(val) + len(test) == len(triples)

    def test_no_overlap_between_splits(self):
        from src.data.splitting import split_polypharmacy_edges
        triples = self._make_triples()
        train, val, test = split_polypharmacy_edges(triples)
        train_set = set(map(tuple, train))
        val_set = set(map(tuple, val))
        test_set = set(map(tuple, test))
        assert len(train_set & val_set) == 0, "Train/val overlap"
        assert len(train_set & test_set) == 0, "Train/test overlap"
        assert len(val_set & test_set) == 0, "Val/test overlap"

    def test_split_is_per_se_type(self):
        """Every SE type present in test should also be in train."""
        from src.data.splitting import split_polypharmacy_edges
        triples = self._make_triples(n_se=10, edges_per_se=200)
        train, val, test = split_polypharmacy_edges(triples)

        train_ses = {r for _, r, _ in train}
        test_ses = {r for _, r, _ in test}
        # All test SE types should appear in training (otherwise model never sees them)
        assert test_ses.issubset(train_ses), (
            f"SE types in test but not train: {test_ses - train_ses}"
        )

    def test_approximate_split_ratio(self):
        from src.data.splitting import split_polypharmacy_edges
        triples = self._make_triples(n_se=10, edges_per_se=300)
        train, val, test = split_polypharmacy_edges(triples, val_frac=0.1, test_frac=0.1)
        total = len(triples)
        assert abs(len(val) / total - 0.1) < 0.05
        assert abs(len(test) / total - 0.1) < 0.05

    def test_small_se_types_kept_in_train(self):
        """SE types with fewer than min_edges_per_se should stay entirely in train."""
        from src.data.splitting import split_polypharmacy_edges
        # Make one small SE type with only 10 edges
        small_triples = [(f"drug_{i}", "SMALL_SE", f"drug_{i+1}") for i in range(10)]
        large_triples = [(f"drug_{i}", "LARGE_SE", f"drug_{i+1}") for i in range(200)]
        all_triples = small_triples + large_triples

        train, val, test = split_polypharmacy_edges(all_triples, min_edges_per_se=50)

        # SMALL_SE should not appear in val or test
        val_ses = {r for _, r, _ in val}
        test_ses = {r for _, r, _ in test}
        assert "SMALL_SE" not in val_ses
        assert "SMALL_SE" not in test_ses

    def test_reproducibility(self):
        from src.data.splitting import split_polypharmacy_edges
        triples = self._make_triples()
        train1, val1, test1 = split_polypharmacy_edges(triples, random_state=42)
        train2, val2, test2 = split_polypharmacy_edges(triples, random_state=42)
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2

    def test_different_seeds_give_different_splits(self):
        from src.data.splitting import split_polypharmacy_edges
        triples = self._make_triples(n_se=5, edges_per_se=200)
        _, _, test1 = split_polypharmacy_edges(triples, random_state=0)
        _, _, test2 = split_polypharmacy_edges(triples, random_state=99)
        assert set(map(tuple, test1)) != set(map(tuple, test2))


class TestBuildTrueEdgeSet:
    def test_symmetric(self):
        """True edge set should include both (i,j) and (j,i)."""
        from src.data.splitting import build_true_edge_set
        triples = [("d1", "SE1", "d2"), ("d3", "SE1", "d4")]
        lookup = build_true_edge_set(triples)
        assert ("d1", "d2") in lookup["SE1"]
        assert ("d2", "d1") in lookup["SE1"]  # symmetric

    def test_different_se_types_separated(self):
        from src.data.splitting import build_true_edge_set
        triples = [("d1", "SE1", "d2"), ("d1", "SE2", "d3")]
        lookup = build_true_edge_set(triples)
        assert "SE1" in lookup
        assert "SE2" in lookup
        assert ("d1", "d3") not in lookup["SE1"]
        assert ("d1", "d2") not in lookup["SE2"]


class TestGenerateFalseEdges:
    def test_length_matches_positives(self):
        from src.data.splitting import generate_false_edges, build_true_edge_set
        positives = [("d1", "SE1", "d2"), ("d3", "SE1", "d4")]
        all_drugs = [f"d{i}" for i in range(20)]
        lookup = build_true_edge_set(positives)
        false_edges = generate_false_edges(positives, all_drugs, lookup, n_false_per_positive=1)
        assert len(false_edges) == len(positives)

    def test_false_edges_not_true_positives(self):
        from src.data.splitting import generate_false_edges, build_true_edge_set
        positives = [("d1", "SE1", "d2")]
        all_drugs = [f"d{i}" for i in range(50)]
        lookup = build_true_edge_set(positives)
        false_edges = generate_false_edges(
            positives, all_drugs, lookup, n_false_per_positive=5, random_state=0
        )
        for h, r, t in false_edges:
            assert (h, t) not in lookup[r], f"False edge ({h},{t}) is a true positive for {r}"

    def test_same_head_and_relation(self):
        from src.data.splitting import generate_false_edges, build_true_edge_set
        positives = [("d1", "SE_X", "d2")]
        all_drugs = [f"d{i}" for i in range(20)]
        lookup = build_true_edge_set(positives)
        false_edges = generate_false_edges(positives, all_drugs, lookup)
        for h, r, t in false_edges:
            assert h == "d1"
            assert r == "SE_X"

    def test_n_false_per_positive(self):
        from src.data.splitting import generate_false_edges, build_true_edge_set
        positives = [("d1", "SE1", "d2"), ("d3", "SE1", "d4")]
        all_drugs = [f"d{i}" for i in range(100)]
        lookup = build_true_edge_set(positives)
        false_edges = generate_false_edges(positives, all_drugs, lookup, n_false_per_positive=5)
        assert len(false_edges) == len(positives) * 5


# ---------------------------------------------------------------------------
# Evaluation protocol tests
# ---------------------------------------------------------------------------

class _MockModel:
    """Minimal mock PyKEEN model that scores triples by tail entity index."""
    def eval(self):
        return self

    def score_hrt(self, hrt_tensor):
        import torch
        # Score = tail id (just to have a deterministic scorer)
        return hrt_tensor[:, 2].float().unsqueeze(-1)


class TestEvaluateFalseEdgeProtocol:
    def _make_inputs(self, n_se=3, n_pos_per_se=20):
        """Build minimal test + false triples and id maps."""
        drugs = [f"drug:d{i}" for i in range(50)]
        entity_to_id = {d: i for i, d in enumerate(drugs)}
        ses = [f"SE:se{j}" for j in range(n_se)]
        relation_to_id = {r: i for i, r in enumerate(ses)}

        test_triples = []
        false_triples = []
        for r in ses:
            for k in range(n_pos_per_se):
                h = drugs[k % len(drugs)]
                t_pos = drugs[(k + 1) % len(drugs)]
                t_neg = drugs[(k + 2) % len(drugs)]
                test_triples.append((h, r, t_pos))
                false_triples.append((h, r, t_neg))

        return test_triples, false_triples, entity_to_id, relation_to_id

    def test_returns_dataframe_with_correct_columns(self):
        from src.evaluation.protocols import evaluate_false_edge_protocol
        test, false, e2id, r2id = self._make_inputs()
        model = _MockModel()
        df = evaluate_false_edge_protocol(model, test, false, e2id, r2id, device="cpu")
        assert set(["side_effect", "n_pos", "auroc", "auprc", "ap50"]).issubset(df.columns)

    def test_one_row_per_se_type(self):
        from src.evaluation.protocols import evaluate_false_edge_protocol
        test, false, e2id, r2id = self._make_inputs(n_se=5)
        model = _MockModel()
        df = evaluate_false_edge_protocol(model, test, false, e2id, r2id, device="cpu")
        assert len(df) == 5

    def test_raises_on_length_mismatch(self):
        from src.evaluation.protocols import evaluate_false_edge_protocol
        test, false, e2id, r2id = self._make_inputs()
        model = _MockModel()
        with pytest.raises(AssertionError):
            evaluate_false_edge_protocol(model, test, false[:-1], e2id, r2id, device="cpu")

    def test_metrics_in_valid_range(self):
        from src.evaluation.protocols import evaluate_false_edge_protocol
        test, false, e2id, r2id = self._make_inputs(n_pos_per_se=50)
        model = _MockModel()
        df = evaluate_false_edge_protocol(model, test, false, e2id, r2id, device="cpu")
        valid = df.dropna(subset=["auroc", "auprc"])
        assert (valid["auroc"].between(0, 1)).all()
        assert (valid["auprc"].between(0, 1)).all()
        assert (valid["ap50"].between(0, 1)).all()


class TestEvaluateSampledNegativesProtocol:
    def _make_inputs(self, n_se=3, n_pos_per_se=10):
        drugs = [f"drug:d{i}" for i in range(50)]
        entity_to_id = {d: i for i, d in enumerate(drugs)}
        ses = [f"SE:se{j}" for j in range(n_se)]
        relation_to_id = {r: i for i, r in enumerate(ses)}

        test_triples = []
        from src.data.splitting import build_true_edge_set
        for r in ses:
            for k in range(n_pos_per_se):
                h = drugs[k % len(drugs)]
                t = drugs[(k + 1) % len(drugs)]
                test_triples.append((h, r, t))

        true_lookup = build_true_edge_set(test_triples)
        return test_triples, drugs, true_lookup, entity_to_id, relation_to_id

    def test_returns_correct_columns(self):
        from src.evaluation.protocols import evaluate_sampled_negatives_protocol
        test, drugs, lookup, e2id, r2id = self._make_inputs()
        model = _MockModel()
        df = evaluate_sampled_negatives_protocol(
            model, test, drugs, lookup, e2id, r2id, n_negatives=10, device="cpu"
        )
        assert set(["side_effect", "n_pos", "auroc", "auprc", "ap50"]).issubset(df.columns)

    def test_one_row_per_se_type(self):
        from src.evaluation.protocols import evaluate_sampled_negatives_protocol
        test, drugs, lookup, e2id, r2id = self._make_inputs(n_se=4)
        model = _MockModel()
        df = evaluate_sampled_negatives_protocol(
            model, test, drugs, lookup, e2id, r2id, n_negatives=10, device="cpu"
        )
        assert len(df) == 4


class TestAssignCoverageBins:
    def test_none_bin(self):
        from src.evaluation.protocols import assign_coverage_bins
        drug_targets = {"drugA": set()}
        protein_to_id = {"p1": 0}
        bins = assign_coverage_bins(drug_targets, protein_to_id)
        assert bins["drugA"] == "none"

    def test_low_bin(self):
        from src.evaluation.protocols import assign_coverage_bins
        drug_targets = {"drugA": {"p1", "p2"}}
        protein_to_id = {"p1": 0, "p2": 1}
        bins = assign_coverage_bins(drug_targets, protein_to_id)
        assert bins["drugA"] == "low"

    def test_medium_bin(self):
        from src.evaluation.protocols import assign_coverage_bins
        drug_targets = {"drugA": {f"p{i}" for i in range(5)}}
        protein_to_id = {f"p{i}": i for i in range(5)}
        bins = assign_coverage_bins(drug_targets, protein_to_id)
        assert bins["drugA"] == "medium"

    def test_high_bin(self):
        from src.evaluation.protocols import assign_coverage_bins
        drug_targets = {"drugA": {f"p{i}" for i in range(15)}}
        protein_to_id = {f"p{i}": i for i in range(15)}
        bins = assign_coverage_bins(drug_targets, protein_to_id)
        assert bins["drugA"] == "high"

    def test_uncovered_targets_excluded(self):
        """Proteins not in the PPI graph don't count toward coverage."""
        from src.evaluation.protocols import assign_coverage_bins
        drug_targets = {"drugA": {"p1", "p_not_in_ppi"}}
        protein_to_id = {"p1": 0}  # p_not_in_ppi absent
        bins = assign_coverage_bins(drug_targets, protein_to_id)
        assert bins["drugA"] == "low"  # only 1 covered target
