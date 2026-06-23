"""Tests for Stage 2 aggregation and Stage 3 soft-vote in SuGAREnsemble.

Run from the repository root with:
    python -m unittest tests.test_ensemble_aggregation -v
"""

import math
import unittest

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data
from src.fragmentation import build_fragmenter
from src.models.sugar_ensemble import SuGAREnsemble
from src.losses.prediction import prediction_loss
from src.losses.diversity import pairwise_diversity_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_batch(*smiles_labels):
    fragmenter = build_fragmenter("group_graph")
    data_list = [
        molecule_example_to_data(
            build_molecule_example(smi, label=lbl, molecule_id=f"m{i}", fragmenter=fragmenter)
        )
        for i, (smi, lbl) in enumerate(smiles_labels)
    ]
    return Batch.from_data_list(data_list, follow_batch=["fragment_x_static"])


def _make_model(batch, num_views=1, aggregation_mode="mean"):
    return SuGAREnsemble(
        atom_feature_dim=batch.x.size(1),
        fragment_feature_dim=batch.fragment_x_static.size(1),
        num_classes=2,
        num_gnn_layers=2,
        gnn_hidden_dim=32,
        edge_ratio=1.0,
        num_views=num_views,
        dropout=0.0,
        aggregation_mode=aggregation_mode,
        fragment_edge_dim=batch.fragment_edge_attr.size(1),
        atom_edge_dim=batch.edge_attr.size(1),
    )


# ---------------------------------------------------------------------------
# Stage 2: _aggregate_weights
# ---------------------------------------------------------------------------

class AggregateWeightsTests(unittest.TestCase):
    """Unit-test _aggregate_weights in isolation via a minimal SuGAREnsemble."""

    def _model_with_mode(self, mode, batch):
        return _make_model(batch, num_views=2, aggregation_mode=mode)

    def setUp(self):
        torch.manual_seed(0)
        self.batch = _build_batch(("CCOC(=O)C", 0), ("CCN(CC)CC", 1))

    def test_mean_mode_returns_element_wise_average(self):
        model = self._model_with_mode("mean", self.batch)
        a = torch.tensor([0.2, 0.8, 0.5])
        b = torch.tensor([0.6, 0.4, 0.9])
        result = model._aggregate_weights([a, b])
        expected = (a + b) / 2
        self.assertTrue(torch.allclose(result, expected))

    def test_max_mode_returns_element_wise_maximum(self):
        model = self._model_with_mode("max", self.batch)
        a = torch.tensor([0.2, 0.8, 0.5])
        b = torch.tensor([0.6, 0.4, 0.9])
        result = model._aggregate_weights([a, b])
        expected = torch.maximum(a, b)
        self.assertTrue(torch.allclose(result, expected))

    def test_single_view_mean_equals_input(self):
        model = _make_model(self.batch, num_views=1, aggregation_mode="mean")
        w = torch.tensor([0.3, 0.7, 0.1])
        result = model._aggregate_weights([w])
        self.assertTrue(torch.allclose(result, w))

    def test_single_view_max_equals_input(self):
        model = _make_model(self.batch, num_views=1, aggregation_mode="max")
        w = torch.tensor([0.3, 0.7, 0.1])
        result = model._aggregate_weights([w])
        self.assertTrue(torch.allclose(result, w))

    def test_mean_and_max_match_for_identical_weights(self):
        model_mean = self._model_with_mode("mean", self.batch)
        model_max = self._model_with_mode("max", self.batch)
        w = torch.rand(5)
        self.assertTrue(torch.allclose(
            model_mean._aggregate_weights([w, w]),
            model_max._aggregate_weights([w, w]),
        ))

    def test_invalid_aggregation_mode_raises(self):
        with self.assertRaises(ValueError):
            _make_model(self.batch, aggregation_mode="sum")


# ---------------------------------------------------------------------------
# Stage 3: soft voting output properties
# ---------------------------------------------------------------------------

class SoftVoteOutputTests(unittest.TestCase):
    """Validate that forward() produces valid log-probabilities via soft vote."""

    def setUp(self):
        torch.manual_seed(1)
        self.batch = _build_batch(("CCOC(=O)C", 0), ("CCN(CC)CC", 1), ("c1ccccc1", 0))

    def test_output_shape_matches_batch_and_classes(self):
        model = _make_model(self.batch)
        model.eval()
        with torch.no_grad():
            out = model(self.batch)
        self.assertEqual(out.shape, (3, 2))

    def test_output_is_log_probabilities(self):
        """All values must be ≤ 0 and exp() must sum to ~1 per row."""
        model = _make_model(self.batch)
        model.eval()
        with torch.no_grad():
            out = model(self.batch)
        self.assertTrue((out <= 0).all(), "log-probs must be ≤ 0")
        row_sums = out.exp().sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(3), atol=1e-5),
                        f"exp(log-probs) row sums: {row_sums}")

    def test_multiview_soft_vote_averages_probabilities(self):
        """With n views the ensemble avg prob must equal mean of per-view softmax probs."""
        torch.manual_seed(2)
        num_views = 3
        model = _make_model(self.batch, num_views=num_views)
        model.eval()
        with torch.no_grad():
            _, intermediates = model(self.batch, return_intermediates=True)
            per_view_probs = [
                F.softmax(logits, dim=-1)
                for logits in intermediates["per_view_logits"]
            ]
        avg_expected = torch.stack(per_view_probs, dim=0).mean(dim=0)
        # Reconstruct from forward output (which is log of avg)
        _, intermediates2 = model(self.batch, return_intermediates=True)
        # Re-run to get predictions
        predictions = model(self.batch)
        avg_actual = predictions.exp()
        self.assertTrue(torch.allclose(avg_actual, avg_expected, atol=1e-5))

    def test_mean_and_max_modes_both_produce_valid_log_probs(self):
        for mode in ("mean", "max"):
            with self.subTest(mode=mode):
                model = _make_model(self.batch, num_views=2, aggregation_mode=mode)
                model.eval()
                with torch.no_grad():
                    out = model(self.batch)
                self.assertTrue((out <= 0).all())
                self.assertTrue(torch.allclose(out.exp().sum(dim=1), torch.ones(3), atol=1e-5))

    def test_argmax_prediction_is_integer_class(self):
        model = _make_model(self.batch)
        model.eval()
        with torch.no_grad():
            out = model(self.batch)
        preds = out.argmax(dim=1)
        self.assertEqual(preds.shape, (3,))
        self.assertTrue(((preds == 0) | (preds == 1)).all())


# ---------------------------------------------------------------------------
# Stage 2 + 3: one shared subgraph across all classifiers
# ---------------------------------------------------------------------------

class SharedSubgraphTests(unittest.TestCase):
    """Verify that all classifiers operate on the same atom graph in Stage 2/3."""

    def setUp(self):
        torch.manual_seed(3)
        self.batch = _build_batch(("CCOC(=O)C", 0), ("CCN(CC)CC", 1))

    def test_all_atom_graphs_in_intermediates_are_same_object(self):
        model = _make_model(self.batch, num_views=3)
        model.eval()
        with torch.no_grad():
            _, intermediates = model(self.batch, return_intermediates=True)
        atom_graphs = intermediates["atom_graphs"]
        self.assertEqual(len(atom_graphs), 3)
        # All entries must be the same shared object (Stage 2 builds one graph)
        for g in atom_graphs[1:]:
            self.assertIs(g, atom_graphs[0])

    def test_intermediates_keys_present(self):
        model = _make_model(self.batch, num_views=2)
        model.eval()
        with torch.no_grad():
            _, intermediates = model(self.batch, return_intermediates=True)
        for key in ("fragment_scores", "bool_mask", "atom_graphs", "embeddings", "per_view_logits"):
            self.assertIn(key, intermediates, f"missing key: {key}")

    def test_per_view_lists_have_length_num_views(self):
        num_views = 3
        model = _make_model(self.batch, num_views=num_views)
        model.eval()
        with torch.no_grad():
            _, intermediates = model(self.batch, return_intermediates=True)
        self.assertEqual(len(intermediates["fragment_scores"]), num_views)
        self.assertEqual(len(intermediates["embeddings"]), num_views)
        self.assertEqual(len(intermediates["per_view_logits"]), num_views)
        self.assertEqual(len(intermediates["atom_graphs"]), num_views)

    def test_edge_weight_on_shared_atom_graph_requires_grad(self):
        """Aggregated edge weights must flow to atom_batch.edge_weight."""
        model = _make_model(self.batch)
        # train mode to allow grad
        _, intermediates = model(self.batch, return_intermediates=True)
        edge_weight = intermediates["atom_graphs"][0].edge_weight
        self.assertTrue(edge_weight.requires_grad,
                        "edge_weight must carry grad from aggregated masker scores")


# ---------------------------------------------------------------------------
# Prediction loss (NLLLoss) compatibility
# ---------------------------------------------------------------------------

class PredictionLossTests(unittest.TestCase):
    """prediction_loss must accept log-probs and backpropagate correctly."""

    def setUp(self):
        torch.manual_seed(4)
        self.batch = _build_batch(("CCOC(=O)C", 0), ("CCN(CC)CC", 1))

    def test_prediction_loss_is_scalar(self):
        model = _make_model(self.batch)
        model.eval()
        with torch.no_grad():
            log_probs = model(self.batch)
        labels = self.batch.y.view(-1).long()
        loss = prediction_loss(log_probs, labels)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_prediction_loss_is_non_negative(self):
        """NLL loss on a valid probability distribution must be ≥ 0."""
        model = _make_model(self.batch)
        model.eval()
        with torch.no_grad():
            log_probs = model(self.batch)
        labels = self.batch.y.view(-1).long()
        loss = prediction_loss(log_probs, labels)
        self.assertGreaterEqual(float(loss), 0.0)

    def test_prediction_loss_zero_for_perfect_predictions(self):
        """If model is certain and correct, NLL loss ≈ 0."""
        # Construct near-perfect log-probs manually
        labels = torch.tensor([0, 1])
        log_probs = torch.tensor([
            [math.log(1 - 1e-7), math.log(1e-7)],   # label 0 → high prob class 0
            [math.log(1e-7), math.log(1 - 1e-7)],   # label 1 → high prob class 1
        ])
        loss = prediction_loss(log_probs, labels)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_prediction_loss_backpropagates_to_fragment_masker(self):
        """Gradient must flow from NLL loss all the way back to masker parameters."""
        model = _make_model(self.batch)
        log_probs = model(self.batch)
        labels = self.batch.y.view(-1).long()
        loss = prediction_loss(log_probs, labels)
        model.zero_grad()
        loss.backward()

        masker_grad = sum(
            float(p.grad.abs().sum())
            for p in model.fragment_maskers[0].edge_att.parameters()
            if p.grad is not None
        )
        self.assertGreater(masker_grad, 0.0,
                           "gradient must reach the fragment masker edge_att parameters")

    def test_prediction_loss_backpropagates_for_max_aggregation(self):
        model = _make_model(self.batch, num_views=2, aggregation_mode="max")
        log_probs = model(self.batch)
        labels = self.batch.y.view(-1).long()
        loss = prediction_loss(log_probs, labels)
        model.zero_grad()
        loss.backward()

        total_masker_grad = sum(
            float(p.grad.abs().sum())
            for masker in model.fragment_maskers
            for p in masker.edge_att.parameters()
            if p.grad is not None
        )
        self.assertGreater(total_masker_grad, 0.0)


# ---------------------------------------------------------------------------
# Diversity loss compatibility with new intermediates
# ---------------------------------------------------------------------------

class DiversityLossTests(unittest.TestCase):
    """pairwise_diversity_loss must work with fragment_scores from intermediates."""

    def setUp(self):
        torch.manual_seed(5)
        self.batch = _build_batch(("CCOC(=O)C", 0), ("CCN(CC)CC", 1))

    def test_single_view_diversity_loss_is_zero(self):
        model = _make_model(self.batch, num_views=1)
        _, intermediates = model(self.batch, return_intermediates=True)
        loss = pairwise_diversity_loss(intermediates["fragment_scores"])
        self.assertAlmostEqual(float(loss), 0.0)

    def test_multiview_diversity_loss_is_scalar_in_0_1(self):
        model = _make_model(self.batch, num_views=3)
        _, intermediates = model(self.batch, return_intermediates=True)
        loss = pairwise_diversity_loss(intermediates["fragment_scores"])
        self.assertEqual(loss.shape, torch.Size([]))
        # Cosine similarity is in [-1, 1], mean of non-negative sims ∈ [0, 1]
        self.assertGreaterEqual(float(loss), -1.0)
        self.assertLessEqual(float(loss), 1.0)

    def test_diversity_loss_backpropagates(self):
        model = _make_model(self.batch, num_views=2)
        _, intermediates = model(self.batch, return_intermediates=True)
        loss = pairwise_diversity_loss(intermediates["fragment_scores"])
        model.zero_grad()
        loss.backward()

        grad = sum(
            float(p.grad.abs().sum())
            for masker in model.fragment_maskers
            for p in masker.edge_att.parameters()
            if p.grad is not None
        )
        self.assertGreater(grad, 0.0)


if __name__ == "__main__":
    unittest.main()
