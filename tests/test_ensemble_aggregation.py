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
from src.models.fragment_ciga import FragmentCIGA
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


def _build_multifragment_batch(*smiles_labels):
    primary = build_fragmenter("group_graph")
    extra = {
        "group_graph": build_fragmenter("group_graph"),
        "brics": build_fragmenter("brics"),
    }
    data_list = [
        molecule_example_to_data(
            build_molecule_example(
                smi,
                label=lbl,
                molecule_id=f"m{i}",
                fragmenter=primary,
                extra_fragmenters=extra,
            )
        )
        for i, (smi, lbl) in enumerate(smiles_labels)
    ]
    return Batch.from_data_list(
        data_list,
        follow_batch=[
            "fragment_x_static",
            "group_graph_fragment_x_static",
            "brics_fragment_x_static",
        ],
    )


def _make_views(batch, num_views=1):
    return [
        FragmentCIGA(
            atom_feature_dim=batch.x.size(1),
            fragment_feature_dim=batch.fragment_x_static.size(1),
            num_classes=2,
            num_gnn_layers=2,
            gnn_hidden_dim=32,
            edge_ratio=1.0,
            dropout=0.0,
            fragment_edge_dim=batch.fragment_edge_attr.size(1),
            atom_edge_dim=batch.edge_attr.size(1),
        )
        for _ in range(num_views)
    ]


def _make_ensemble(batch, num_views=1, aggregation_mode="mean"):
    return SuGAREnsemble(_make_views(batch, num_views), aggregation_mode=aggregation_mode)


# ---------------------------------------------------------------------------
# Stage 2: _aggregate_weights
# ---------------------------------------------------------------------------

class AggregateWeightsTests(unittest.TestCase):
    """Unit-test _aggregate_weights in isolation via a minimal SuGAREnsemble."""

    def setUp(self):
        torch.manual_seed(0)
        self.batch = _build_batch(("CCOC(=O)C", 0), ("CCN(CC)CC", 1))

    def test_mean_mode_returns_element_wise_average(self):
        model = _make_ensemble(self.batch, num_views=2, aggregation_mode="mean")
        a = torch.tensor([0.2, 0.8, 0.5])
        b = torch.tensor([0.6, 0.4, 0.9])
        result = model._aggregate_weights([a, b])
        expected = (a + b) / 2
        self.assertTrue(torch.allclose(result, expected))

    def test_max_mode_returns_element_wise_maximum(self):
        model = _make_ensemble(self.batch, num_views=2, aggregation_mode="max")
        a = torch.tensor([0.2, 0.8, 0.5])
        b = torch.tensor([0.6, 0.4, 0.9])
        result = model._aggregate_weights([a, b])
        expected = torch.maximum(a, b)
        self.assertTrue(torch.allclose(result, expected))

    def test_single_view_mean_equals_input(self):
        model = _make_ensemble(self.batch, num_views=1, aggregation_mode="mean")
        w = torch.tensor([0.3, 0.7, 0.1])
        result = model._aggregate_weights([w])
        self.assertTrue(torch.allclose(result, w))

    def test_single_view_max_equals_input(self):
        model = _make_ensemble(self.batch, num_views=1, aggregation_mode="max")
        w = torch.tensor([0.3, 0.7, 0.1])
        result = model._aggregate_weights([w])
        self.assertTrue(torch.allclose(result, w))

    def test_mean_and_max_match_for_identical_weights(self):
        model_mean = _make_ensemble(self.batch, num_views=2, aggregation_mode="mean")
        model_max = _make_ensemble(self.batch, num_views=2, aggregation_mode="max")
        w = torch.rand(5)
        self.assertTrue(torch.allclose(
            model_mean._aggregate_weights([w, w]),
            model_max._aggregate_weights([w, w]),
        ))

    def test_invalid_aggregation_mode_raises(self):
        with self.assertRaises(ValueError):
            _make_ensemble(self.batch, aggregation_mode="sum")


# ---------------------------------------------------------------------------
# Stage 3: soft voting output properties
# ---------------------------------------------------------------------------

class SoftVoteOutputTests(unittest.TestCase):
    """Validate that __call__ produces valid log-probabilities via soft vote."""

    def setUp(self):
        torch.manual_seed(1)
        self.batch = _build_batch(("CCOC(=O)C", 0), ("CCN(CC)CC", 1), ("c1ccccc1", 0))

    def test_output_shape_matches_batch_and_classes(self):
        model = _make_ensemble(self.batch)
        model.eval()
        with torch.no_grad():
            out = model(self.batch)
        self.assertEqual(out.shape, (3, 2))

    def test_output_is_log_probabilities(self):
        """All values must be ≤ 0 and exp() must sum to ~1 per row."""
        model = _make_ensemble(self.batch)
        model.eval()
        with torch.no_grad():
            out = model(self.batch)
        self.assertTrue((out <= 0).all(), "log-probs must be ≤ 0")
        row_sums = out.exp().sum(dim=1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(3), atol=1e-5),
                        f"exp(log-probs) row sums: {row_sums}")

    def test_mean_and_max_modes_both_produce_valid_log_probs(self):
        for mode in ("mean", "max"):
            with self.subTest(mode=mode):
                model = _make_ensemble(self.batch, num_views=2, aggregation_mode=mode)
                model.eval()
                with torch.no_grad():
                    out = model(self.batch)
                self.assertTrue((out <= 0).all())
                self.assertTrue(torch.allclose(out.exp().sum(dim=1), torch.ones(3), atol=1e-5))

    def test_argmax_prediction_is_integer_class(self):
        model = _make_ensemble(self.batch)
        model.eval()
        with torch.no_grad():
            out = model(self.batch)
        preds = out.argmax(dim=1)
        self.assertEqual(preds.shape, (3,))
        self.assertTrue(((preds == 0) | (preds == 1)).all())

    def test_view_indices_subset_produces_valid_output(self):
        """Passing view_indices should restrict to that subset without error."""
        model = _make_ensemble(self.batch, num_views=4)
        model.eval()
        with torch.no_grad():
            out = model(self.batch, view_indices=[1, 3])
        self.assertEqual(out.shape, (3, 2))
        self.assertTrue((out <= 0).all())

    def test_mixed_fragmentation_views_soft_vote_without_edge_id_alignment(self):
        batch = _build_multifragment_batch(
            ("CC(=O)NCCOC(=O)c1ccccc1", 0),
            ("CCOC(=O)NCC", 1),
        )
        views = [
            FragmentCIGA(
                atom_feature_dim=batch.x.size(1),
                fragment_feature_dim=batch.group_graph_fragment_x_static.size(1),
                num_classes=2,
                num_gnn_layers=1,
                gnn_hidden_dim=16,
                edge_ratio=1.0,
                dropout=0.0,
                fragment_edge_dim=batch.group_graph_fragment_edge_attr.size(1),
                atom_edge_dim=batch.edge_attr.size(1),
                fragmentation_method="group_graph",
            ),
            FragmentCIGA(
                atom_feature_dim=batch.x.size(1),
                fragment_feature_dim=batch.brics_fragment_x_static.size(1),
                num_classes=2,
                num_gnn_layers=1,
                gnn_hidden_dim=16,
                edge_ratio=1.0,
                dropout=0.0,
                fragment_edge_dim=batch.brics_fragment_edge_attr.size(1),
                atom_edge_dim=batch.edge_attr.size(1),
                fragmentation_method="brics",
            ),
        ]
        model = SuGAREnsemble(views)
        model.eval()

        with torch.no_grad():
            out = model(batch)

        self.assertEqual(out.shape, (2, 2))
        self.assertTrue(torch.allclose(out.exp().sum(dim=1), torch.ones(2), atol=1e-5))

    def test_empty_view_indices_raises(self):
        model = _make_ensemble(self.batch, num_views=2)
        with self.assertRaises(ValueError):
            model(self.batch, view_indices=[])

    def test_num_views_property(self):
        for n in (1, 3, 5):
            model = _make_ensemble(self.batch, num_views=n)
            self.assertEqual(model.num_views, n)


# ---------------------------------------------------------------------------
# Prediction loss (NLLLoss) compatibility
# ---------------------------------------------------------------------------

class PredictionLossTests(unittest.TestCase):
    """prediction_loss must accept log-probs from the ensemble and backpropagate."""

    def setUp(self):
        torch.manual_seed(4)
        self.batch = _build_batch(("CCOC(=O)C", 0), ("CCN(CC)CC", 1))

    def test_prediction_loss_is_scalar(self):
        model = _make_ensemble(self.batch)
        model.eval()
        with torch.no_grad():
            log_probs = model(self.batch)
        labels = self.batch.y.view(-1).long()
        loss = prediction_loss(log_probs, labels)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_prediction_loss_is_non_negative(self):
        model = _make_ensemble(self.batch)
        model.eval()
        with torch.no_grad():
            log_probs = model(self.batch)
        labels = self.batch.y.view(-1).long()
        loss = prediction_loss(log_probs, labels)
        self.assertGreaterEqual(float(loss), 0.0)

    def test_prediction_loss_zero_for_perfect_predictions(self):
        """If model is certain and correct, NLL loss ≈ 0."""
        labels = torch.tensor([0, 1])
        log_probs = torch.tensor([
            [math.log(1 - 1e-7), math.log(1e-7)],
            [math.log(1e-7), math.log(1 - 1e-7)],
        ])
        loss = prediction_loss(log_probs, labels)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_prediction_loss_backpropagates_to_fragment_masker(self):
        """Gradient must flow from NLL loss back to each view's masker parameters."""
        model = _make_ensemble(self.batch)
        log_probs = model(self.batch)
        labels = self.batch.y.view(-1).long()
        loss = prediction_loss(log_probs, labels)
        for v in model.views:
            v.zero_grad()
        loss.backward()

        masker_grad = sum(
            float(p.grad.abs().sum())
            for p in model.views[0].fragment_masker.edge_att.parameters()
            if p.grad is not None
        )
        self.assertGreater(masker_grad, 0.0,
                           "gradient must reach the fragment masker edge_att parameters")

    def test_prediction_loss_backpropagates_for_max_aggregation(self):
        model = _make_ensemble(self.batch, num_views=2, aggregation_mode="max")
        log_probs = model(self.batch)
        labels = self.batch.y.view(-1).long()
        loss = prediction_loss(log_probs, labels)
        for v in model.views:
            v.zero_grad()
        loss.backward()

        total_masker_grad = sum(
            float(p.grad.abs().sum())
            for v in model.views
            for p in v.fragment_masker.edge_att.parameters()
            if p.grad is not None
        )
        self.assertGreater(total_masker_grad, 0.0)


# ---------------------------------------------------------------------------
# Diversity loss compatibility with FragmentCIGA intermediates
# ---------------------------------------------------------------------------

class DiversityLossTests(unittest.TestCase):
    """pairwise_diversity_loss must work with fragment_scores from FragmentCIGA."""

    def setUp(self):
        torch.manual_seed(5)
        self.batch = _build_batch(("CCOC(=O)C", 0), ("CCN(CC)CC", 1))

    def _make_views(self, num_views):
        return _make_views(self.batch, num_views)

    def test_single_view_diversity_loss_is_zero(self):
        views = self._make_views(1)
        views[0].train()
        _, interm = views[0](self.batch, return_intermediates=True)
        loss = pairwise_diversity_loss(
            [interm["fragment_scores"]],
            edge_batch=interm["fragment_edge_batch"],
            eligible_masks=[interm["fragment_edge_eligible"]],
            num_graphs=self.batch.num_graphs,
        )
        self.assertAlmostEqual(float(loss), 0.0)

    def test_multiview_diversity_loss_is_scalar_in_0_1(self):
        views = self._make_views(3)
        fragment_scores = []
        eligible_masks = []
        edge_batch = None
        for v in views:
            v.train()
            _, interm = v(self.batch, return_intermediates=True)
            fragment_scores.append(interm["fragment_scores"])
            eligible_masks.append(interm["fragment_edge_eligible"])
            edge_batch = interm["fragment_edge_batch"]
        loss = pairwise_diversity_loss(
            fragment_scores,
            edge_batch=edge_batch,
            eligible_masks=eligible_masks,
            num_graphs=self.batch.num_graphs,
        )
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreaterEqual(float(loss), 0.0)
        self.assertLessEqual(float(loss), 1.0)

    def test_diversity_loss_backpropagates_to_active_scores(self):
        score_a = torch.tensor([0.1, 0.4, 0.9], requires_grad=True)
        score_b = torch.tensor([0.3, 0.2, 0.6], requires_grad=True)

        loss = pairwise_diversity_loss(
            [score_a, score_b],
            edge_batch=torch.tensor([0, 0, 0]),
            num_graphs=1,
        )
        loss.backward()

        grad = float(score_a.grad.abs().sum() + score_b.grad.abs().sum())
        self.assertGreater(grad, 0.0)


if __name__ == "__main__":
    unittest.main()
