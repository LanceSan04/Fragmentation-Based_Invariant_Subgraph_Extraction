import unittest
from copy import deepcopy
import torch
import torch.nn.functional as F
from src.experiments.motif_check import challenging_fixture, fixture
from src.experiments.scorer_paths import ARMS
from src.models.fragment_ciga import FragmentCIGA
from src.models.sugar_ensemble import SuGAREnsemble


class ScorerPathsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        train, _ = challenging_fixture()
        self.batch, _ = fixture(train[:8], 0, 31)

    def model(self, **kwargs):
        b = self.batch
        return FragmentCIGA(atom_feature_dim=b.x.size(1),
            fragment_feature_dim=b.fragment_x_static.size(1),
            fragment_edge_dim=b.fragment_edge_attr.size(1), atom_edge_dim=b.edge_attr.size(1),
            gnn_hidden_dim=8, num_gnn_layers=2, edge_ratio=.25,
            fragment_sample_ratio=1., sampler_seed=31, dropout=0., **kwargs)

    def test_all_learned_arms_have_prediction_loss_scorer_gradients(self):
        for arm, settings in ARMS.items():
            with self.subTest(arm=arm):
                model = self.model(**settings).train()
                loss = F.nll_loss(model(self.batch), self.batch.y.reshape(-1).long())
                loss.backward()
                norm = sum(float(p.grad.norm()) for p in model.fragment_masker.parameters() if p.grad is not None)
                if arm == "random_unit_gates":
                    self.assertEqual(norm, 0)
                else:
                    self.assertGreater(norm, 0)

    def test_st_unit_training_matches_physical_unit_graph_in_eval_normalization(self):
        model = self.model(**ARMS["st_unit_gates"])
        physical = deepcopy(model)
        # Test-only reference uses physical selection and unit reconstruction;
        # it is not a trainable hard-unit study arm.
        physical.selection_estimator = "hard"
        model.train()
        physical.train()
        actual = model(self.batch)
        expected = physical(self.batch)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        for (name, value), (_, reference) in zip(model.named_buffers(), physical.named_buffers()):
            torch.testing.assert_close(value, reference, atol=2e-6, rtol=2e-5)

    def test_invalid_nonlearning_arm_and_shared_policy_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires straight-through"):
            self.model(message_gate_mode="unit")
        with self.assertRaises(ValueError):
            SuGAREnsemble([self.model(**ARMS["st_unit_gates"])], inference_policy="shared")

    def test_random_controls_have_identical_selected_connections(self):
        gated = self.model(**ARMS["random_learned_gates"])
        unit = self.model(**ARMS["random_unit_gates"])
        unit.load_state_dict(gated.state_dict())
        for training in (True, False):
            gated.train(training)
            unit.train(training)
            _, first = gated(self.batch, return_intermediates=True)
            _, second = unit(self.batch, return_intermediates=True)
            self.assertTrue(torch.equal(first["selected_fragment_mask"], second["selected_fragment_mask"]))
