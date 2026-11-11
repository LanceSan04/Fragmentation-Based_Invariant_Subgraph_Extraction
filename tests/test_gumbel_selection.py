import copy
import unittest
from unittest.mock import patch
import torch
from src.models.learnable_selection import gumbel_ranks, budget_soft_mask
from tests import test_learnable_selection as fixtures


class GumbelSelectionTests(unittest.TestCase):
    def test_full_model_dropped_scores_get_selection_gradients(self):
        batch, model = fixtures.LearnableSelectionTests().fixture()
        model.selection_estimator = "gumbel_st"
        model.message_gate_mode = "unit"
        model.train()
        original = model.score_fragment_edges
        captured = []
        def score(graph):
            scores, eligible = original(graph)
            scores.retain_grad()
            captured.append(scores)
            return scores, eligible
        with patch.object(model, "score_fragment_edges", side_effect=score):
            pred, trace = model(batch, return_intermediates=True)
            torch.nn.functional.nll_loss(pred, batch.y.long()).backward()
        dropped = ~trace["selected_fragment_mask"] & trace["fragment_edge_eligible"]
        self.assertTrue(dropped.any())
        self.assertGreater(float(captured[0].grad[dropped].abs().sum()), 0)

    def test_noisy_hard_forward_matches_physical_graph(self):
        batch, model = fixtures.LearnableSelectionTests().fixture()
        model.selection_estimator = "gumbel_st"
        physical = copy.deepcopy(model)
        physical.selection_estimator = "hard"
        model.train(); physical.train()
        actual, trace = model(batch, return_intermediates=True)
        with patch.object(physical, "select_fragment_edges", return_value=trace["selected_fragment_mask"]):
            expected = physical(batch)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_seed_rng_and_eligible_gradients(self):
        scores = torch.tensor([.2, .4, .6, .8], requires_grad=True)
        state = torch.random.get_rng_state()
        ranks = gumbel_ranks(scores, 44)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(ranks, gumbel_ranks(scores, 44)))
        self.assertFalse(torch.equal(ranks, gumbel_ranks(scores, 45)))
        soft = budget_soft_mask(ranks, torch.zeros(4, dtype=torch.long),
            torch.tensor([False, True, False, False]), torch.tensor([True, True, True, False]), .5)
        self.assertAlmostEqual(float(soft.sum()), 1., places=5)
        soft[0].backward()
        self.assertGreater(float(scores.grad[0].abs()), 0)
        self.assertEqual(float(scores.grad[3]), 0.)

    def test_eval_matches_hard_and_training_has_scorer_gradient(self):
        batch, hard = fixtures.LearnableSelectionTests().fixture()
        model = copy.deepcopy(hard)
        model.selection_estimator = "gumbel_st"
        hard.eval(); model.eval()
        torch.testing.assert_close(model(batch), hard(batch))
        model.train()
        loss = torch.nn.functional.nll_loss(model(batch), batch.y.long())
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.fragment_masker.parameters()))
        self.assertTrue(torch.isfinite(loss))
