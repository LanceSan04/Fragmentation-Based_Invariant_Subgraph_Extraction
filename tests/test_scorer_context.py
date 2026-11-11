import unittest
from unittest.mock import patch
import copy
import torch
from tests import test_random_selection_control as fixtures
from src.models.fragment_graph import build_fragment_graph_batch


class ScorerContextTests(unittest.TestCase):
    def test_default_and_full_eval_match_and_sampling_only_limits_eligibility(self):
        f=fixtures.RandomSelectionControlTests();f.setUpClass()
        batch=f.batch([0,1,2]);model=f.view(batch);model.selection_mode='learned'
        full=copy.deepcopy(model);full.scorer_context='full'
        model.eval();full.eval()
        torch.testing.assert_close(model(batch),full(batch))
        fragments=build_fragment_graph_batch(batch)
        full.fragment_edge_sampler.train()
        with patch.object(full.fragment_masker,'forward', wraps=full.fragment_masker.forward) as encoder:
            scores,eligible=full.score_fragment_edges(fragments)
            self.assertEqual(encoder.call_args.args[0].edge_index.size(1),fragments.edge_index.size(1))
            self.assertEqual(scores.numel(),fragments.edge_index.size(1))
            self.assertTrue((scores>0).all())
            self.assertTrue((~eligible).any())

