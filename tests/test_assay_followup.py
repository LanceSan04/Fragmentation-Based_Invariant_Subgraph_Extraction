import unittest
import torch
from torch_geometric.data import Data
from src.experiments.assay_followup import require_parity
from src.experiments.retention_audit import probe_settings, size_bin
from src.models.fragment_ciga import FragmentCIGA, topk_mask_over_eligible


class FollowupTests(unittest.TestCase):
    def test_parity_rejects_drift_and_nan(self):
        require_parity(.80001,.8)
        for value in [.81,float('nan')]:
            with self.assertRaises(ValueError): require_parity(value,.8)

    def test_probe_only_enables_sampler_and_restores_state(self):
        v=FragmentCIGA(fragment_feature_dim=8,gnn_hidden_dim=8,num_gnn_layers=1,
                       fragment_sample_ratio=.7,sampler_seed=9)
        v.train()
        v.fragment_edge_sampler._call_count.fill_(5)
        graph=Data(x=torch.zeros(11,8),edge_index=torch.stack([torch.arange(10),torch.arange(1,11)]),
                   edge_attr=torch.zeros(10,1),batch=torch.zeros(11,dtype=torch.long))
        with probe_settings(v,.3,.5,17):
            self.assertFalse(v.atom_classifier.training)
            self.assertFalse(v.fragment_masker.training)
            self.assertTrue(v.fragment_edge_sampler.training)
            _,eligible=v.fragment_edge_sampler(graph)
            self.assertEqual(int(eligible.sum()),5)
            mask=topk_mask_over_eligible(torch.arange(10,dtype=torch.float),torch.zeros(10,dtype=torch.long),.7,eligible)
            self.assertEqual(int(mask.sum()),5)  # Full-edge budget is capped by eligibility.
        self.assertTrue(v.training)
        self.assertTrue(v.atom_classifier.training)
        self.assertEqual(v.fragment_edge_sampler.keep_ratio,.7)
        self.assertEqual(v.fragment_edge_sampler.seed,9)
        self.assertEqual(int(v.fragment_edge_sampler._call_count),5)
        self.assertEqual(v.fragment_masker.ratio,.5)
        with probe_settings(v,.5):
            _,eligible=v.fragment_edge_sampler(graph)
            self.assertEqual(int(eligible.sum()),10)

    def test_size_bin_boundaries(self):
        self.assertEqual(size_bin(25),'small_1_25')
        self.assertEqual(size_bin(26),'medium_26_50')
        self.assertEqual(size_bin(50),'medium_26_50')
        self.assertEqual(size_bin(51),'large_51_plus')
