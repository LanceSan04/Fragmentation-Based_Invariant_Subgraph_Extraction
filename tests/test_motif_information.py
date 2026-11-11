import unittest
import torch
from src.experiments.motif_check import challenging_fixture, fixture
from src.experiments.motif_information import information_paths
from src.models.fragment_ciga import FragmentCIGA


class InformationPathsTests(unittest.TestCase):
    def test_random_unit_gates_and_batch_preservation(self):
        torch.set_num_threads(1)
        _, held = challenging_fixture()
        batch, _ = fixture(held[:4], 0, 31)
        before = {key: value.clone() for key, value in batch.to_dict().items() if torch.is_tensor(value)}
        model = FragmentCIGA(atom_feature_dim=batch.x.size(1),
            fragment_feature_dim=batch.fragment_x_static.size(1),
            fragment_edge_dim=batch.fragment_edge_attr.size(1), atom_edge_dim=batch.edge_attr.size(1),
            gnn_hidden_dim=8, num_gnn_layers=1, edge_ratio=.25,
            selection_mode="random", sampler_seed=31)
        report = information_paths(model, batch)
        self.assertEqual(report["variants"]["unit_gates_fixed_selection"]["mean_absolute_probability_change"], 0)
        self.assertEqual(len(report["variants"]), 7)
        for key, value in before.items():
            torch.testing.assert_close(batch[key], value)
