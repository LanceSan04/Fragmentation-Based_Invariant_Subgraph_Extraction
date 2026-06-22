"""Tests for learnable fragment-edge scoring in the SuGAr prototype.

Run this file from the repository root with:

    .\.venv\Scripts\python.exe -m unittest tests.test_sugar_scorer -v

Run the full test suite with:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

import unittest

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data
from src.fragmentation import build_fragmenter
from src.models.sugar_ensemble import SuGAREnsemble


class SuGArScorerTests(unittest.TestCase):
    def build_data(self, smiles, label):
        example = build_molecule_example(
            smiles,
            label=label,
            molecule_id=f"scorer_{label}",
            fragmenter=build_fragmenter("group_graph"),
        )
        return molecule_example_to_data(example)

    def test_prediction_loss_backpropagates_to_fragment_edge_scorer(self):
        torch.manual_seed(7)
        data_list = [
            self.build_data("CCOC(=O)C", 0),
            self.build_data("CCN(CC)CC", 1),
        ]
        batch = Batch.from_data_list(data_list, follow_batch=["fragment_x_static"])

        model = SuGAREnsemble(
            atom_feature_dim=batch.x.size(1),
            fragment_feature_dim=batch.fragment_x_static.size(1),
            num_classes=2,
            num_gnn_layers=2,
            gnn_hidden_dim=32,
            edge_ratio=1.0,
            num_views=1,
            dropout=0.0,
            fragment_edge_dim=batch.fragment_edge_attr.size(1),
            atom_edge_dim=batch.edge_attr.size(1),
        )
        model.eval()

        logits, intermediates = model(batch, return_intermediates=True)
        labels = batch.y.view(-1).long()
        loss = F.cross_entropy(logits, labels)

        model.zero_grad()
        loss.backward()

        scorer_grad = 0.0
        for parameter in model.fragment_maskers[0].edge_att.parameters():
            if parameter.grad is not None:
                scorer_grad += float(parameter.grad.abs().sum())

        self.assertGreater(scorer_grad, 0.0)
        self.assertTrue(intermediates["fragment_scores"][0].requires_grad)
        self.assertTrue(intermediates["atom_graphs"][0].edge_weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
