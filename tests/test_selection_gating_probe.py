import unittest
import torch
from torch_geometric.data import Data
from src.experiments.selection_gating_probe import unit_gates


class GateProbeTests(unittest.TestCase):
    def test_intervention_preserves_graph_and_original_weights(self):
        graph = Data(
            x=torch.randn(3, 4),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.randn(2, 2),
            edge_weight=torch.tensor([0.2, 0.2]),
            batch=torch.zeros(3, dtype=torch.long),
            selected_original_atom_indices=torch.tensor([3, 5, 8]),
        )
        result = unit_gates(graph)
        for key in ["x", "edge_index", "edge_attr", "batch", "selected_original_atom_indices"]:
            torch.testing.assert_close(result[key], graph[key])
        torch.testing.assert_close(result.edge_weight, torch.ones(2))
        torch.testing.assert_close(graph.edge_weight, torch.tensor([0.2, 0.2]))

    def test_edgeless_graph_is_supported(self):
        graph = Data(x=torch.ones(1, 2), edge_weight=torch.empty(0))
        self.assertEqual(unit_gates(graph).edge_weight.numel(), 0)
