import unittest
import torch
from torch_geometric.data import Data
from src.models.fragment_masker import (
    FragmentMasker,
    FRAGMENT_EDGE_DIM,
    reverse_fragment_edge_attr,
    segment_topk_mask,
)


class SymmetricScoringTests(unittest.TestCase):
    def test_reconstructed_prediction_and_loss_gradient(self):
        from tests.test_sugar_scorer import SuGArScorerTests
        from torch_geometric.data import Batch
        from src.models.fragment_graph import build_fragment_graph_batch

        fixture = SuGArScorerTests()
        torch.manual_seed(8)
        batch = Batch.from_data_list(
            [fixture.build_data("CCOC(=O)c1ccc(N)cc1", 0)], follow_batch=["fragment_x_static"]
        )
        view = fixture._make_view(batch, scoring_mode="symmetric", edge_ratio=0.5).eval()
        g = build_fragment_graph_batch(batch)
        r = g.clone()
        n = g.x.size(0)
        r.x = g.x.flip(0)
        r.batch = g.batch.flip(0)
        r.edge_index = (n - 1 - g.edge_index).flip(0)
        r.edge_attr = reverse_fragment_edge_attr(g.edge_attr)
        a, ta = view(batch, True, fragment_batch=g)
        b, tb = view(batch, True, fragment_batch=r)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(torch.equal(ta["selected_bond_ids"], tb["selected_bond_ids"]))
        (-a[0, 0]).backward()
        self.assertGreater(
            sum(
                float(p.grad.abs().sum())
                for p in view.fragment_masker.edge_att.parameters()
                if p.grad is not None
            ),
            0.0,
        )

    def test_ties_do_not_depend_on_unrelated_graph(self):
        scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        other = torch.tensor(
            [
                0.0,
                1.0,
                1.0,
                0.0,
                2.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                2.0,
                2.0,
                0.0,
                2.0,
                2.0,
                2.0,
                1.0,
                0.0,
                2.0,
                0.0,
                1.0,
                2.0,
                2.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        alone = segment_topk_mask(scores, torch.zeros(7, dtype=torch.long), 0.5)
        batched = segment_topk_mask(
            torch.cat([other, scores]),
            torch.cat([torch.zeros(len(other), dtype=torch.long), torch.ones(7, dtype=torch.long)]),
            0.5,
        )[len(other) :]
        self.assertTrue(torch.equal(alone, batched))
        self.assertEqual(alone.nonzero().flatten().tolist(), [0, 1, 2])

    def test_symmetric_scores_and_selection_survive_renumbering(self):
        torch.manual_seed(42)
        graph = Data(
            x=torch.randn(5, 7),
            edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 3, 4, 4]]),
            edge_attr=torch.randn(5, FRAGMENT_EDGE_DIM),
            batch=torch.zeros(5, dtype=torch.long),
        )
        model = FragmentMasker(
            7, hidden_dim=8, num_layers=2, edge_dim=FRAGMENT_EDGE_DIM, scoring_mode="symmetric"
        ).eval()
        relabeled = graph.clone()
        relabeled.x = graph.x.flip(0)
        relabeled.edge_index = (4 - graph.edge_index).flip(0)
        relabeled.edge_attr = reverse_fragment_edge_attr(graph.edge_attr)
        a, ma = model(graph)
        b, mb = model(relabeled)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertTrue(torch.equal(ma, mb))
        a.sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.edge_att.parameters()), 0)

    def test_old_state_dict_loads_strictly_into_both_modes(self):
        old = FragmentMasker(7, hidden_dim=8, num_layers=1)
        new = FragmentMasker(7, hidden_dim=8, num_layers=1, scoring_mode="symmetric")
        new.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(old.scoring_mode, "ordered")

    def test_unknown_directional_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            FragmentMasker(7, edge_dim=3, scoring_mode="symmetric")
