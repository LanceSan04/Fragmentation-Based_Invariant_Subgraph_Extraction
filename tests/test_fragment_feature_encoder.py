import copy
import unittest
import torch
from src.models.fragment_feature_encoder import InternalFragmentEncoder
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.fragment_ciga import FragmentCIGA
from src.models.sugar_ensemble import SuGAREnsemble
from tests.test_random_selection_control import RandomSelectionControlTests


class FragmentFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        RandomSelectionControlTests.setUpClass()

    def test_fragment_locality_and_single_atom(self):
        from torch_geometric.data import Data
        encoder = InternalFragmentEncoder(3, 2, 8, 2).eval()
        # Two fragments, with a boundary bond that must never enter the encoder.
        batch = Data(x=torch.randn(3, 3), edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
                     edge_attr=torch.randn(4, 2))
        assignment = torch.tensor([0,0,1])
        static = torch.randn(2, 7)
        a = encoder(batch, assignment, static)
        changed = batch.clone(); changed.x[2] += 100
        b = encoder(changed, assignment, static)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        torch.testing.assert_close(a[:, 3:], static[:, 3:], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(a).all())
        a.sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in encoder.parameters()
                               if p.grad is not None), 0)

    def test_default_identity_and_trainable_both_inference_paths(self):
        f = RandomSelectionControlTests(); batch = f.batch([0,1,2])
        kwargs = dict(atom_feature_dim=batch.x.size(1), fragment_feature_dim=batch.fragment_x_static.size(1),
                      fragment_edge_dim=batch.fragment_edge_attr.size(1), atom_edge_dim=batch.edge_attr.size(1),
                      gnn_hidden_dim=16, num_gnn_layers=1, sampler_seed=7)
        torch.manual_seed(33); original = FragmentCIGA(**kwargs).eval()
        torch.manual_seed(33); explicit = FragmentCIGA(**kwargs, fragment_feature_mode='static_pool').eval()
        self.assertEqual(list(original.state_dict()), list(explicit.state_dict()))
        torch.testing.assert_close(original(batch), explicit(batch), rtol=0, atol=0)
        model = FragmentCIGA(**kwargs, fragment_feature_mode='internal_gnn').eval()
        raw = batch.fragment_x_static.clone()
        prediction = model(batch)
        (-prediction[:, 1].mean()).backward()
        grads = [p.grad for p in model.fragment_feature_encoder.parameters() if p.grad is not None]
        self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
        for policy in ('shared', 'independent'):
            torch.testing.assert_close(SuGAREnsemble([model], inference_policy=policy)(batch), prediction)
        self.assertEqual(model.get_invariant_subgraph(batch).num_graphs, 3)
        torch.testing.assert_close(raw, batch.fragment_x_static, rtol=0, atol=0)
        restored = FragmentCIGA(**kwargs, fragment_feature_mode='internal_gnn').eval()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        torch.testing.assert_close(model(batch), restored(batch), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
