"""Contracts for the paired scorer-only versus feature-sharing experiment."""
import copy
import unittest

import torch

from src.models.fragment_ciga import FragmentCIGA
from src.models.sugar_ensemble import SuGAREnsemble
from tests.test_random_selection_control import RandomSelectionControlTests


class ContextFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        RandomSelectionControlTests.setUpClass()

    def model_batch(self, mode="encoder"):
        batch = RandomSelectionControlTests().batch([0, 1, 2])
        torch.manual_seed(73)
        model = FragmentCIGA(atom_feature_dim=batch.x.size(1),
            fragment_feature_dim=batch.fragment_x_static.size(1),
            fragment_edge_dim=batch.fragment_edge_attr.size(1),
            atom_edge_dim=batch.edge_attr.size(1), gnn_hidden_dim=16,
            num_gnn_layers=1, sampler_seed=17, fragment_sample_ratio=.7,
            fragment_feature_mode="molecular_gnn", classifier_feature_mode=mode).eval()
        return model, batch

    def test_paired_initialization_feature_routing_and_no_cache_mutation(self):
        raw, batch = self.model_batch("raw")
        shared, _ = self.model_batch("encoder")
        for key, value in raw.state_dict().items():
            torch.testing.assert_close(value, shared.state_dict()[key], rtol=0, atol=0)
        original = batch.x.clone()
        atom_h = shared.encode_atom_features(batch)
        graph = raw.get_invariant_subgraph(batch)
        attached = shared.classifier_graph(graph, batch, atom_h)
        ids = graph.selected_original_atom_indices + batch.ptr[graph.batch]
        torch.testing.assert_close(graph.x, batch.x[ids], rtol=0, atol=0)
        torch.testing.assert_close(attached.x, atom_h[ids], rtol=0, atol=0)
        torch.testing.assert_close(batch.x, original, rtol=0, atol=0)
        for policy in ("shared", "independent"):
            torch.testing.assert_close(SuGAREnsemble([shared], inference_policy=policy)(batch),
                                       shared(batch), rtol=1e-5, atol=1e-6)
        restored, _ = self.model_batch()
        restored.load_state_dict(copy.deepcopy(shared.state_dict()))
        torch.testing.assert_close(shared(batch), restored(batch), rtol=0, atol=0)

    def test_direct_classifier_gradient_without_score_or_gate_path(self):
        model, batch = self.model_batch()
        graph = model.get_invariant_subgraph(batch)
        graph.edge_weight = graph.edge_weight.detach()
        graph.x = graph.x.detach()
        atom_h = model.encode_atom_features(batch)
        atom_h.retain_grad()
        graph = model.classifier_graph(graph, batch, atom_h)
        model.atom_classifier(graph).square().sum().backward()
        self.assertGreater(float(atom_h.grad.abs().sum()), 0)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in
            model.fragment_feature_encoder.parameters() if p.grad is not None), 0)

    def test_scorer_only_encoder_still_learns_through_gates(self):
        model, batch = self.model_batch("raw")
        loss = -model(batch)[:, 1].mean()
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in
            model.fragment_feature_encoder.parameters() if p.grad is not None), 0)

    def test_two_view_shared_uses_each_views_own_embeddings(self):
        a, batch = self.model_batch()
        b, _ = self.model_batch()
        with torch.no_grad():
            b.fragment_feature_encoder.output.bias.add_(.3)
        ensemble = SuGAREnsemble([a, b], inference_policy="shared").eval()
        graph, _ = ensemble._aggregate_subgraph(batch, [a, b], fast=True)
        expected = torch.stack([torch.softmax(v.atom_classifier(v.classifier_graph(graph, batch)), -1)
                                for v in (a, b)]).mean(0)
        torch.testing.assert_close(ensemble(batch).exp(), expected)

    def test_molecular_context_crosses_fragment_boundary(self):
        from torch_geometric.data import Data
        from src.models.fragment_feature_encoder import InternalFragmentEncoder
        torch.manual_seed(77)
        encoder = InternalFragmentEncoder(3, 2, 8, 2, context="molecular").eval()
        graph = Data(x=torch.randn(3, 3), edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
                     edge_attr=torch.randn(4, 2))
        assignment = torch.tensor([0,0,1])
        first = encoder.encode_atoms(graph, assignment)
        changed = graph.clone(); changed.x[2] += torch.tensor([10., -7., 2.])
        second = encoder.encode_atoms(changed, assignment)
        self.assertGreater(float((first[1]-second[1]).abs().sum()), 1e-6)


if __name__ == "__main__":
    unittest.main()
