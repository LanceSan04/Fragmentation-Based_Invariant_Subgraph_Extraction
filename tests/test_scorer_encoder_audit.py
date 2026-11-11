"""Independent checks of molecular embeddings routed only into the scorer."""
import unittest
import torch
from torch_geometric.data import Data

from src.models.fragment_feature_encoder import InternalFragmentEncoder
from src.models.fragment_graph import build_fragment_graph_batch
from tests.test_context_features import ContextFeatureTests


class ScorerEncoderAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ContextFeatureTests.setUpClass()

    def test_optimizer_contains_and_updates_encoder_from_prediction_loss(self):
        model,batch=ContextFeatureTests().model_batch('raw')
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        ids={id(p) for g in optimizer.param_groups for p in g['params']}
        before={k:p.detach().clone() for k,p in model.fragment_feature_encoder.named_parameters()}
        optimizer.zero_grad();(-model(batch)[:,1].mean()).backward()
        for p in model.fragment_feature_encoder.parameters():
            self.assertIn(id(p),ids)
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[k],p) for k,p in model.fragment_feature_encoder.named_parameters()))

    def test_raw_classifier_route_and_gate_only_gradient(self):
        model,batch=ContextFeatureTests().model_batch('raw')
        captured=[]
        def check(module,args):
            graph=args[0]
            torch.testing.assert_close(graph.x,batch.x[graph.selected_original_atom_indices],rtol=0,atol=0)
            captured.append(True)
        hook=model.atom_classifier.register_forward_pre_hook(check)
        (-model(batch)[:,1].mean()).backward();hook.remove()
        self.assertTrue(captured)
        # Diagnostic ablation only: removing all differentiable gates blocks
        # encoder gradients through the otherwise detached hard selection.
        model.zero_grad(set_to_none=True);model.message_gate_mode='unit'
        (-model(batch)[:,1].mean()).backward()
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in model.fragment_feature_encoder.parameters()))

    def test_atom_encoding_is_full_context_despite_sampler(self):
        model,batch=ContextFeatureTests().model_batch('raw');model.train()
        h=model.encode_atom_features(batch)
        graph=build_fragment_graph_batch(batch)
        original=graph.x.clone()
        model.fragment_edge_sampler.keep_ratio=.3
        scores,eligible=model.score_fragment_edges(graph,batch,h)
        torch.testing.assert_close(h,model.encode_atom_features(batch),rtol=0,atol=0)
        torch.testing.assert_close(graph.x,original,rtol=0,atol=0)
        self.assertEqual(scores.numel(),graph.edge_index.size(1))
        self.assertTrue((scores[~eligible]==0).all())

    def test_encoder_batch_locality_permutation_and_bondless_atoms(self):
        torch.manual_seed(19)
        encoder=InternalFragmentEncoder(3,2,8,2,context='molecular').eval()
        first=Data(x=torch.randn(3,3),edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),edge_attr=torch.randn(4,2))
        assignment=torch.tensor([0,0,1]);alone=encoder.encode_atoms(first,assignment)
        combined=Data(x=torch.cat([first.x,torch.randn(2,3)]),edge_index=first.edge_index,edge_attr=first.edge_attr)
        together=encoder.encode_atoms(combined,torch.tensor([0,0,1,2,3]))
        torch.testing.assert_close(alone,together[:3],rtol=1e-5,atol=1e-6)
        order=torch.tensor([2,0,1]);inverse=torch.argsort(order)
        permuted=Data(x=first.x[order],edge_index=inverse[first.edge_index],edge_attr=first.edge_attr)
        torch.testing.assert_close(encoder.encode_atoms(permuted,assignment[order]),alone[order],rtol=1e-5,atol=1e-6)
        isolated=Data(x=torch.randn(1,3),edge_index=torch.empty((2,0),dtype=torch.long),edge_attr=torch.empty((0,2)))
        self.assertTrue(torch.isfinite(encoder.encode_atoms(isolated,torch.tensor([0]))).all())


if __name__=='__main__':unittest.main()
