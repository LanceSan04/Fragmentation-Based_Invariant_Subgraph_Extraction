import copy
import unittest
import torch
from torch_geometric.data import Batch
from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data
from src.fragmentation import build_fragmenter
from src.models.fragment_ciga import FragmentCIGA


class FullGraphGatingTests(unittest.TestCase):
    def fixture(self, smiles, full=True, unit=False):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        data=[molecule_example_to_data(build_molecule_example(s,label=i%2,
            fragmenter=build_fragmenter('group_graph'))) for i,s in enumerate(smiles)]
        batch=Batch.from_data_list(data,follow_batch=['fragment_x_static'])
        view=FragmentCIGA(atom_feature_dim=batch.x.size(1),fragment_feature_dim=batch.fragment_x_static.size(1),
            fragment_edge_dim=batch.fragment_edge_attr.size(1),atom_edge_dim=batch.edge_attr.size(1),
            gnn_hidden_dim=8,num_gnn_layers=2,dropout=0.,edge_ratio=1.,fragment_sample_ratio=1.,
            scorer_context='full',full_graph_gating=full,message_gate_mode='unit' if unit else 'learned')
        return batch,view

    def test_disconnected_components_isolated_atoms_and_provenance(self):
        batch,view=self.fixture(['CCOC(=O)C.[Na+]','CC(=O)O.CC','[He]'])
        view.eval()
        pred,trace=view(batch,True)
        graph=view.get_invariant_subgraph(batch)
        torch.testing.assert_close(graph.x,batch.x)
        torch.testing.assert_close(graph.bond_index,batch.bond_index)
        self.assertEqual(trace['selected_atom_ids'].tolist(),list(range(batch.x.size(0))))
        self.assertEqual(trace['selected_bond_ids'].numel(),batch.bond_index.size(1))
        self.assertTrue(torch.isfinite(pred).all())
        pred[:,1].sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in view.fragment_masker.parameters()))

    def test_connected_full_graph_matches_legacy_all_edges(self):
        batch,view=self.fixture(['CCOC(=O)C','CCN(CC)CC'])
        legacy=copy.deepcopy(view);legacy.full_graph_gating=False
        view.eval();legacy.eval()
        torch.testing.assert_close(view(batch),legacy(batch),atol=1e-6,rtol=1e-5)

    def test_unit_gates_match_erm_on_disconnected_and_edgeless_molecules(self):
        batch,view=self.fixture(['CCO.[Na+]','[He]','CCOC(=O)C'],unit=True)
        erm=copy.deepcopy(view);erm.erm_baseline=True
        view.eval();erm.eval()
        torch.testing.assert_close(view(batch),erm(batch),atol=1e-6,rtol=1e-5)

