import unittest
import torch
from torch_geometric.data import Data
from src.models.atom_reconstructor import reconstruct_batch_fast
from src.experiments.connectivity_probe import close_selected_bonds, random_bonds_same_budget


class ConnectivityProbeTests(unittest.TestCase):
    def setUp(self):
        self.batch=Data(x=torch.ones(4,46),batch=torch.zeros(4,dtype=torch.long),ptr=torch.tensor([0,4]),num_graphs=1,
            bond_index=torch.tensor([[0,1,2],[1,2,3]]),bond_attr=torch.ones(3,14),
            fragment_edge_index=torch.tensor([[0,1,2],[1,2,3]]),
            fragment_edge_to_bond_index=torch.arange(3),atom_to_fragment=torch.arange(4),
            fragment_x_static_batch=torch.zeros(4,dtype=torch.long))
        self.mask=torch.tensor([True,False,True]);self.scores=torch.tensor([.8,.3,.9])
        self.original=reconstruct_batch_fast(self.batch,self.mask,self.scores)
        self.trace=dict(selected_atom_ids=self.original.selected_original_atom_indices,
                        selected_bond_ids=self.original.selected_original_bond_indices,
                        selected_fragment_mask=self.mask,fragment_scores=self.scores)

    def test_closure_restores_only_original_bond_and_preserves_gates(self):
        result=close_selected_bonds(self.batch,self.trace,None)
        self.assertTrue(torch.equal(result.x,self.original.x))
        self.assertEqual(result.selected_original_bond_indices.tolist(),[0,1,2])
        torch.testing.assert_close(result.edge_weight[:3],self.scores)
        self.assertEqual(self.original.selected_original_bond_indices.tolist(),[0,2])

    def test_random_preserves_atoms_budget_directions_and_reproducibility(self):
        result=close_selected_bonds(self.batch,self.trace,None)
        a=random_bonds_same_budget(result,self.original,self.batch,8301,0,0,0)
        b=random_bonds_same_budget(result,self.original,self.batch,8301,0,0,0)
        self.assertTrue(torch.equal(a.x,self.original.x))
        self.assertTrue(torch.equal(a.selected_original_atom_indices,self.original.selected_original_atom_indices))
        self.assertEqual(a.selected_original_bond_indices.numel(),2)
        self.assertTrue(torch.equal(a.edge_index[:,2:],a.edge_index[:,:2].flip(0)))
        self.assertTrue(torch.equal(a.edge_index,b.edge_index))
        torch.testing.assert_close(a.edge_weight[:2],self.scores[a.selected_original_bond_indices])

    def test_closure_does_not_add_unselected_atoms(self):
        mask=torch.tensor([True,False,False]);original=reconstruct_batch_fast(self.batch,mask,self.scores)
        trace=dict(selected_atom_ids=original.selected_original_atom_indices,selected_bond_ids=original.selected_original_bond_indices,
                   selected_fragment_mask=mask,fragment_scores=self.scores)
        result=close_selected_bonds(self.batch,trace,None)
        self.assertEqual(result.selected_original_atom_indices.tolist(),[0,1])
        self.assertEqual(result.selected_original_bond_indices.tolist(),[0])
