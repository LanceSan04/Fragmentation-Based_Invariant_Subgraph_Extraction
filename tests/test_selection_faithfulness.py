import unittest
import torch
from torch_geometric.data import Data
from rdkit import Chem
from src.analysis.selection_faithfulness import induced,chemical_metrics,geometry


class SelectionFaithfulnessTests(unittest.TestCase):
    def test_induced_graph_preserves_surviving_features_and_gates(self):
        graph=Data(x=torch.tensor([[1.],[2.],[3.]]),edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
            edge_attr=torch.arange(4.).reshape(4,1),edge_weight=torch.tensor([.1,.2,.3,.4]),
            batch=torch.zeros(3,dtype=torch.long),selected_original_atom_indices=torch.tensor([4,7,9]))
        before=graph.clone();cropped=induced(graph,[False,True,True])
        self.assertEqual(cropped.x.flatten().tolist(),[2.,3.])
        self.assertEqual(cropped.selected_original_atom_indices.tolist(),[7,9])
        torch.testing.assert_close(cropped.edge_weight,torch.tensor([.3,.4]))
        self.assertEqual(cropped.edge_index.tolist(),[[0,1],[1,0]])
        torch.testing.assert_close(graph.edge_weight,before.edge_weight)
        self.assertEqual(geometry(cropped),dict(atoms=2,bonds=1,components=1))

    def test_ring_is_not_intact_when_atoms_remain_but_bond_is_missing(self):
        mol=Chem.MolFromSmiles('C1CCCCC1')
        bonds=[(b.GetBeginAtomIdx(),b.GetEndAtomIdx()) for b in mol.GetBonds()]
        full=chemical_metrics(mol,range(6),bonds)
        broken=chemical_metrics(mol,range(6),bonds[:-1])
        self.assertEqual(full['ring']['complete'],1)
        self.assertEqual(broken['ring']['complete'],0)
        self.assertEqual(broken['ring']['cut'],1)

    def test_untouched_groups_not_counted_as_broken(self):
        metrics=chemical_metrics(Chem.MolFromSmiles('CC(=O)N'),[],[])
        self.assertEqual(metrics['amide']['touched'],0)
        self.assertEqual(metrics['amide']['cut'],0)
