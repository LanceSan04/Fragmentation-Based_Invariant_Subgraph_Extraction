"""Behavioral checks for the matched random fragment-selection control."""

import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data
from src.fragmentation import build_fragmenter
from src.models.fragment_ciga import FragmentCIGA
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.random_fragment_selection import random_edge_ranks
from src.models.sugar_ensemble import SuGAREnsemble


class RandomSelectionControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fragmenter=build_fragmenter('group_graph')
        cls.molecules=[molecule_example_to_data(build_molecule_example(
            smiles,label=i%2,molecule_id=f'm{i}',fragmenter=fragmenter))
            for i,smiles in enumerate(('CCOC(=O)C','CCN(CC)CC','c1ccccc1CC(=O)NC'))]

    def batch(self, indices):
        return Batch.from_data_list([self.molecules[i] for i in indices],
                                    follow_batch=['fragment_x_static'])

    def view(self, batch, seed=7):
        return FragmentCIGA(atom_feature_dim=batch.x.size(1),
            fragment_feature_dim=batch.fragment_x_static.size(1),
            fragment_edge_dim=batch.fragment_edge_attr.size(1),
            atom_edge_dim=batch.edge_attr.size(1),gnn_hidden_dim=16,
            num_gnn_layers=1,edge_ratio=0.5,fragment_sample_ratio=0.7,
            sampler_seed=seed,selection_mode='random')

    def test_evaluation_ranks_are_stable_across_batch_order(self):
        first=build_fragment_graph_batch(self.batch([0,1,2]))
        second=build_fragment_graph_batch(self.batch([2,0,1]))
        a=random_edge_ranks(first,11,False)
        b=random_edge_ranks(second,11,False)
        for old,new in ((0,1),(1,2),(2,0)):
            ea=first.batch[first.edge_index[0]]==old
            eb=second.batch[second.edge_index[0]]==new
            self.assertTrue(torch.equal(a[ea],b[eb]))
        self.assertTrue(torch.equal(a,random_edge_ranks(first,11,False)))

    def test_random_control_keeps_budget_and_uses_unit_gate(self):
        batch=self.batch([0,1,2]); view=self.view(batch).eval()
        with patch.object(view.fragment_masker,'forward',side_effect=AssertionError('learned scorer called')):
            predictions,info=view(batch,return_intermediates=True)
            subgraph=view.get_invariant_subgraph(batch)
            shared=SuGAREnsemble([view],inference_policy='shared')(batch)
        self.assertEqual(predictions.shape,(3,2))
        self.assertEqual(shared.shape,(3,2))
        self.assertIsNone(info['fragment_scores'])
        self.assertTrue(torch.all(subgraph.edge_weight==1))
        fragment=build_fragment_graph_batch(batch)
        edge_batch=fragment.batch[fragment.edge_index[0]]
        for graph in range(3):
            count=int((edge_batch==graph).sum())
            expected=max(1,int(count*0.5)) if count else 0
            self.assertEqual(int(info['selected_fragment_mask'][edge_batch==graph].sum()),expected)

    def test_erm_rejects_random_fragment_mode(self):
        batch=self.batch([0])
        with self.assertRaisesRegex(ValueError,'ERM has no fragment selection'):
            FragmentCIGA(atom_feature_dim=batch.x.size(1),
                fragment_feature_dim=batch.fragment_x_static.size(1),
                fragment_edge_dim=batch.fragment_edge_attr.size(1),
                erm_baseline=True,selection_mode='random')


if __name__=='__main__':unittest.main()
