"""Tests for the current SuGAr fragment-edge learning path.

This file covers two closely related pieces of the prototype:

1. Learnable fragment-edge scoring:
   prediction loss must still backpropagate into the CIGA-style edge scorer.
   The hard top-k mask is non-differentiable, so selected soft edge scores are
   passed into atom reconstruction as edge weights to preserve a gradient path.

2. SuGAr-style random sampling:
   training mode may randomly drop fragment-edge instances before scoring, but
   it must not modify preprocessing outputs, atom nodes, atom bonds, fragment
   nodes, or molecule records. Sampled edge selections must map back to the full
   preprocessed fragment-edge order before atom reconstruction. Evaluation mode
   should use the full fragment graph.

Run this file from the repository root with:

    .\.venv\Scripts\python.exe -m unittest tests.test_sugar_scorer -v

Run the full test suite with:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

import unittest

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data
from src.fragmentation import build_fragmenter
from src.models.fragment_edge_sampler import FragmentEdgeSampler
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.sugar_ensemble import SuGAREnsemble
from src.training.sugar_trainer import SuGARTrainer


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

    def test_training_sampler_randomly_drops_fragment_edges_only(self):
        torch.manual_seed(11)
        data_list = [
            self.build_data("CCOC(=O)C", 0),
            self.build_data("CCN(CC)CC", 1),
        ]
        batch = Batch.from_data_list(data_list, follow_batch=["fragment_x_static"])
        full_num_fragment_edges = batch.fragment_edge_index.size(1)

        fragment_batch = build_fragment_graph_batch(batch)
        sampler = FragmentEdgeSampler(keep_ratio=0.5, seed=123)
        sampler.train()
        sampled_fragment_batch, direct_eligible_mask = sampler(fragment_batch)

        self.assertTrue(torch.equal(sampled_fragment_batch.x, fragment_batch.x))
        self.assertTrue(torch.equal(sampled_fragment_batch.batch, fragment_batch.batch))
        self.assertEqual(sampled_fragment_batch.num_nodes, fragment_batch.num_nodes)
        self.assertEqual(sampled_fragment_batch.edge_index.size(1), int(direct_eligible_mask.sum()))
        self.assertLess(sampled_fragment_batch.edge_index.size(1), full_num_fragment_edges)

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
            fragment_sample_ratio=0.5,
            sampler_seed=123,
        )

        model.train()
        logits, intermediates = model(batch, return_intermediates=True)
        eligible_mask = intermediates["eligible_fragment_masks"][0]
        selected_mask = intermediates["bool_mask"]
        sampled_edge_ids = intermediates["sampled_fragment_edge_ids"][0]

        self.assertEqual(logits.size(0), batch.y.numel())
        self.assertEqual(eligible_mask.numel(), full_num_fragment_edges)
        self.assertGreater(int(eligible_mask.sum()), 0)
        self.assertLess(int(eligible_mask.sum()), full_num_fragment_edges)
        self.assertTrue(torch.equal(selected_mask, eligible_mask))
        self.assertTrue(torch.equal(sampled_edge_ids.cpu(), eligible_mask.nonzero().view(-1).cpu()))

        atom_graph = intermediates["atom_graphs"][0]
        self.assertEqual(atom_graph.num_graphs, batch.y.numel())
        self.assertEqual(batch.x.size(0), sum(data.x.size(0) for data in data_list))
        self.assertEqual(batch.fragment_x_static.size(0), sum(data.fragment_x_static.size(0) for data in data_list))

        model.eval()
        with torch.no_grad():
            _, eval_intermediates = model(batch, return_intermediates=True)
        eval_eligible_mask = eval_intermediates["eligible_fragment_masks"][0]
        eval_selected_mask = eval_intermediates["bool_mask"]

        self.assertTrue(torch.all(eval_eligible_mask))
        self.assertTrue(torch.all(eval_selected_mask))

    def test_trainer_runs_prediction_contrastive_and_diversity_losses(self):
        torch.manual_seed(17)
        data_list = [
            self.build_data("CCOC(=O)C", 0),
            self.build_data("CCN(CC)CC", 0),
            self.build_data("CCCOC(=O)C", 1),
            self.build_data("CCCN(CC)CC", 1),
        ]
        loader = DataLoader(
            data_list,
            batch_size=4,
            shuffle=False,
            follow_batch=["fragment_x_static"],
        )
        batch = next(iter(loader))

        model = SuGAREnsemble(
            atom_feature_dim=batch.x.size(1),
            fragment_feature_dim=batch.fragment_x_static.size(1),
            num_classes=2,
            num_gnn_layers=2,
            gnn_hidden_dim=32,
            edge_ratio=1.0,
            num_views=2,
            dropout=0.0,
            fragment_edge_dim=batch.fragment_edge_attr.size(1),
            atom_edge_dim=batch.edge_attr.size(1),
            fragment_sample_ratio=1.0,
            sampler_seed=17,
        )
        trainer = SuGARTrainer(
            model,
            learning_rate=1e-3,
            lambda_contrast=1.0,
            lambda_diversity=1.0,
            lambda_irm=0.0,
            contrast_sampling="mul",
        )

        loss, accuracy = trainer.train_epoch(loader)

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_trainer_irm_requires_multiple_environments(self):
        """IRM should require both a positive weight and multiple environments."""
        model = SuGAREnsemble(
            atom_feature_dim=133,
            fragment_feature_dim=149,
            num_classes=2,
            num_gnn_layers=1,
            gnn_hidden_dim=16,
            edge_ratio=1.0,
            num_views=1,
            dropout=0.0,
            fragment_edge_dim=14,
            atom_edge_dim=14,
        )

        trainer = SuGARTrainer(model, lambda_irm=1.0, num_envs=1)
        self.assertFalse(trainer.uses_irm())

        trainer = SuGARTrainer(model, lambda_irm=1.0, num_envs=2)
        self.assertTrue(trainer.uses_irm())


if __name__ == "__main__":
    unittest.main()
