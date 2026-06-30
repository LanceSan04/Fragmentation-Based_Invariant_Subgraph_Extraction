"""Tests for the SuGAR fragment-edge learning path.

Covers:
1. Per-view gradient path: prediction loss backpropagates into FragmentCIGA's
   fragment-edge scorer via the edge-weight soft-path in atom reconstruction.
2. Random sampling: FragmentEdgeSampler in train mode drops fragment edges;
   eval mode uses the full fragment graph. Preprocessing data is untouched.
3. SuGARTrainer: runs prediction, contrastive, and diversity losses for a list
   of FragmentCIGA views.
4. IRM gating: `uses_irm()` requires both a positive weight and multiple envs.

Run from the repository root with:

    .\.venv\Scripts\python.exe -m unittest tests.test_sugar_scorer -v
"""

import unittest

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data
from src.fragmentation import build_fragmenter
from src.models.fragment_ciga import FragmentCIGA
from src.models.fragment_edge_sampler import FragmentEdgeSampler
from src.models.fragment_graph import build_fragment_graph_batch
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

    def _make_view(self, batch, **kwargs):
        defaults = dict(
            atom_feature_dim=batch.x.size(1),
            fragment_feature_dim=batch.fragment_x_static.size(1),
            num_classes=2,
            num_gnn_layers=2,
            gnn_hidden_dim=32,
            edge_ratio=1.0,
            dropout=0.0,
            fragment_edge_dim=batch.fragment_edge_attr.size(1),
            atom_edge_dim=batch.edge_attr.size(1),
        )
        defaults.update(kwargs)
        return FragmentCIGA(**defaults)

    def test_prediction_loss_backpropagates_to_fragment_edge_scorer(self):
        """NLL loss on FragmentCIGA outputs must backpropagate into the masker."""
        torch.manual_seed(7)
        data_list = [
            self.build_data("CCOC(=O)C", 0),
            self.build_data("CCN(CC)CC", 1),
        ]
        batch = Batch.from_data_list(data_list, follow_batch=["fragment_x_static"])

        view = self._make_view(batch)
        view.train()

        predictions, intermediates = view(batch, return_intermediates=True)
        labels = batch.y.view(-1).long()
        loss = F.nll_loss(predictions, labels)

        view.zero_grad()
        loss.backward()

        scorer_grad = sum(
            float(p.grad.abs().sum())
            for p in view.fragment_masker.edge_att.parameters()
            if p.grad is not None
        )
        self.assertGreater(scorer_grad, 0.0)
        self.assertTrue(intermediates["fragment_scores"].requires_grad)

    def test_training_sampler_randomly_drops_fragment_edges_only(self):
        """FragmentEdgeSampler in train mode drops edges but leaves nodes intact.

        Also verifies that FragmentCIGA with sub-sampling produces valid log-prob
        outputs in both train and eval modes, and that atom/fragment node tensors
        in the original batch are not modified.
        """
        torch.manual_seed(11)
        data_list = [
            self.build_data("CCOC(=O)C", 0),
            self.build_data("CCN(CC)CC", 1),
        ]
        batch = Batch.from_data_list(data_list, follow_batch=["fragment_x_static"])
        full_num_fragment_edges = batch.fragment_edge_index.size(1)

        # Direct sampler tests — untouched by model refactor
        fragment_batch = build_fragment_graph_batch(batch)
        sampler = FragmentEdgeSampler(keep_ratio=0.5, seed=123)
        sampler.train()
        sampled_fragment_batch, direct_eligible_mask = sampler(fragment_batch)

        self.assertTrue(torch.equal(sampled_fragment_batch.x, fragment_batch.x))
        self.assertTrue(torch.equal(sampled_fragment_batch.batch, fragment_batch.batch))
        self.assertEqual(sampled_fragment_batch.num_nodes, fragment_batch.num_nodes)
        self.assertEqual(sampled_fragment_batch.edge_index.size(1), int(direct_eligible_mask.sum()))
        self.assertLess(sampled_fragment_batch.edge_index.size(1), full_num_fragment_edges)

        # FragmentCIGA with sub-sampling: train and eval produce valid log-probs
        view = self._make_view(batch, fragment_sample_ratio=0.5, sampler_seed=123)

        view.train()
        predictions_train = view(batch)
        self.assertEqual(predictions_train.shape[0], batch.y.numel())
        self.assertTrue((predictions_train <= 0).all(), "train log-probs must be ≤ 0")

        view.eval()
        with torch.no_grad():
            predictions_eval = view(batch)
        self.assertEqual(predictions_eval.shape[0], batch.y.numel())
        self.assertTrue((predictions_eval <= 0).all(), "eval log-probs must be ≤ 0")

        # Batch tensors are not modified by sampling
        self.assertEqual(batch.x.size(0), sum(d.x.size(0) for d in data_list))
        self.assertEqual(
            batch.fragment_x_static.size(0),
            sum(d.fragment_x_static.size(0) for d in data_list),
        )

    def test_trainer_runs_prediction_contrastive_and_diversity_losses(self):
        """SuGARTrainer.train_epoch must complete without error with aux losses enabled."""
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

        views = [self._make_view(batch) for _ in range(2)]
        trainer = SuGARTrainer(
            views,
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
        """IRM activates only when lambda_irm > 0 AND num_envs > 1."""
        view = FragmentCIGA(
            atom_feature_dim=133,
            fragment_feature_dim=149,
            num_classes=2,
            num_gnn_layers=1,
            gnn_hidden_dim=16,
            edge_ratio=1.0,
            dropout=0.0,
            fragment_edge_dim=14,
            atom_edge_dim=14,
        )

        trainer = SuGARTrainer([view], lambda_irm=1.0, num_envs=1)
        self.assertFalse(trainer.uses_irm())

        trainer = SuGARTrainer([view], lambda_irm=1.0, num_envs=2)
        self.assertTrue(trainer.uses_irm())


if __name__ == "__main__":
    unittest.main()
