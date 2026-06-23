"""Ensemble model combining multiple fragment views for invariant subgraph learning."""

import torch
import torch.nn as nn
from src.models.fragment_graph_ciga import FragmentMasker, SimpleGNN
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.fragment_edge_sampler import FragmentEdgeSampler
from src.models.atom_reconstructor import reconstruct_batch
from src.data.atom_graph import ATOM_FDIM, BOND_FDIM

class AtomClassifier(nn.Module):
    """GNN classifier for atom-level invariant subgraphs.

    Expects atom graphs produced by atom_reconstructor.reconstruct_batch —
    subgraphs already pruned to the invariant fragment selection.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.0,
        num_classes=2,
        edge_dim=BOND_FDIM,
    ):
        super().__init__()
        self.gnn = SimpleGNN(input_dim, hidden_dim, num_layers, dropout, edge_dim=edge_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, num_classes),
        )

    def forward(self, data):
        """
        Args:
            data: Batched PyG Data (reconstructed atom subgraphs)

        Returns:
            logits: [batch_size, num_classes]
        """
        edge_attr = data.edge_attr
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            edge_attr = edge_attr * data.edge_weight.view(-1, 1)
        _, h_graph = self.gnn(data.x, data.edge_index, data.batch, edge_attr=edge_attr)
        return self.pred_head(h_graph)


class SuGAREnsemble(nn.Module):
    """Multi-view fragment learning with invariant subgraph extraction.

    Per-view pipeline:
      1. FragmentMasker   — fragment graph → boolean edge mask
      2. reconstruct_batch — deterministic atom-level subgraph reconstruction
      3. AtomClassifier   — atom subgraph → class logits
    Logits from all views are concatenated and passed through an aggregator MLP.
    """

    def __init__(
        self,
        atom_feature_dim=ATOM_FDIM,
        fragment_feature_dim=None,
        num_classes=2,
        num_gnn_layers=3,
        gnn_hidden_dim=128,
        edge_ratio=0.5,
        num_views=1,
        dropout=0.0,
        fragment_edge_dim=None,
        atom_edge_dim=BOND_FDIM,
        fragment_sample_ratio=1.0,
        sampler_seed=None,
    ):
        super().__init__()
        if fragment_feature_dim is None:
            raise ValueError("fragment_feature_dim must be provided")
        self.num_views = num_views
        self.fragment_edge_samplers = nn.ModuleList([
            FragmentEdgeSampler(
                keep_ratio=fragment_sample_ratio,
                seed=None if sampler_seed is None else int(sampler_seed) + view_idx,
            )
            for view_idx in range(num_views)
        ])

        self.fragment_maskers = nn.ModuleList([
            FragmentMasker(
                input_dim=fragment_feature_dim,
                hidden_dim=gnn_hidden_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
                ratio=edge_ratio,
                edge_dim=fragment_edge_dim,
            )
            for _ in range(num_views)
        ])

        self.atom_classifiers = nn.ModuleList([
            AtomClassifier(
                input_dim=atom_feature_dim,
                hidden_dim=gnn_hidden_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
                num_classes=num_classes,
                edge_dim=atom_edge_dim,
            )
            for _ in range(num_views)
        ])

        self.aggregator = nn.Sequential(
            nn.Linear(num_views * num_classes, 2 * num_classes),
            nn.ReLU(),
            nn.Linear(2 * num_classes, num_classes),
        )

    def forward(self, batch_data, return_intermediates=False):
        """
        Args:
            batch_data: Batched MoleculeData objects
            return_intermediates: If True, also return per-view fragment masks
                                  and reconstructed atom graphs

        Returns:
            predictions: [batch_size, num_classes]
            intermediates: (optional) dict with 'fragment_masks' and 'atom_graphs'
        """
        device = batch_data.x.device
        fragment_batch = build_fragment_graph_batch(batch_data).to(device)

        all_logits = []
        intermediates = {
            "fragment_masks": [],
            "fragment_scores": [],
            "eligible_fragment_masks": [],
            "sampled_fragment_edge_ids": [],
            "atom_graphs": [],
        }

        for view_idx in range(self.num_views):
            sampled_fragment_batch, eligible_mask = self.fragment_edge_samplers[view_idx](
                fragment_batch
            )
            sampled_edge_scores, sampled_bool_mask = self.fragment_maskers[view_idx](
                sampled_fragment_batch
            )
            bool_mask, edge_scores = self._restore_full_edge_outputs(
                fragment_batch.edge_index.size(1),
                sampled_fragment_batch.full_fragment_edge_id,
                sampled_bool_mask,
                sampled_edge_scores,
            )
            atom_batch = reconstruct_batch(
                batch_data,
                bool_mask,
                selected_fragment_edge_weight=edge_scores,
            ).to(device)
            logits = self.atom_classifiers[view_idx](atom_batch)
            all_logits.append(logits)

            if return_intermediates:
                intermediates["fragment_masks"].append(bool_mask)
                intermediates["fragment_scores"].append(edge_scores)
                intermediates["eligible_fragment_masks"].append(eligible_mask)
                intermediates["sampled_fragment_edge_ids"].append(
                    sampled_fragment_batch.full_fragment_edge_id
                )
                intermediates["atom_graphs"].append(atom_batch)

        predictions = self.aggregator(torch.cat(all_logits, dim=-1))

        if return_intermediates:
            return predictions, intermediates
        return predictions

    @staticmethod
    def _restore_full_edge_outputs(
        num_full_edges,
        sampled_full_edge_ids,
        sampled_bool_mask,
        sampled_edge_scores,
    ):
        """Map sampled-graph scores/masks back to full fragment-edge order."""
        device = sampled_edge_scores.device
        full_mask = torch.zeros(num_full_edges, dtype=torch.bool, device=device)
        full_scores = torch.zeros(num_full_edges, dtype=sampled_edge_scores.dtype, device=device)
        if sampled_full_edge_ids.numel() == 0:
            return full_mask, full_scores

        sampled_full_edge_ids = sampled_full_edge_ids.to(device)
        full_mask[sampled_full_edge_ids] = sampled_bool_mask
        full_scores[sampled_full_edge_ids] = sampled_edge_scores
        return full_mask, full_scores

    def get_invariant_subgraph(self, batch_data, view_idx=0):
        """Return the reconstructed atom subgraph batch for the given view.

        Args:
            batch_data: Batched MoleculeData objects
            view_idx: Which fragment masker to use

        Returns:
            Batched reconstructed atom-level PyG Data
        """
        device = batch_data.x.device
        fragment_batch = build_fragment_graph_batch(batch_data).to(device)
        sampled_fragment_batch, _ = self.fragment_edge_samplers[view_idx](fragment_batch)
        sampled_edge_scores, sampled_bool_mask = self.fragment_maskers[view_idx](
            sampled_fragment_batch
        )
        bool_mask, edge_scores = self._restore_full_edge_outputs(
            fragment_batch.edge_index.size(1),
            sampled_fragment_batch.full_fragment_edge_id,
            sampled_bool_mask,
            sampled_edge_scores,
        )
        return reconstruct_batch(
            batch_data,
            bool_mask,
            selected_fragment_edge_weight=edge_scores,
        )
