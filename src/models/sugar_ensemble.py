"""Ensemble model combining multiple fragment views for invariant subgraph learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.fragment_masker import FragmentMasker
from src.models.gnn_encoders import SimpleGAT
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.fragment_edge_sampler import FragmentEdgeSampler
from src.models.atom_reconstructor import reconstruct_batch
from src.data.atom_graph import ATOM_FDIM, BOND_FDIM


class AtomClassifier(nn.Module):
    """GAT classifier for atom-level invariant subgraphs.

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
        self.gnn = SimpleGAT(input_dim, hidden_dim, num_layers, dropout, edge_dim=edge_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, num_classes),
        )

    def forward(self, data, return_embeddings=False):
        """
        Args:
            data: Batched PyG Data (reconstructed atom subgraphs)
            return_embeddings: If True, also return graph-level pooled embedding.

        Returns:
            logits: [batch_size, num_classes]
            h_graph: (optional) [batch_size, hidden_dim] — only when return_embeddings=True
        """
        edge_attr = data.edge_attr
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            edge_attr = edge_attr * data.edge_weight.view(-1, 1)
        _, h_graph = self.gnn(data.x, data.edge_index, data.batch, edge_attr=edge_attr)
        logits = self.pred_head(h_graph)
        if return_embeddings:
            return logits, h_graph
        return logits


class SuGAREnsemble(nn.Module):
    """Multi-view fragment learning with invariant subgraph extraction.

    Stage 2 — aggregate edge weights from all n maskers (mean or max) and apply
    a single top-k selection to obtain one shared invariant subgraph G_c^bm:

        G_c^bm = TopK( (1/n) * sum_i E^(i) )        [mean mode]
        G_c^bm = TopK( max_i E^(i) )                 [max mode]

    Stage 3 — all n classifiers predict on G_c^bm; their softmax probabilities
    are averaged (soft voting) to produce the final class distribution:

        FinalDecision = argmax_k (1/n) * sum_i P_i(Y=k)
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
        aggregation_mode="mean",
        fragment_edge_dim=None,
        atom_edge_dim=BOND_FDIM,
        fragment_sample_ratio=1.0,
        sampler_seed=None,
    ):
        """
        Args:
            aggregation_mode: Stage 2 weight aggregation — 'mean' averages edge
                weights across all maskers; 'max' takes the per-edge maximum.
        """
        super().__init__()
        if fragment_feature_dim is None:
            raise ValueError("fragment_feature_dim must be provided")
        if aggregation_mode not in ("mean", "max"):
            raise ValueError("aggregation_mode must be 'mean' or 'max'")

        self.num_views = num_views
        self.fragment_edge_samplers = nn.ModuleList([
            FragmentEdgeSampler(
                keep_ratio=fragment_sample_ratio,
                seed=None if sampler_seed is None else int(sampler_seed) + view_idx,
            )
            for view_idx in range(num_views)
        ])
        self.aggregation_mode = aggregation_mode

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

    def _aggregate_weights(self, all_soft_weights):
        """Stage 2: combine per-masker edge weight vectors into one."""
        stacked = torch.stack(all_soft_weights, dim=0)  # [n, num_edges]
        if self.aggregation_mode == "mean":
            return stacked.mean(dim=0)
        return stacked.max(dim=0).values

    def forward(self, batch_data, return_intermediates=False, view_indices=None):
        """
        Stage 2: Aggregate masker edge weights (mean or max): top-k shared subgraph.
        Stage 3: All classifiers predict on the shared subgraph; soft vote (avg softmax).

        Args:
            batch_data: Batched MoleculeData objects
            return_intermediates: If True, also return per-masker scores, shared atom
                                  graph, per-classifier embeddings and logits.
            view_indices: Optional list of view indices to use. Defaults to all views.
                          Used by greedy_view_selection to evaluate view subsets.

        Returns:
            predictions: [batch_size, num_classes] — log-probabilities (use with F.nll_loss)
            intermediates: (optional) dict with 'fragment_scores', 'bool_mask',
                           'atom_graphs', 'embeddings', 'per_view_logits'
        """
        views = list(view_indices) if view_indices is not None else list(range(self.num_views))
        if not views:
            raise ValueError("view_indices must contain at least one view")

        device = batch_data.x.device
        fragment_batch = build_fragment_graph_batch(batch_data).to(device)

        all_soft_weights = []
        all_eligible_masks = []
        intermediates = {
            "bool_mask": None,
            "fragment_scores": [],
            "fragment_masks": [],
            "eligible_fragment_masks": [],
            "sampled_fragment_edge_ids": [],
            "atom_graphs": [],
            "embeddings": [],
            "per_view_logits": [],
        }

        for view_idx in views:
            sampled_fragment_batch, eligible_mask = self.fragment_edge_samplers[view_idx](
                fragment_batch
            )
            sampled_edge_scores, sampled_bool_mask = self.fragment_maskers[view_idx](
                sampled_fragment_batch
            )
            full_mask, edge_scores = self._restore_full_edge_outputs(
                fragment_batch.edge_index.size(1),
                sampled_fragment_batch.full_fragment_edge_id,
                sampled_bool_mask,
                sampled_edge_scores,
            )
            all_soft_weights.append(edge_scores)
            all_eligible_masks.append(eligible_mask)

            if return_intermediates:
                intermediates["fragment_scores"].append(edge_scores)
                intermediates["fragment_masks"].append(full_mask)
                intermediates["eligible_fragment_masks"].append(eligible_mask)
                intermediates["sampled_fragment_edge_ids"].append(sampled_fragment_batch.full_fragment_edge_id)

        # Aggregate across maskers, then single top-k → one shared subgraph G_c^bm
        agg_weights = self._aggregate_weights(all_soft_weights)
        aggregate_eligible_mask = torch.stack(
            [mask.to(device=agg_weights.device, dtype=torch.bool) for mask in all_eligible_masks],
            dim=0,
        ).any(dim=0)
        edge_batch = fragment_batch.batch[fragment_batch.edge_index[0]]
        bool_mask = self._topk_mask_over_eligible(
            agg_weights.detach(),
            edge_batch,
            self.fragment_maskers[views[0]].ratio,
            aggregate_eligible_mask,
        )
        atom_batch = reconstruct_batch(
            batch_data, bool_mask, selected_fragment_edge_weight=agg_weights
        ).to(device)

        # Stage 3: each classifier predicts on the shared subgraph; soft vote
        if return_intermediates:
            intermediates["bool_mask"] = bool_mask

        all_probs = []
        for view_idx in views:
            classifier = self.atom_classifiers[view_idx]

            if return_intermediates:
                logits, h_graph = classifier(atom_batch, return_embeddings=True)
                intermediates["atom_graphs"].append(atom_batch)
                intermediates["embeddings"].append(h_graph)
                intermediates["per_view_logits"].append(logits)
            else:
                logits = classifier(atom_batch)

            all_probs.append(F.softmax(logits, dim=-1))

        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)  # [batch, num_classes]
        # log-probs: argmax preserved (log monotone), compatible with F.nll_loss
        predictions = torch.log(avg_probs.clamp(min=1e-8))

        if return_intermediates:
            return predictions, intermediates
        return predictions

    @staticmethod
    def _topk_mask_over_eligible(scores, edge_batch, ratio, eligible_mask):
        """Select top-k edges per graph while forbidding sampler-dropped edges."""
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if scores.numel() == 0:
            return mask

        eligible_mask = eligible_mask.to(device=scores.device, dtype=torch.bool)
        for gid in edge_batch.unique():
            graph_idx = (edge_batch == gid).nonzero(as_tuple=True)[0]
            eligible_idx = graph_idx[eligible_mask[graph_idx]]
            if eligible_idx.numel() == 0:
                continue

            k = max(1, int(eligible_idx.numel() * ratio))
            top_local = scores[eligible_idx].topk(min(k, eligible_idx.numel()), largest=True).indices
            mask[eligible_idx[top_local]] = True
        return mask

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
