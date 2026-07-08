"""SuGAREnsemble: inference-only ensemble over trained FragmentCIGA views."""

import torch
import torch.nn.functional as F

from src.models.fragment_ciga import topk_mask_over_eligible, restore_full_edge_outputs
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.atom_reconstructor import reconstruct_batch, reconstruct_batch_fast


class SuGAREnsemble:
    """Inference-only ensemble over trained FragmentCIGA views.

    Not an nn.Module — purely coordinates trained views for test-time inference.
    Each view has been trained to independently identify invariant subgraphs.
    At inference time their edge scores are aggregated into one shared invariant
    subgraph, and all classifiers soft-vote on that subgraph.

    Stage 1: Each view scores fragment edges independently (with its own sampler
             and masker).
    Stage 2: Aggregate edge scores (mean or max) → shared top-k subgraph.
    Stage 3: All classifiers predict on the shared subgraph; average softmax
             probabilities → final log-probs.
    """

    def __init__(self, views, aggregation_mode="mean"):
        """
        Args:
            views: Iterable of trained FragmentCIGA instances (one per ensemble view).
            aggregation_mode: 'mean' averages edge scores across views;
                'max' takes the per-edge maximum.
        """
        if aggregation_mode not in ("mean", "max"):
            raise ValueError("aggregation_mode must be 'mean' or 'max'")
        self.views = list(views)
        self.aggregation_mode = aggregation_mode

    @property
    def num_views(self):
        return len(self.views)

    def eval(self):
        """Set all views to eval mode."""
        for v in self.views:
            v.eval()
        return self

    def train(self):
        """Set all views to train mode."""
        for v in self.views:
            v.train()
        return self

    def _aggregate_weights(self, all_soft_weights):
        """Combine per-view edge weight vectors using the configured mode."""
        stacked = torch.stack(all_soft_weights, dim=0)  # [n_views, num_edges]
        if self.aggregation_mode == "mean":
            return stacked.mean(dim=0)
        return stacked.max(dim=0).values

    def _aggregate_subgraph(self, batch_data, views, fast=False):
        """Stage 1+2: per-view fragment edge scores → one shared aggregated subgraph.

        Args:
            batch_data: Batched MoleculeData
            views: List of FragmentCIGA views to aggregate over.
            fast: When True, reconstruct only the tensors the classifier needs
                via ``reconstruct_batch_fast`` (the soft-vote inference path).
                When False, use the full provenance-carrying ``reconstruct_batch``
                (analysis / subgraph-extraction paths).

        Returns:
            atom_batch: batched reconstructed atom subgraph for the shared subgraph.
            agg_weights: [num_fragment_edges] aggregated soft edge scores.
        """
        device = batch_data.x.device
        fragment_batch = build_fragment_graph_batch(batch_data).to(device)

        # Stage 1: per-view fragment edge sampling and scoring
        all_soft_weights = []
        all_eligible_masks = []
        for view in views:
            sampled_fragment_batch, eligible_mask = view.fragment_edge_sampler(fragment_batch)
            sampled_edge_scores, sampled_bool_mask = view.fragment_masker(sampled_fragment_batch)
            _, edge_scores = restore_full_edge_outputs(
                fragment_batch.edge_index.size(1),
                sampled_fragment_batch.full_fragment_edge_id,
                sampled_bool_mask,
                sampled_edge_scores,
            )
            all_soft_weights.append(edge_scores)
            all_eligible_masks.append(eligible_mask)

        # Stage 2: aggregate → single shared top-k subgraph
        agg_weights = self._aggregate_weights(all_soft_weights)

        aggregate_eligible = torch.stack(
            [m.to(device=agg_weights.device, dtype=torch.bool) for m in all_eligible_masks],
            dim=0,
        ).any(dim=0)
        edge_batch = fragment_batch.batch[fragment_batch.edge_index[0]]
        bool_mask = topk_mask_over_eligible(
            agg_weights.detach(),
            edge_batch,
            views[0].fragment_masker.ratio,
            aggregate_eligible,
        )
        if fast:
            atom_batch = reconstruct_batch_fast(
                batch_data, bool_mask, selected_fragment_edge_weight=agg_weights
            )
        else:
            atom_batch = reconstruct_batch(
                batch_data, bool_mask, selected_fragment_edge_weight=agg_weights
            ).to(device)
        return atom_batch, agg_weights

    def __call__(self, batch_data, view_indices=None):
        """Ensemble inference: aggregate masker scores → shared subgraph → soft vote.

        Args:
            batch_data: Batched MoleculeData
            view_indices: Optional list of view indices to use. Defaults to all views.
                Used by greedy_view_selection to evaluate view subsets.

        Returns:
            predictions: [batch_size, num_classes] log-probabilities
        """
        views = [self.views[i] for i in view_indices] if view_indices is not None else self.views
        if not views:
            raise ValueError("view_indices must contain at least one view")

        # Stage 1+2: per-view scoring → shared aggregated subgraph
        atom_batch, _ = self._aggregate_subgraph(batch_data, views, fast=True)

        # Stage 3: each classifier predicts on the shared subgraph; soft vote
        all_probs = []
        for view in views:
            logits = view.atom_classifier(atom_batch)
            all_probs.append(F.softmax(logits, dim=-1))

        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)
        return torch.log(avg_probs.clamp(min=1e-8))

    def get_invariant_subgraph(self, batch_data, view_idx=0):
        """Return the reconstructed atom subgraph from one view (for analysis).

        Args:
            batch_data: Batched MoleculeData
            view_idx: Which view's masker to use for subgraph extraction

        Returns:
            Batched reconstructed atom-level PyG Data
        """
        return self.views[view_idx].get_invariant_subgraph(batch_data)

    def extract_subgraphs(self, batch_data, view_indices=None):
        """Return each view's own subgraph plus the shared aggregated subgraph.

        Args:
            batch_data: Batched MoleculeData.
            view_indices: Optional subset of view indices. Defaults to all views.

        Returns:
            dict with:
              "per_view": list of batched ReconstructedAtomData, one per selected view
                  (that view's own independent top-k selection, via
                  get_invariant_subgraph).
              "aggregated": batched ReconstructedAtomData — the shared subgraph Stage 3
                  actually votes on (mean/max-aggregated scores across the selected
                  views).
              "aggregated_edge_scores": [num_fragment_edges] aggregated soft scores.
        """
        views = [self.views[i] for i in view_indices] if view_indices is not None else self.views
        if not views:
            raise ValueError("view_indices must contain at least one view")

        per_view = [view.get_invariant_subgraph(batch_data) for view in views]
        aggregated, agg_weights = self._aggregate_subgraph(batch_data, views)
        return {
            "per_view": per_view,
            "aggregated": aggregated,
            "aggregated_edge_scores": agg_weights,
        }
