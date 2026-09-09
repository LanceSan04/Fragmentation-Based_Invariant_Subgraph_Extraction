"""SuGAREnsemble: inference-only ensemble over trained FragmentCIGA views."""

import math

import torch
import torch.nn.functional as F

from src.models.fragment_ciga import topk_mask_over_eligible, restore_full_edge_outputs
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.atom_reconstructor import reconstruct_batch, reconstruct_batch_fast


class SuGAREnsemble:
    """Inference-only ensemble over trained FragmentCIGA views.

    Views using the same fragmentation method can safely aggregate fragment-edge
    scores because their edge IDs refer to the same preprocessed graph. Mixed
    fragmentation ensembles aggregate within each method group, then combine
    classifier probabilities by soft vote.
    """

    def __init__(self, views, aggregation_mode="mean"):
        if aggregation_mode not in ("mean", "max"):
            raise ValueError("aggregation_mode must be 'mean' or 'max'")
        self.views = list(views)
        self.aggregation_mode = aggregation_mode

    @property
    def num_views(self):
        return len(self.views)

    def eval(self):
        """Set all views to eval mode."""
        for view in self.views:
            view.eval()
        return self

    def train(self):
        """Set all views to train mode."""
        for view in self.views:
            view.train()
        return self

    def _aggregate_weights(self, all_soft_weights):
        """Combine per-view edge weight vectors using the configured mode."""
        stacked = torch.stack(all_soft_weights, dim=0)
        if self.aggregation_mode == "mean":
            return stacked.mean(dim=0)
        return stacked.max(dim=0).values

    @staticmethod
    def _view_method(view):
        return getattr(view, "fragmentation_method", None)

    @classmethod
    def _views_by_fragmentation(cls, views):
        """Group views by compatible fragmentation method in first-use order."""
        groups = []
        group_index = {}
        for view in views:
            method = cls._view_method(view)
            if method not in group_index:
                group_index[method] = len(groups)
                groups.append((method, []))
            groups[group_index[method]][1].append(view)
        return groups

    def _aggregate_subgraph(self, batch_data, views, fast=False):
        """Score compatible views and reconstruct one method-specific subgraph."""
        if not views:
            raise ValueError("views must contain at least one view")
        method = self._view_method(views[0])
        if any(self._view_method(view) != method for view in views):
            raise ValueError(
                "_aggregate_subgraph can only aggregate views using the same fragmentation method"
            )

        device = batch_data.x.device
        fragment_batch = build_fragment_graph_batch(batch_data, method=method).to(device)

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

        agg_weights = self._aggregate_weights(all_soft_weights)
        aggregate_eligible = torch.stack(
            [mask.to(device=agg_weights.device, dtype=torch.bool) for mask in all_eligible_masks],
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
                batch_data,
                bool_mask,
                selected_fragment_edge_weight=agg_weights,
                fragmentation_method=method,
            )
        else:
            atom_batch = reconstruct_batch(
                batch_data,
                bool_mask,
                selected_fragment_edge_weight=agg_weights,
                fragmentation_method=method,
            ).to(device)
        return atom_batch, agg_weights

    def __call__(self, batch_data, view_indices=None):
        """Ensemble inference: method-compatible aggregation plus soft vote."""
        views = [self.views[i] for i in view_indices] if view_indices is not None else self.views
        if not views:
            raise ValueError("view_indices must contain at least one view")

        all_log_probs = []
        # ERM views have no fragment scores to aggregate; each one classifies the
        # full atom graph on its own.
        erm_views = [view for view in views if getattr(view, "erm_baseline", False)]
        subgraph_views = [view for view in views if not getattr(view, "erm_baseline", False)]
        for view in erm_views:
            all_log_probs.append(view(batch_data))

        for _, method_views in self._views_by_fragmentation(subgraph_views):
            atom_batch, _ = self._aggregate_subgraph(batch_data, method_views, fast=True)
            for view in method_views:
                logits = view.atom_classifier(atom_batch)
                all_log_probs.append(F.log_softmax(logits, dim=-1))

        # log of the mean probability, computed in log space so a confidently
        # wrong view cannot underflow the average to the clamp floor.
        stacked = torch.stack(all_log_probs, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(stacked.size(0))

    def get_invariant_subgraph(self, batch_data, view_idx=0):
        """Return the reconstructed atom subgraph from one view for analysis."""
        return self.views[view_idx].get_invariant_subgraph(batch_data)

    def extract_subgraphs(self, batch_data, view_indices=None):
        """Return each view's subgraph plus method-group aggregate subgraphs."""
        views = [self.views[i] for i in view_indices] if view_indices is not None else self.views
        if not views:
            raise ValueError("view_indices must contain at least one view")

        per_view = [view.get_invariant_subgraph(batch_data) for view in views]
        aggregated_by_method = {}
        weights_by_method = {}
        for method, method_views in self._views_by_fragmentation(views):
            aggregated, agg_weights = self._aggregate_subgraph(batch_data, method_views)
            key = method or "default"
            aggregated_by_method[key] = aggregated
            weights_by_method[key] = agg_weights

        first_key = next(iter(aggregated_by_method))
        return {
            "per_view": per_view,
            "aggregated": aggregated_by_method[first_key],
            "aggregated_edge_scores": weights_by_method[first_key],
            "aggregated_by_method": aggregated_by_method,
            "aggregated_edge_scores_by_method": weights_by_method,
        }
