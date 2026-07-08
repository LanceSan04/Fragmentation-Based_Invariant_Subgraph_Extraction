"""FragmentCIGA: single-view trainable model for fragment-based invariant subgraph learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fragment_masker import FragmentMasker, segment_topk_mask
from src.models.gnn_encoders import SimpleGAT
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.fragment_edge_sampler import FragmentEdgeSampler
from src.models.atom_reconstructor import reconstruct_batch, reconstruct_batch_fast
from src.data.atom_graph import ATOM_FDIM, BOND_FDIM


def topk_mask_over_eligible(scores, edge_batch, ratio, eligible_mask):
    """Select top-k edges per graph, forbidding sampler-dropped edges."""
    return segment_topk_mask(scores, edge_batch, ratio, eligible_mask=eligible_mask)


def restore_full_edge_outputs(
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


class AtomClassifier(nn.Module):
    """GAT classifier for atom-level invariant subgraphs."""

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
        edge_attr = data.edge_attr
        if hasattr(data, "edge_weight") and data.edge_weight is not None:
            edge_attr = edge_attr * data.edge_weight.view(-1, 1)
        _, h_graph = self.gnn(data.x, data.edge_index, data.batch, edge_attr=edge_attr)
        logits = self.pred_head(h_graph)
        if return_embeddings:
            return logits, h_graph
        return logits


class FragmentCIGA(nn.Module):
    """Single-view trainable model: fragment masking → reconstruction → classification.

    One FragmentCIGA instance is one "view" in the SuGAR ensemble. During training
    each view operates independently — its masker scores fragment edges, the top-k
    selection defines an invariant subgraph specific to this view, and its classifier
    is trained on that reconstructed atom-level subgraph. Multiple trained views are
    then wrapped in SuGAREnsemble for inference-time aggregation.
    """

    def __init__(
        self,
        atom_feature_dim=ATOM_FDIM,
        fragment_feature_dim=None,
        num_classes=2,
        num_gnn_layers=3,
        gnn_hidden_dim=128,
        edge_ratio=0.5,
        dropout=0.0,
        fragment_edge_dim=None,
        atom_edge_dim=BOND_FDIM,
        fragment_sample_ratio=1.0,
        sampler_seed=None,
    ):
        super().__init__()
        if fragment_feature_dim is None:
            raise ValueError("fragment_feature_dim must be provided")

        self.fragment_edge_sampler = FragmentEdgeSampler(
            keep_ratio=fragment_sample_ratio,
            seed=sampler_seed,
        )
        self.fragment_masker = FragmentMasker(
            input_dim=fragment_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            ratio=edge_ratio,
            edge_dim=fragment_edge_dim,
        )
        self.atom_classifier = AtomClassifier(
            input_dim=atom_feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            num_classes=num_classes,
            edge_dim=atom_edge_dim,
        )

    def forward(self, batch_data, return_intermediates=False, fragment_batch=None):
        """Per-view forward pass: mask → reconstruct → classify.

        Args:
            batch_data: Batched MoleculeData
            return_intermediates: If True, also return embeddings, fragment_scores,
                and logits (needed by SuGARTrainer's auxiliary losses).
            fragment_batch: Optional prebuilt batched fragment graph. The fragment
                graph is a pure function of ``batch_data`` and identical across all
                views, so the trainer builds it once per batch and passes it to
                every view.

        Returns:
            predictions: [batch_size, num_classes] log-probabilities
            intermediates: (only when return_intermediates=True) dict with
                'embeddings', 'fragment_scores', 'logits'
        """
        device = batch_data.x.device
        if fragment_batch is None:
            fragment_batch = build_fragment_graph_batch(batch_data).to(device)

        sampled_fragment_batch, eligible_mask = self.fragment_edge_sampler(fragment_batch)
        sampled_edge_scores, sampled_bool_mask = self.fragment_masker(sampled_fragment_batch)

        _, edge_scores = restore_full_edge_outputs(
            fragment_batch.edge_index.size(1),
            sampled_fragment_batch.full_fragment_edge_id,
            sampled_bool_mask,
            sampled_edge_scores,
        )

        edge_batch = fragment_batch.batch[fragment_batch.edge_index[0]]
        bool_mask = topk_mask_over_eligible(
            edge_scores.detach(),
            edge_batch,
            self.fragment_masker.ratio,
            eligible_mask.to(device=edge_scores.device, dtype=torch.bool),
        )

        atom_batch = reconstruct_batch_fast(
            batch_data, bool_mask, selected_fragment_edge_weight=edge_scores
        )

        logits, h_graph = self.atom_classifier(atom_batch, return_embeddings=True)
        predictions = torch.log(F.softmax(logits, dim=-1).clamp(min=1e-8))

        if return_intermediates:
            return predictions, {
                "embeddings": h_graph,
                "fragment_scores": edge_scores,
                "fragment_edge_batch": edge_batch,
                "fragment_edge_eligible": eligible_mask.to(
                    device=edge_scores.device, dtype=torch.bool
                ),
                "logits": logits,
            }
        return predictions

    def get_invariant_subgraph(self, batch_data):
        """Return the reconstructed atom subgraph for this view (for analysis).

        Args:
            batch_data: Batched MoleculeData

        Returns:
            Batched reconstructed atom-level PyG Data
        """
        device = batch_data.x.device
        fragment_batch = build_fragment_graph_batch(batch_data).to(device)
        sampled_fragment_batch, _ = self.fragment_edge_sampler(fragment_batch)
        sampled_edge_scores, sampled_bool_mask = self.fragment_masker(sampled_fragment_batch)
        full_mask, edge_scores = restore_full_edge_outputs(
            fragment_batch.edge_index.size(1),
            sampled_fragment_batch.full_fragment_edge_id,
            sampled_bool_mask,
            sampled_edge_scores,
        )
        return reconstruct_batch(batch_data, full_mask, selected_fragment_edge_weight=edge_scores)
