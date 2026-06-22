"""Ensemble model combining multiple fragment views for invariant subgraph learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.fragment_graph_ciga import FragmentMasker, SimpleGNN
from src.models.fragment_graph import build_fragment_graph_batch
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

    def forward(self, batch_data, return_intermediates=False):
        """
        Stage 2: Aggregate masker edge weights (mean or max): top-k shared subgraph.
        Stage 3: All classifiers predict on the shared subgraph; soft vote (avg softmax).

        Args:
            batch_data: Batched MoleculeData objects
            return_intermediates: If True, also return per-masker scores, shared atom
                                  graph, per-classifier embeddings and logits.

        Returns:
            predictions: [batch_size, num_classes] — log-probabilities (use with F.nll_loss)
            intermediates: (optional) dict with 'fragment_scores', 'bool_mask',
                           'atom_graphs', 'embeddings', 'per_view_logits'
        """
        device = batch_data.x.device
        fragment_batch = build_fragment_graph_batch(batch_data).to(device)

        # Stage 2: each masker scores fragment edges; collect continuous weights
        all_soft_weights = []
        for masker in self.fragment_maskers:
            soft_weights, _ = masker(fragment_batch)
            all_soft_weights.append(soft_weights)

        # Aggregate across maskers, then single top-k → one shared subgraph G_c^bm
        agg_weights = self._aggregate_weights(all_soft_weights)
        edge_batch = fragment_batch.batch[fragment_batch.edge_index[0]]
        bool_mask = FragmentMasker._topk_mask(
            agg_weights.detach(), edge_batch, self.fragment_maskers[0].ratio
        )
        atom_batch = reconstruct_batch(
            batch_data, bool_mask, selected_fragment_edge_weight=agg_weights
        ).to(device)

        # Stage 3: each classifier predicts on the shared subgraph; soft vote
        all_probs = []
        embeddings_list = []
        per_view_logits_list = []
        for classifier in self.atom_classifiers:
            logits, h_graph = classifier(atom_batch, return_embeddings=True)
            all_probs.append(F.softmax(logits, dim=-1))
            embeddings_list.append(h_graph)
            per_view_logits_list.append(logits)

        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)  # [batch, num_classes]
        # log-probs: argmax preserved (log monotone), compatible with F.nll_loss
        predictions = torch.log(avg_probs.clamp(min=1e-8))

        if return_intermediates:
            intermediates = {
                "fragment_scores": all_soft_weights,
                "bool_mask": bool_mask,
                "atom_graphs": [atom_batch] * self.num_views,
                "embeddings": embeddings_list,
                "per_view_logits": per_view_logits_list,
            }
            return predictions, intermediates
        return predictions

    def get_invariant_subgraph(self, batch_data):
        """Return the aggregated invariant atom subgraph for the full ensemble.

        Args:
            batch_data: Batched MoleculeData objects

        Returns:
            Batched reconstructed atom-level PyG Data
        """
        device = batch_data.x.device
        fragment_batch = build_fragment_graph_batch(batch_data).to(device)

        all_soft_weights = []
        for masker in self.fragment_maskers:
            soft_weights, _ = masker(fragment_batch)
            all_soft_weights.append(soft_weights)

        agg_weights = self._aggregate_weights(all_soft_weights)
        edge_batch = fragment_batch.batch[fragment_batch.edge_index[0]]
        bool_mask = FragmentMasker._topk_mask(
            agg_weights.detach(), edge_batch, self.fragment_maskers[0].ratio
        )
        return reconstruct_batch(batch_data, bool_mask, selected_fragment_edge_weight=agg_weights)
