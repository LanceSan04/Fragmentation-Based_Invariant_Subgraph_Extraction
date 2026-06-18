"""Simplified CIGA model adapted from SuGAr for invariant subgraph extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn.conv import MessagePassing


def _set_edge_masks(mask: torch.Tensor, model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = mask


def _clear_edge_masks(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None


class SimpleGNN(nn.Module):
    """Simple GNN encoder for embedding graph nodes."""

    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch=None):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        h_graph = global_mean_pool(h, batch)

        return h, h_graph


class SimpleCIGA(nn.Module):
    """Simplified CIGA model for extracting invariant subgraphs.

    A featurizer GNN produces node embeddings used to compute continuous
    edge-attention weights; those weights are injected into a classifier GNN via
    edge masks (set_masks / clear_masks).  
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.0, ratio=0.5, out_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ratio = ratio
        self.out_dim = out_dim

        # Pass 1: featurizer — produces node embeddings used to score edges
        self.gnn_encoder = SimpleGNN(input_dim, hidden_dim, num_layers, dropout)

        # Edge attention MLP: cat(h_src, h_dst) → continuous scalar in (0, 1)
        self.edge_att = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, 1),
        )

        # Pass 2: classifier GNN — runs on the full graph with soft edge weights
        # injected via set_masks so that gradients reach edge_att
        self.classifier_gnn = SimpleGNN(input_dim, hidden_dim, num_layers, dropout)

        # Prediction head operating on the soft-masked graph embedding
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, out_dim),
        )

    def forward(self, data, return_edge_mask=False):
        """
        Args:
            data: PyG Data object for the fragment graph
            return_edge_mask: If True, also return the binary edge mask
                              (derived from detached scores — for visualization
                               and atom-level reconstruction only)

        Returns:
            pred: Graph-level predictions [num_graphs, 1]
            edge_mask: (optional) Boolean mask [num_edges] — True for kept edges
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # --- Featurizer pass -------------------------------------------------
        h, _ = self.gnn_encoder(x, edge_index, batch)

        # --- Edge attention --------------------------------------------------
        src, dst = edge_index
        edge_rep = torch.cat([h[src], h[dst]], dim=-1)
        edge_weights = torch.sigmoid(self.edge_att(edge_rep).view(-1))

        # --- Hard mask for visualization / reconstruction (detached) ----------
        edge_batch = batch[src]
        edge_mask = self._topk_mask(edge_weights.detach(), edge_batch, self.ratio)

        # --- Classifier pass with soft edge weights --------------------------
        _set_edge_masks(edge_weights, self.classifier_gnn)
        _, h_graph = self.classifier_gnn(x, edge_index, batch)
        _clear_edge_masks(self.classifier_gnn)

        pred = self.pred_head(h_graph)

        if return_edge_mask:
            return pred, edge_mask
        return pred


    @staticmethod
    def _topk_mask(
        scores: torch.Tensor,
        edge_batch: torch.Tensor,
        ratio: float,
    ) -> torch.Tensor:
        """Boolean top-k mask keeping `ratio` fraction of edges per graph.

        Args:
            scores: Detached edge scores [num_edges]
            edge_batch: Graph index for each edge, i.e. batch[src] [num_edges]
            ratio: Fraction of edges to keep (0 < ratio <= 1)

        Returns:
            Boolean mask [num_edges]
        """
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for gid in edge_batch.unique():
            idx = (edge_batch == gid).nonzero(as_tuple=True)[0]
            k = max(1, int(idx.numel() * ratio))
            top_local = scores[idx].topk(min(k, idx.numel()), largest=True).indices
            mask[idx[top_local]] = True
        return mask
