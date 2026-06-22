"""Fragment-level CIGA masker for scoring and selecting fragment-edge instances."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool


class SimpleGNN(nn.Module):
    """GIN/GINE encoder producing node and graph-level embeddings."""

    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.0, edge_dim=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dim = edge_dim

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
            if edge_dim is None:
                self.convs.append(GINConv(mlp))
            else:
                self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        h = x
        for i, conv in enumerate(self.convs):
            if self.edge_dim is None:
                h = conv(h, edge_index)
            else:
                if edge_attr is None:
                    raise ValueError("edge_attr is required for edge-aware SimpleGNN")
                h = conv(h, edge_index, edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        return h, global_mean_pool(h, batch)


class FragmentMasker(nn.Module):
    """CIGA-style module: fragment graph -> edge weights + invariant edge mask.

    The hard top-k mask chooses which fragment-edge instances are reconstructed.
    The continuous scores must still be passed downstream as selected edge
    weights so prediction loss can train the scorer.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.0,
        ratio=0.5,
        edge_dim=None,
    ):
        super().__init__()
        self.ratio = ratio
        self.gnn_encoder = SimpleGNN(input_dim, hidden_dim, num_layers, dropout, edge_dim=edge_dim)
        self.edge_att = nn.Sequential(
            nn.Linear((hidden_dim * 2) + (edge_dim or 0), hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, 1),
        )

    def forward(self, data):
        """
        Args:
            data: Batched PyG Data for the fragment graph.

        Returns:
            soft_weights: Continuous edge scores [num_edges] in (0, 1).
            bool_mask: Boolean top-k mask [num_edges], True for kept edges.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)

        h, _ = self.gnn_encoder(x, edge_index, batch, edge_attr=edge_attr)

        src, dst = edge_index
        edge_rep_pieces = [h[src], h[dst]]
        if self.gnn_encoder.edge_dim is not None:
            edge_rep_pieces.append(edge_attr)
        edge_rep = torch.cat(edge_rep_pieces, dim=-1)
        soft_weights = torch.sigmoid(self.edge_att(edge_rep).view(-1))

        # Top-k indices are discrete. Detaching here only isolates mask
        # construction; selected soft_weights are still passed downstream.
        bool_mask = self._topk_mask(soft_weights.detach(), batch[src], self.ratio)
        return soft_weights, bool_mask

    @staticmethod
    def _topk_mask(scores, edge_batch, ratio):
        """Boolean top-k mask keeping `ratio` fraction of edges per graph."""
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for gid in edge_batch.unique():
            idx = (edge_batch == gid).nonzero(as_tuple=True)[0]
            k = max(1, int(idx.numel() * ratio))
            top_local = scores[idx].topk(min(k, idx.numel()), largest=True).indices
            mask[idx[top_local]] = True
        return mask
