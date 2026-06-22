"""Fragment-level CIGA masker: maps a fragment graph to an invariant boolean edge mask."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool


class SimpleGNN(nn.Module):
    """GIN-based encoder producing node and graph-level embeddings."""

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
        return h, global_mean_pool(h, batch)


class FragmentMasker(nn.Module):
    """CIGA-style module: fragment graph → continuous edge weights + boolean invariant mask.

    The boolean mask (derived from detached scores) selects the top-k fraction of
    fragment edges per molecule for downstream atom-level reconstruction.
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.0, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.gnn_encoder = SimpleGNN(input_dim, hidden_dim, num_layers, dropout)
        self.edge_att = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, 1),
        )

    def forward(self, data):
        """
        Args:
            data: Batched PyG Data for the fragment graph

        Returns:
            soft_weights: Continuous edge scores [num_edges] in (0, 1)
            bool_mask:    Boolean top-k mask [num_edges] — True for kept edges
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h, _ = self.gnn_encoder(x, edge_index, batch)

        src, dst = edge_index
        edge_rep = torch.cat([h[src], h[dst]], dim=-1)
        soft_weights = torch.sigmoid(self.edge_att(edge_rep).view(-1))

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
