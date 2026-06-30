"""Shared GNN encoder backbones used across masker and classifier modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, GATv2Conv, global_mean_pool


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


class SimpleGAT(nn.Module):
    """GATv2 encoder producing node and graph-level embeddings."""

    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.0, edge_dim=None, heads=4):
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
            self.convs.append(
                GATv2Conv(in_dim, hidden_dim, heads=heads, concat=False, dropout=dropout, edge_dim=edge_dim)
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        h = x
        for i, conv in enumerate(self.convs):
            if self.edge_dim is None:
                h = conv(h, edge_index)
            else:
                if edge_attr is None:
                    raise ValueError("edge_attr is required for edge-aware SimpleGAT")
                h = conv(h, edge_index, edge_attr=edge_attr)
            h = self.bns[i](h)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        return h, global_mean_pool(h, batch)
