"""Shared GNN encoder backbones used across masker and classifier modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, GATv2Conv, global_add_pool, global_mean_pool


def masked_batch_norm(layer, values, mask):
    """Use active rows for statistics; keep inactive rows for surrogate gradients."""
    active = mask.detach() > 0.5
    if not active.any():
        raise ValueError("Masked graph batch has no active atoms")
    selected = values[active]
    normalized = layer(selected)
    if layer.training:
        mean = selected.mean(0)
        variance = selected.var(0, unbiased=False)
    else:
        mean, variance = layer.running_mean, layer.running_var
    full = (values - mean) / torch.sqrt(variance + layer.eps)
    if layer.affine:
        full = full * layer.weight + layer.bias
    return full.index_copy(0, active.nonzero(as_tuple=True)[0], normalized)


class SimpleGNN(nn.Module):
    """GIN/GINE encoder producing node and graph-level embeddings."""

    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.0, edge_dim=None, readout='mean'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dim = edge_dim
        if readout not in ('mean', 'sum', 'mean_count'):
            raise ValueError(f'Unsupported graph readout: {readout}')
        self.readout = readout

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

    def forward(self, x, edge_index, batch=None, edge_attr=None, node_mask=None):
        h = x
        for i, conv in enumerate(self.convs):
            if node_mask is not None:
                h = h * node_mask[:, None]
            if node_mask is None:
                if self.edge_dim is None:
                    h = conv(h, edge_index)
                else:
                    if edge_attr is None:
                        raise ValueError("edge_attr is required for edge-aware SimpleGNN")
                    h = conv(h, edge_index, edge_attr)
            else:
                # Same GIN/GINE update, but both BatchNorm locations use only
                # hard-active rows. No module mutation or temporary hooks.
                kwargs = {} if self.edge_dim is None else {"edge_attr": edge_attr}
                aggregate = conv.propagate(edge_index, x=(h, h), size=None, **kwargs)
                h = aggregate + (1 + conv.eps) * h
                for layer in conv.nn:
                    h = masked_batch_norm(layer, h, node_mask) if isinstance(layer, nn.BatchNorm1d) else layer(h)
            h = self.bns[i](h) if node_mask is None else masked_batch_norm(self.bns[i], h, node_mask)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        if node_mask is None and self.readout == "mean":
            return h, global_mean_pool(h, batch)
        if node_mask is None and self.readout == "sum":
            return h, global_add_pool(h, batch)
        if self.readout == "mean":
            count = int(batch.max()) + 1
            total = h.new_zeros((count, h.size(1))).index_add(0, batch, h * node_mask[:, None])
            weight = h.new_zeros(count).index_add(0, batch, node_mask)
            return h, total / weight.clamp_min(1e-8)[:, None]
        weights = h.new_ones(h.size(0)) if node_mask is None else node_mask
        total = global_add_pool(h * weights[:, None], batch)
        if self.readout == "sum":
            return h, total
        count = global_add_pool(weights[:, None], batch)
        mean = total / count.clamp_min(1e-8)
        if self.readout == "mean_count":
            mean = torch.cat((mean, count.log1p()), dim=-1)
        return h, mean


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
                GATv2Conv(
                    in_dim,
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
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
