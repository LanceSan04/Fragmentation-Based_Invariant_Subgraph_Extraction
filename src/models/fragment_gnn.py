"""Simplified CIGA model adapted from SuGAr for invariant subgraph extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.data import Batch


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
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            h: Node embeddings [num_nodes, hidden_dim]
            h_graph: Graph embeddings [num_graphs, hidden_dim]
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        if batch is None:
            h_graph = global_mean_pool(h, torch.zeros(h.size(0), dtype=torch.long, device=h.device))
        else:
            h_graph = global_mean_pool(h, batch)

        return h, h_graph


class EdgeScorer(nn.Module):
    """Score edges based on node embeddings."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 1),
        )

    def forward(self, h, edge_index):
        """
        Args:
            h: Node embeddings [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            edge_scores: Edge importance scores [num_edges]
        """
        src, dst = edge_index
        edge_h = torch.cat([h[src], h[dst]], dim=-1)
        edge_scores = self.edge_mlp(edge_h).squeeze(-1)
        return edge_scores


class SimpleCIGA(nn.Module):
    """Simplified CIGA model for extracting invariant subgraphs."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.0, ratio=0.5):
        """
        Args:
            input_dim: Input node feature dimension
            hidden_dim: Hidden dimension for GNN
            num_layers: Number of GNN layers
            dropout: Dropout rate
            ratio: Ratio of edges to keep in invariant subgraph (top-k)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ratio = ratio

        self.gnn_encoder = SimpleGNN(input_dim, hidden_dim, num_layers, dropout)
        self.edge_scorer = EdgeScorer(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 1),
        )

    def forward(self, data, return_edge_mask=False):
        """
        Args:
            data: PyG Data object for fragment graph
            return_edge_mask: If True, also return the binary edge mask
            
        Returns:
            pred: Predictions
            edge_mask: (optional) Binary mask for selected edges
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encode the fragment graph
        h, h_graph = self.gnn_encoder(x, edge_index, batch)

        # Score edges for importance
        edge_scores = self.edge_scorer(h, edge_index)

        # Select top-k edges (invariant subgraph)
        edge_mask = self._select_top_k_edges(edge_scores, edge_index, batch, self.ratio)

        # Make predictions on graph embeddings
        pred = self.classifier(h_graph)

        if return_edge_mask:
            return pred, edge_mask
        return pred

    def _select_top_k_edges(self, edge_scores, edge_index, batch, ratio):
        """Select top-k edges per graph based on edge importance scores.
        
        Args:
            edge_scores: Importance scores for each edge
            edge_index: Edge indices
            batch: Batch assignment for nodes
            ratio: Ratio of edges to keep
            
        Returns:
            mask: Binary mask indicating selected edges
        """
        device = edge_scores.device
        src, dst = edge_index

        # Get graph assignment for edges (based on source node)
        edge_batch = batch[src]

        # For each graph, compute top-k edges
        mask = torch.zeros_like(edge_scores, dtype=torch.bool)

        for graph_id in batch.unique():
            graph_mask = (edge_batch == graph_id)
            graph_edge_scores = edge_scores[graph_mask]

            if graph_edge_scores.numel() > 0:
                k = max(1, int(graph_edge_scores.numel() * ratio))
                _, top_indices = torch.topk(graph_edge_scores, k=min(k, graph_edge_scores.numel()))

                local_positions = torch.arange(graph_mask.sum(), device=device)[top_indices]
                global_positions = torch.nonzero(graph_mask, as_tuple=True)[0][local_positions]
                mask[global_positions] = True

        return mask
