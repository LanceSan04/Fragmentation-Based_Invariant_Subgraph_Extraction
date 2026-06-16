"""Ensemble model combining multiple fragment views for invariant subgraph learning."""

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from src.models.fragment_gnn import SimpleCIGA
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.fragment_reconstruction import batch_reconstruct_from_fragment_edges
from src.data.atom_graph import ATOM_FDIM


class SuGAREnsemble(nn.Module):
    """Sugar Ensemble: multi-view fragment learning with invariant subgraph extraction."""

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
    ):
        """
        Args:
            atom_feature_dim: Dimension of atom features
            fragment_feature_dim: Dimension of fragment features (pooled from atoms)
            num_classes: Number of output classes
            num_gnn_layers: Number of layers in fragment GNN
            gnn_hidden_dim: Hidden dimension for GNN
            edge_ratio: Ratio of fragment edges to keep in invariant subgraph
            num_views: Number of fragment views (for future multi-view extension)
            dropout: Dropout rate
        """
        super().__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.edge_ratio = edge_ratio

        # Fragment-level GNN for each view
        # For now, fragment_feature_dim will be inferred from data
        # but typically it's pooled atom features
        self.fragment_gnns = nn.ModuleList()
        for _ in range(num_views):
            # Will be properly initialized after first forward pass with known dims
            gnn = SimpleCIGA(
                input_dim=fragment_feature_dim or 64,
                hidden_dim=gnn_hidden_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
                ratio=edge_ratio,
            )
            self.fragment_gnns.append(gnn)

        # Classifiers for reconstructed atom-level subgraphs
        self.atom_classifiers = nn.ModuleList()
        for _ in range(num_views):
            classifier = nn.Sequential(
                nn.Linear(atom_feature_dim, gnn_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gnn_hidden_dim, num_classes),
            )
            self.atom_classifiers.append(classifier)

        # Aggregator to combine predictions from all views
        self.aggregator = nn.Sequential(
            nn.Linear(num_views * num_classes, 2 * num_classes),
            nn.ReLU(),
            nn.Linear(2 * num_classes, num_classes),
        )

        self.initialized_fragment_dim = False

    def forward(self, batch_data, return_intermediates=False):
        """
        Process molecular graphs through multi-view fragment learning.
        
        Args:
            batch_data: Batched MoleculeData objects
            return_intermediates: If True, return intermediate results for debugging
            
        Returns:
            predictions: Aggregated predictions for each molecule
            (optional) intermediates: Dict with fragment embeddings, masks, etc.
        """
        device = batch_data.x.device

        # Build fragment-level graphs from molecule batch
        fragment_batch = build_fragment_graph_batch(batch_data)
        fragment_batch = fragment_batch.to(device)

        # Re-initialize fragment GNNs with correct input dimension if needed
        if not self.initialized_fragment_dim:
            fragment_input_dim = fragment_batch.x.size(1)
            for gnn in self.fragment_gnns:
                # Reinit with correct dimension
                gnn.gnn_encoder = gnn.gnn_encoder.__class__(
                    fragment_input_dim,
                    gnn.gnn_encoder.hidden_dim,
                    gnn.gnn_encoder.num_layers,
                    gnn.gnn_encoder.dropout,
                )
            self.initialized_fragment_dim = True

        all_predictions = []
        intermediates = {"fragment_edge_masks": [], "reconstructed_subgraphs": []}

        # Process each view (for now just one, but extensible)
        for view_idx in range(self.num_views):
            # Get fragment-level GNN predictions and edge masks
            frag_pred, fragment_edge_mask = self.fragment_gnns[view_idx](
                fragment_batch, return_edge_mask=True
            )

            # Reconstruct atom-level subgraphs based on selected fragment edges
            reconstructed_atom_graphs = batch_reconstruct_from_fragment_edges(
                batch_data, fragment_edge_mask
            )

            if return_intermediates:
                intermediates["fragment_edge_masks"].append(fragment_edge_mask)
                intermediates["reconstructed_subgraphs"].append(reconstructed_atom_graphs)

            # Pool atom features from reconstructed subgraphs for classification
            view_predictions = []
            for atom_graph in reconstructed_atom_graphs:
                # Mean-pool atom features
                atom_features = atom_graph.x.mean(dim=0, keepdim=True)
                view_pred = self.atom_classifiers[view_idx](atom_features)
                view_predictions.append(view_pred)

            view_predictions = torch.cat(view_predictions, dim=0)  # [batch_size, num_classes]
            all_predictions.append(view_predictions)

        # Concatenate predictions from all views
        combined_preds = torch.cat(all_predictions, dim=-1)  # [batch_size, num_views * num_classes]

        # Aggregate predictions
        final_predictions = self.aggregator(combined_preds)

        if return_intermediates:
            return final_predictions, intermediates

        return final_predictions

    def get_invariant_subgraph(self, batch_data, view_idx=0):
        """Extract invariant subgraph for a given view.
        
        Args:
            batch_data: Batched MoleculeData objects
            view_idx: Which view to extract (default 0)
            
        Returns:
            List of reconstructed atom-level Data objects
        """
        device = batch_data.x.device

        # Build fragment graphs
        fragment_batch = build_fragment_graph_batch(batch_data)
        fragment_batch = fragment_batch.to(device)

        # Get fragment edge mask from GNN
        _, fragment_edge_mask = self.fragment_gnns[view_idx](
            fragment_batch, return_edge_mask=True
        )

        # Reconstruct atoms
        reconstructed = batch_reconstruct_from_fragment_edges(batch_data, fragment_edge_mask)

        return reconstructed