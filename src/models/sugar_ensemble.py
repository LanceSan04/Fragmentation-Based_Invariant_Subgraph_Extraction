"""Ensemble model combining multiple fragment views for invariant subgraph learning."""

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from src.models.simple_ciga import SimpleCIGA
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.fragment_reconstruction import batch_reconstruct_from_fragment_edges
from src.data.atom_graph import ATOM_FDIM


class SuGAREnsemble(nn.Module):
    """Sugar Ensemble: multi-view fragment learning with invariant subgraph extraction.

    Pipeline per view:
      1. Fragment GNN (SimpleCIGA): scores fragment-level edges with continuous soft
         attention weights; produces a detached boolean mask for reconstruction.
      2. Reconstruction: the boolean mask selects which fragment-to-fragment edges
         (and thus which atoms) form the invariant subgraph.
      3. Consolidation: all reconstructed atom subgraphs in the batch are combined
         into a single PyG Batch.
      4. Atom GNN (SimpleCIGA, ratio=1.0): runs on the consolidated atom batch with
         its own soft edge-attention weights injected via set_masks, so gradients
         reach the atom-level edge scorer.
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
    ):
        super().__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.edge_ratio = edge_ratio

        # Fragment-level GNNs — input dim unknown until first batch; lazy-reinit below.
        self.fragment_gnns = nn.ModuleList([
            SimpleCIGA(
                input_dim=fragment_feature_dim or 64,
                hidden_dim=gnn_hidden_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
                ratio=edge_ratio,
                out_dim=num_classes,
            )
            for _ in range(num_views)
        ])

        # Atom-level GNNs — operate on consolidated reconstructed atom subgraphs.
        # ratio=1.0: fragment selection already pruned the graph; use soft attention
        # weights only (no further hard edge removal at atom level).
        self.atom_gnns = nn.ModuleList([
            SimpleCIGA(
                input_dim=atom_feature_dim,
                hidden_dim=gnn_hidden_dim,
                num_layers=num_gnn_layers,
                dropout=dropout,
                ratio=1.0,
                out_dim=num_classes,
            )
            for _ in range(num_views)
        ])

        # Aggregator: combines per-view atom GNN predictions
        self.aggregator = nn.Sequential(
            nn.Linear(num_views * num_classes, 2 * num_classes),
            nn.ReLU(),
            nn.Linear(2 * num_classes, num_classes),
        )

        self.initialized_fragment_dim = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_data, return_intermediates=False):
        """
        Args:
            batch_data: Batched MoleculeData objects
            return_intermediates: If True, also return fragment masks and
                                  reconstructed subgraph lists for debugging

        Returns:
            predictions: [batch_size, num_classes]
            intermediates: (optional) dict with 'fragment_edge_masks' and
                           'reconstructed_subgraphs'
        """
        device = batch_data.x.device

        # Build fragment-level graphs
        fragment_batch = build_fragment_graph_batch(batch_data).to(device)

        # Lazy reinit: fix fragment GNN input dims once we see real fragment features.
        # Skipped when the model was already built with the correct input_dim.
        if not self.initialized_fragment_dim:
            fragment_input_dim = fragment_batch.x.size(1)
            if fragment_input_dim != self.fragment_gnns[0].gnn_encoder.input_dim:
                for gnn in self.fragment_gnns:
                    gnn.gnn_encoder = gnn.gnn_encoder.__class__(
                        fragment_input_dim,
                        gnn.gnn_encoder.hidden_dim,
                        gnn.gnn_encoder.num_layers,
                        gnn.gnn_encoder.dropout,
                    )
                    gnn.classifier_gnn = gnn.classifier_gnn.__class__(
                        fragment_input_dim,
                        gnn.classifier_gnn.hidden_dim,
                        gnn.classifier_gnn.num_layers,
                        gnn.classifier_gnn.dropout,
                    )
            self.initialized_fragment_dim = True

        all_predictions = []
        intermediates = {"fragment_edge_masks": [], "reconstructed_subgraphs": []}

        for view_idx in range(self.num_views):
            # ---- Step 1: Fragment-level edge attention -------------------------
            # Returns (pred, bool_mask). The bool_mask is derived from detached
            # scores and used only for reconstruction + visualization.
            _, fragment_edge_mask = self.fragment_gnns[view_idx](
                fragment_batch, return_edge_mask=True
            )

            # ---- Step 2: Reconstruct atom subgraphs ---------------------------
            reconstructed_atom_graphs = batch_reconstruct_from_fragment_edges(
                batch_data, fragment_edge_mask
            )  # list of Data, one per molecule

            if return_intermediates:
                intermediates["fragment_edge_masks"].append(fragment_edge_mask)
                intermediates["reconstructed_subgraphs"].append(reconstructed_atom_graphs)

            # ---- Step 3: Consolidate into a single batched atom graph ---------
            batched_atoms = Batch.from_data_list(reconstructed_atom_graphs).to(device)

            # ---- Step 4: Atom GNN with soft edge masks ------------------------
            # SimpleCIGA internally runs set_masks / clear_masks on classifier_gnn,
            # so gradients flow: loss → pred_head → classifier_gnn → edge_att.
            view_predictions = self.atom_gnns[view_idx](batched_atoms)  # [batch_size, num_classes]

            all_predictions.append(view_predictions)

        # ---- Aggregate across views -------------------------------------------
        combined_preds = torch.cat(all_predictions, dim=-1)  # [batch_size, num_views * num_classes]
        final_predictions = self.aggregator(combined_preds)

        if return_intermediates:
            return final_predictions, intermediates

        return final_predictions

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_invariant_subgraph(self, batch_data, view_idx=0):
        """Return reconstructed atom subgraphs for the given view (no grad).

        Args:
            batch_data: Batched MoleculeData objects
            view_idx: Which fragment GNN view to use

        Returns:
            list of reconstructed atom-level Data objects
        """
        device = batch_data.x.device
        fragment_batch = build_fragment_graph_batch(batch_data).to(device)

        _, fragment_edge_mask = self.fragment_gnns[view_idx](
            fragment_batch, return_edge_mask=True
        )

        return batch_reconstruct_from_fragment_edges(batch_data, fragment_edge_mask)
