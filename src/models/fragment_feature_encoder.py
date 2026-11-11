"""Trainable atom encoding with internal or molecular context; raw caches stay unchanged."""
import torch
from torch import nn
from torch_geometric.nn import GINEConv

from src.models.fragment_pooler import pool_atom_features, find_internal_bond_mask


class InternalFragmentEncoder(nn.Module):
    """Replace the raw-atom mean block, retaining raw bond means and counts.

    With internal context, no cross-fragment edges or batch statistics transmit
    information between fragments. Molecular context includes all input bonds.
    LayerNorm acts on one atom at a time in both modes. The output atom block
    keeps its raw width, so every mode reuses exactly the same raw cache.
    """

    def __init__(self, atom_dim, bond_dim, hidden_dim, layers=2, context="internal"):
        super().__init__()
        if layers < 1:
            raise ValueError("fragment_encoder_layers must be positive")
        self.atom_dim = atom_dim
        if context not in ("internal", "molecular"):
            raise ValueError("Invalid atom encoder context")
        self.context = context
        self.input = nn.Linear(atom_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim)), edge_dim=bond_dim)
            for _ in range(layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.output = nn.Linear(hidden_dim, atom_dim)

    def encode_atoms(self, atom_batch, atom_to_fragment):
        mask = (find_internal_bond_mask(atom_batch.edge_index, atom_to_fragment)
                if self.context == "internal" else
                torch.ones(atom_batch.edge_index.size(1), dtype=torch.bool, device=atom_batch.x.device))
        edges, attributes = atom_batch.edge_index[:, mask], atom_batch.edge_attr[mask]
        h = self.input(atom_batch.x)
        for conv, norm in zip(self.convs, self.norms):
            h = torch.relu(norm(conv(h, edges, attributes)))
        return self.output(h)

    def forward(self, atom_batch, atom_to_fragment, static_features, atom_embeddings=None):
        h = self.encode_atoms(atom_batch, atom_to_fragment) if atom_embeddings is None else atom_embeddings
        pooled = pool_atom_features(h, atom_to_fragment, static_features.size(0))
        return torch.cat([pooled, static_features[:, self.atom_dim:]], dim=-1)
