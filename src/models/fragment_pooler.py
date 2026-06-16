"""No-GNN pooling from raw atom and internal-bond features to fragments."""

import torch.nn as nn


class FragmentPooler(nn.Module):
    """Create initial supernode vectors for the fragment-level GNN."""

    def __init__(self, pooling="mean", include_size_features=True):
        super().__init__()
        self.pooling = pooling
        self.include_size_features = include_size_features

    def forward(
        self,
        atom_x,
        atom_edge_index,
        atom_edge_attr,
        atom_to_fragment,
        num_fragments=None,
    ):
        raise NotImplementedError


def pool_atom_features(atom_x, atom_to_fragment, num_fragments=None):
    """Mean-pool raw atom features into their assigned fragments."""
    raise NotImplementedError


def find_internal_bond_mask(atom_edge_index, atom_to_fragment):
    """Identify atom edges whose endpoints belong to the same fragment."""
    raise NotImplementedError


def pool_internal_bond_features(
    atom_edge_index,
    atom_edge_attr,
    atom_to_fragment,
    num_fragments=None,
):
    """Mean-pool internal bond features and zero-fill bondless fragments."""
    raise NotImplementedError


def fragment_size_features(atom_to_fragment, internal_bond_mask, num_fragments=None):
    """Return atom-count and internal-bond-count features per fragment."""
    raise NotImplementedError
