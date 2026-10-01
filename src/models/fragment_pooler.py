"""No-GNN pooling from raw atom and internal-bond features to fragments."""

import torch
import torch.nn as nn

from src.data.atom_graph import ATOM_FDIM, BOND_FDIM

POOLING_MEAN_ATOM_AND_INTERNAL_BOND = "mean_atom_and_internal_bond"
SUPPORTED_POOLING_MODES = {POOLING_MEAN_ATOM_AND_INTERNAL_BOND}
FRAGMENT_FEATURE_DIM = ATOM_FDIM + BOND_FDIM + 2


class FragmentPooler(nn.Module):
    """Create initial supernode vectors for the fragment-level GNN."""

    def __init__(self, pooling=POOLING_MEAN_ATOM_AND_INTERNAL_BOND, include_size_features=True):
        super().__init__()
        if pooling not in SUPPORTED_POOLING_MODES:
            available = ", ".join(sorted(SUPPORTED_POOLING_MODES))
            raise ValueError(f"Unsupported fragment pooling {pooling!r}. Available: {available}")
        self.pooling = pooling
        self.include_size_features = include_size_features

    def forward(
        self,
        atom_x,
        bond_index,
        bond_attr,
        atom_to_fragment,
        num_fragments=None,
    ):
        atom_pooled = pool_atom_features(atom_x, atom_to_fragment, num_fragments)
        bond_pooled, internal_bond_mask = pool_internal_bond_features(
            bond_index,
            bond_attr,
            atom_to_fragment,
            num_fragments,
            return_mask=True,
        )

        pieces = [atom_pooled, bond_pooled]
        if self.include_size_features:
            pieces.append(
                fragment_size_features(
                    atom_to_fragment,
                    internal_bond_mask,
                    num_fragments,
                    bond_index=bond_index,
                )
            )

        return torch.cat(pieces, dim=-1)


def pool_atom_features(atom_x, atom_to_fragment, num_fragments=None):
    """Mean-pool raw atom features into their assigned fragments."""
    if atom_x.dim() != 2:
        raise ValueError("atom_x must have shape [num_atoms, feature_dim]")
    if atom_to_fragment.dim() != 1 or atom_to_fragment.numel() != atom_x.size(0):
        raise ValueError("atom_to_fragment must contain one fragment ID per atom")

    num_fragments = _infer_num_fragments(atom_to_fragment, num_fragments)
    pooled = atom_x.new_zeros((num_fragments, atom_x.size(1)))
    counts = atom_x.new_zeros((num_fragments, 1))

    pooled.index_add_(0, atom_to_fragment, atom_x)
    counts.index_add_(0, atom_to_fragment, atom_x.new_ones((atom_x.size(0), 1)))
    counts = counts.clamp_min(1.0)
    return pooled / counts


def find_internal_bond_mask(bond_index, atom_to_fragment):
    """Identify atom edges whose endpoints belong to the same fragment."""
    if bond_index.numel() == 0:
        return torch.zeros((0,), dtype=torch.bool, device=atom_to_fragment.device)
    if bond_index.dim() != 2 or bond_index.size(0) != 2:
        raise ValueError("bond_index must have shape [2, num_bonds]")

    src_fragments = atom_to_fragment[bond_index[0]]
    dst_fragments = atom_to_fragment[bond_index[1]]
    return src_fragments == dst_fragments


def pool_internal_bond_features(
    bond_index,
    bond_attr,
    atom_to_fragment,
    num_fragments=None,
    return_mask=False,
):
    """Mean-pool internal bond features and zero-fill bondless fragments."""
    if bond_attr.dim() != 2:
        raise ValueError("bond_attr must have shape [num_bonds, feature_dim]")
    if bond_index.shape != (2, bond_attr.size(0)):
        raise ValueError("bond_index must have shape [2, num_bonds]")

    num_fragments = _infer_num_fragments(atom_to_fragment, num_fragments)
    internal_bond_mask = find_internal_bond_mask(bond_index, atom_to_fragment)
    pooled = bond_attr.new_zeros((num_fragments, bond_attr.size(1)))
    counts = bond_attr.new_zeros((num_fragments, 1))

    if internal_bond_mask.any():
        internal_bonds = bond_index[:, internal_bond_mask]
        internal_features = bond_attr[internal_bond_mask]
        internal_fragments = atom_to_fragment[internal_bonds[0]]
        pooled.index_add_(0, internal_fragments, internal_features)
        counts.index_add_(
            0,
            internal_fragments,
            bond_attr.new_ones((internal_features.size(0), 1)),
        )

    pooled = pooled / counts.clamp_min(1.0)
    if return_mask:
        return pooled, internal_bond_mask
    return pooled


def fragment_size_features(
    atom_to_fragment, internal_bond_mask, num_fragments=None, bond_index=None
):
    """Return atom-count and internal-bond-count features per fragment."""
    num_fragments = _infer_num_fragments(atom_to_fragment, num_fragments)
    features = torch.zeros((num_fragments, 2), dtype=torch.float32, device=atom_to_fragment.device)

    atom_counts = torch.bincount(atom_to_fragment, minlength=num_fragments).to(torch.float32)
    features[:, 0] = atom_counts

    if internal_bond_mask.numel() > 0:
        if bond_index is None:
            raise ValueError("bond_index is required to count internal bonds")
        internal_bonds = bond_index[:, internal_bond_mask]
        internal_fragments = atom_to_fragment[internal_bonds[0]]
        bond_counts = torch.bincount(internal_fragments, minlength=num_fragments).to(torch.float32)
        features[:, 1] = bond_counts

    return features


def _infer_num_fragments(atom_to_fragment, num_fragments=None):
    if atom_to_fragment.numel() == 0:
        if num_fragments is None:
            raise ValueError("num_fragments is required when atom_to_fragment is empty")
        return int(num_fragments)

    inferred = int(atom_to_fragment.max().item()) + 1
    if num_fragments is None:
        return inferred
    if int(num_fragments) < inferred:
        raise ValueError("num_fragments is smaller than the maximum atom_to_fragment value")
    return int(num_fragments)
