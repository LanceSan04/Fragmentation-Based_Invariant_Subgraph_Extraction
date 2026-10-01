"""PyTorch Geometric batching helpers for aligned molecular graph views."""

import torch
from torch_geometric.data import Data

FRAGMENT_VIEW_TENSOR_FIELDS = {
    "fragment_x_static",
    "fragment_edge_index",
    "fragment_edge_attr",
    "atom_to_fragment",
    "fragment_to_atom_index",
    "fragment_edge_to_bond",
    "fragment_edge_to_bond_index",
    "fragment_edge_to_attachment_atoms",
}

FRAGMENT_VIEW_LIST_FIELDS = {
    "fragment_types",
    "fragment_smiles",
}


class MoleculeData(Data):
    """Project-specific PyG record with atom/fragment provenance."""

    @property
    def num_fragments(self):
        if hasattr(self, "fragment_x_static") and self.fragment_x_static is not None:
            return int(self.fragment_x_static.size(0))
        return 0

    @property
    def num_bonds(self):
        if hasattr(self, "bond_attr") and self.bond_attr is not None:
            return int(self.bond_attr.size(0))
        return 0

    def num_fragments_for_method(self, method=None):
        field = fragment_view_field(method, "fragment_x_static")
        if hasattr(self, field) and getattr(self, field) is not None:
            return int(getattr(self, field).size(0))
        return self.num_fragments

    def __inc__(self, key, value, *args, **kwargs):
        """Offset every atom-, fragment-, and bond-indexed mapping correctly."""
        if key == "bond_index":
            return self.num_nodes
        if key == "directed_edge_to_bond":
            return self.num_bonds
        method, base_key = split_fragment_view_field(key)
        if base_key == "atom_to_fragment":
            return self.num_fragments_for_method(method)
        if base_key == "fragment_edge_index":
            return self.num_fragments_for_method(method)
        if base_key == "fragment_edge_to_attachment_atoms":
            return self.num_nodes
        if base_key == "fragment_edge_to_bond_index":
            return self.num_bonds
        if base_key == "fragment_to_atom_index":
            return torch.tensor([[self.num_fragments_for_method(method)], [self.num_nodes]])
        if key in {"atom_ids", "bond_ids"} or base_key == "fragment_edge_to_bond":
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        """Concatenate custom edge/index tensors along their column dimension."""
        _, base_key = split_fragment_view_field(key)
        if base_key in {
            "bond_index",
            "fragment_edge_index",
            "fragment_edge_to_attachment_atoms",
            "fragment_to_atom_index",
        }:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def molecule_example_to_data(example) -> MoleculeData:
    """Convert MoleculeExample into the project-specific PyG data object."""
    data_kwargs = {
        "x": example.atom_x,
        "edge_index": example.atom_edge_index,
        "edge_attr": example.atom_edge_attr,
        "bond_index": example.bond_index,
        "bond_attr": example.bond_attr,
        "atom_ids": example.atom_ids,
        "bond_ids": example.bond_ids,
        "directed_edge_to_bond": example.directed_edge_to_bond,
        "fragment_x_static": example.fragment_x_static,
        "fragment_edge_index": example.fragment_edge_index,
        "fragment_edge_attr": example.fragment_edge_attr,
        "atom_to_fragment": example.atom_to_fragment,
        "fragment_to_atom_index": example.fragment_to_atom_index,
        "fragment_edge_to_bond": example.fragment_edge_to_bond,
        "fragment_edge_to_bond_index": example.fragment_edge_to_bond_index,
        "fragment_edge_to_attachment_atoms": example.fragment_edge_to_attachment_atoms,
    }

    if example.fragment_types is not None:
        data_kwargs["fragment_types"] = list(example.fragment_types)
    if example.fragment_smiles is not None:
        data_kwargs["fragment_smiles"] = list(example.fragment_smiles)
    if example.y is not None:
        data_kwargs["y"] = example.y
    if example.environment_or_group is not None:
        data_kwargs["environment_or_group"] = example.environment_or_group
    if example.smiles is not None:
        data_kwargs["smiles"] = example.smiles
    if example.molecule_id is not None:
        data_kwargs["molecule_id"] = example.molecule_id
    if example.metadata is not None:
        data_kwargs["metadata"] = dict(example.metadata)
    for method, view in (example.fragment_views or {}).items():
        for field in FRAGMENT_VIEW_TENSOR_FIELDS:
            data_kwargs[fragment_view_field(method, field)] = view[field]
        for field in FRAGMENT_VIEW_LIST_FIELDS:
            if view.get(field) is not None:
                data_kwargs[fragment_view_field(method, field)] = list(view[field])

    return MoleculeData(**data_kwargs)


def fragment_view_field(method, field):
    """Return the Data attribute for one fragmentation method and base field."""
    if method is None:
        return field
    return f"{method}_{field}"


def split_fragment_view_field(key):
    """Return ``(method, base_field)`` for default or method-prefixed fragment fields."""
    if key in FRAGMENT_VIEW_TENSOR_FIELDS or key in FRAGMENT_VIEW_LIST_FIELDS:
        return None, key
    for field in sorted(
        FRAGMENT_VIEW_TENSOR_FIELDS | FRAGMENT_VIEW_LIST_FIELDS, key=len, reverse=True
    ):
        suffix = f"_{field}"
        if key.endswith(suffix):
            return key[: -len(suffix)], field
    return None, key
