"""PyTorch Geometric batching helpers for aligned molecular graph views."""

import torch
from torch_geometric.data import Batch, Data


class MultiGraphData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None,
                 fg_x=None, fg_edge_index=None, atom2fg_index=None,
                 **kwargs):
        super(MultiGraphData, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self.fg_x = fg_x
        self.fg_edge_index = fg_edge_index
        self.num_fgs = fg_x.size(0) if fg_x is not None else None
        self.atom2fg_index = atom2fg_index
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key, value):
        r"""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only `*index*` and `*face*` attributes should be cumulatively summed
        # up when creating batches.
        if key == 'fg_edge_index':
            return self.fg_x.size(0)
        elif key == 'atom2fg_index':
            return torch.tensor([[self.num_nodes], [self.fg_x.size(0)]])
        else:
            return super(MultiGraphData, self).__inc__(key, value)


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

    def __inc__(self, key, value, *args, **kwargs):
        """Offset every atom-, fragment-, and bond-indexed mapping correctly."""
        if key == "bond_index":
            return self.num_nodes
        if key == "directed_edge_to_bond":
            return self.num_bonds
        if key == "atom_to_fragment":
            return self.num_fragments
        if key == "fragment_edge_index":
            return self.num_fragments
        if key == "fragment_edge_to_attachment_atoms":
            return self.num_nodes
        if key == "fragment_edge_to_bond_index":
            return self.num_bonds
        if key == "fragment_to_atom_index":
            return torch.tensor([[self.num_fragments], [self.num_nodes]])
        if key in {"atom_ids", "bond_ids", "fragment_edge_to_bond"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        """Concatenate custom edge/index tensors along their column dimension."""
        if key in {
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

    if example.y is not None:
        data_kwargs["y"] = example.y
    if example.environment_or_group is not None:
        data_kwargs["environment_or_group"] = example.environment_or_group
    if example.smiles is not None:
        data_kwargs["smiles"] = example.smiles
    if example.molecule_id is not None:
        data_kwargs["molecule_id"] = example.molecule_id

    return MoleculeData(**data_kwargs)


def batch_molecules(examples):
    """Batch MoleculeExample objects while preserving molecule-local mappings."""
    data_list = [
        molecule_example_to_data(example) if not isinstance(example, Data) else example
        for example in examples
    ]
    return Batch.from_data_list(data_list)


def split_batch_by_molecule(batch):
    """Separate batched mappings or model outputs back into molecules."""
    if hasattr(batch, "to_data_list"):
        return batch.to_data_list()
    raise TypeError("Expected a PyG Batch with a to_data_list() method")
