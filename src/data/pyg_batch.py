"""PyTorch Geometric batching helpers for aligned molecular graph views."""

import torch
from torch_geometric.data import Data


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

    def __inc__(self, key, value, *args, **kwargs):
        """Offset every atom-, fragment-, and bond-indexed mapping correctly."""
        raise NotImplementedError


def molecule_example_to_data(example) -> MoleculeData:
    """Convert MoleculeExample into the project-specific PyG data object."""
    raise NotImplementedError


def batch_molecules(examples):
    """Batch MoleculeExample objects while preserving molecule-local mappings."""
    raise NotImplementedError


def split_batch_by_molecule(batch):
    """Separate batched mappings or model outputs back into molecules."""
    raise NotImplementedError
