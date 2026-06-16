"""Aligned atom-graph and fragment-graph record for one molecule."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MoleculeExample:
    atom_x: Any = None
    atom_edge_index: Any = None
    atom_edge_attr: Any = None
    atom_ids: Any = None
    bond_ids: Any = None
    fragment_x_static: Any = None
    fragment_edge_index: Any = None
    fragment_edge_attr: Any = None
    atom_to_fragment: Any = None
    fragment_to_atoms: Any = None
    fragment_edge_to_bond: Any = None
    fragment_edge_to_attachment_atoms: Any = None
    y: Any = None
    environment_or_group: Optional[Any] = None
    smiles: Optional[str] = None
    molecule_id: Optional[Any] = None

    def validate(self):
        """Validate both graph views and all cross-level provenance."""
        raise NotImplementedError


def build_molecule_example(
    smiles,
    label=None,
    environment_or_group=None,
    molecule_id=None,
):
    """Construct a MoleculeExample from raw molecular inputs."""
    raise NotImplementedError
