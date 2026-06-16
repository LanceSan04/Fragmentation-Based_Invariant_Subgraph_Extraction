"""Aligned atom-graph and fragment-graph record for one molecule."""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from rdkit import Chem

from src.data.atom_graph import mol_to_atom_graph
from src.fragmentation.group_graph import (
    GroupGraphFragmenter,
    build_fragment_edge_features,
)
from src.models.fragment_pooler import FragmentPooler


@dataclass
class MoleculeExample:
    atom_x: Any = None
    bond_index: Any = None
    bond_attr: Any = None
    atom_edge_index: Any = None
    atom_edge_attr: Any = None
    atom_ids: Any = None
    bond_ids: Any = None
    directed_edge_to_bond: Any = None
    fragment_x_static: Any = None
    fragment_edge_index: Any = None
    fragment_edge_attr: Any = None
    atom_to_fragment: Any = None
    fragment_to_atoms: Any = None
    fragment_to_atom_index: Any = None
    fragment_edge_to_bond: Any = None
    fragment_edge_to_bond_index: Any = None
    fragment_edge_to_attachment_atoms: Any = None
    y: Any = None
    environment_or_group: Optional[Any] = None
    smiles: Optional[str] = None
    molecule_id: Optional[Any] = None

    def validate(self):
        """Validate both graph views and all cross-level provenance."""
        required = [
            "atom_x",
            "bond_index",
            "bond_attr",
            "atom_edge_index",
            "atom_edge_attr",
            "atom_ids",
            "bond_ids",
            "directed_edge_to_bond",
            "fragment_x_static",
            "fragment_edge_index",
            "atom_to_fragment",
            "fragment_to_atoms",
            "fragment_to_atom_index",
            "fragment_edge_to_bond",
            "fragment_edge_to_bond_index",
            "fragment_edge_to_attachment_atoms",
        ]
        for field_name in required:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} is required")

        num_atoms = self.atom_x.size(0)
        num_bonds = self.bond_attr.size(0)
        num_fragments = self.fragment_x_static.size(0)
        num_fragment_edges = self.fragment_edge_index.size(1)

        if self.atom_to_fragment.shape != (num_atoms,):
            raise ValueError("atom_to_fragment must contain one value per atom")
        if self.fragment_edge_to_bond.shape != (num_fragment_edges,):
            raise ValueError("fragment_edge_to_bond must contain one value per fragment edge")
        if self.fragment_edge_to_bond_index.shape != (num_fragment_edges,):
            raise ValueError("fragment_edge_to_bond_index must contain one value per fragment edge")
        if self.fragment_edge_to_attachment_atoms.shape != (2, num_fragment_edges):
            raise ValueError("fragment_edge_to_attachment_atoms must have shape [2, num_edges]")
        if self.fragment_to_atom_index.dim() != 2 or self.fragment_to_atom_index.size(0) != 2:
            raise ValueError("fragment_to_atom_index must have shape [2, num_memberships]")

        if self.atom_to_fragment.numel() > 0:
            if int(self.atom_to_fragment.min()) < 0:
                raise ValueError("atom_to_fragment contains a negative fragment ID")
            if int(self.atom_to_fragment.max()) >= num_fragments:
                raise ValueError("atom_to_fragment contains a fragment ID outside fragment_x_static")

        if num_bonds > 0 and self.directed_edge_to_bond.numel() > 0:
            if int(self.directed_edge_to_bond.max()) >= num_bonds:
                raise ValueError("directed_edge_to_bond references a missing bond")

        if num_fragment_edges > 0:
            if int(self.fragment_edge_index.max()) >= num_fragments:
                raise ValueError("fragment_edge_index references a missing fragment")
            if int(self.fragment_edge_to_bond_index.max()) >= num_bonds:
                raise ValueError("fragment_edge_to_bond_index references a missing bond")
            if int(self.fragment_edge_to_attachment_atoms.max()) >= num_atoms:
                raise ValueError("fragment_edge_to_attachment_atoms references a missing atom")

        for fragment_id, atoms in enumerate(self.fragment_to_atoms):
            atoms = torch.as_tensor(atoms, dtype=torch.long)
            if atoms.numel() == 0:
                raise ValueError(f"fragment {fragment_id} is empty")
            if not torch.all(self.atom_to_fragment[atoms] == fragment_id):
                raise ValueError(f"fragment_to_atoms[{fragment_id}] disagrees with atom_to_fragment")

        return self


def build_molecule_example(
    smiles,
    label=None,
    environment_or_group=None,
    molecule_id=None,
):
    """Construct a MoleculeExample from raw molecular inputs."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    atom_graph = mol_to_atom_graph(mol)
    fragmentation = GroupGraphFragmenter().fragment(mol)
    fragment_edge_attr = build_fragment_edge_features(mol, fragmentation)

    pooler = FragmentPooler()
    fragment_x_static = pooler(
        atom_graph.atom_x,
        atom_graph.bond_index,
        atom_graph.bond_attr,
        fragmentation.atom_to_fragment,
        num_fragments=fragmentation.num_fragments,
    )

    if label is None:
        y = None
    else:
        y = torch.as_tensor(label, dtype=torch.float32)
        if y.dim() == 0:
            y = y.view(1)

    return MoleculeExample(
        atom_x=atom_graph.atom_x,
        bond_index=atom_graph.bond_index,
        bond_attr=atom_graph.bond_attr,
        atom_edge_index=atom_graph.atom_edge_index,
        atom_edge_attr=atom_graph.atom_edge_attr,
        atom_ids=atom_graph.atom_ids,
        bond_ids=atom_graph.bond_ids,
        directed_edge_to_bond=atom_graph.directed_edge_to_bond,
        fragment_x_static=fragment_x_static,
        fragment_edge_index=fragmentation.fragment_edge_index,
        fragment_edge_attr=fragment_edge_attr,
        atom_to_fragment=fragmentation.atom_to_fragment,
        fragment_to_atoms=fragmentation.fragment_to_atoms,
        fragment_to_atom_index=build_fragment_to_atom_index(fragmentation.fragment_to_atoms),
        fragment_edge_to_bond=fragmentation.fragment_edge_to_bond,
        fragment_edge_to_bond_index=fragmentation.fragment_edge_to_bond_index,
        fragment_edge_to_attachment_atoms=fragmentation.fragment_edge_to_attachment_atoms,
        y=y,
        environment_or_group=environment_or_group,
        smiles=smiles,
        molecule_id=molecule_id,
    ).validate()


def build_fragment_to_atom_index(fragment_to_atoms):
    """Build a [fragment_id, atom_id] membership index for PyG batching."""
    pairs = []
    for fragment_id, atoms in enumerate(fragment_to_atoms):
        for atom_id in torch.as_tensor(atoms, dtype=torch.long).tolist():
            pairs.append([fragment_id, atom_id])

    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()
