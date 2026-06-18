"""Common interfaces for molecular fragmentation strategies."""

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class FragmentationResult:
    """Tensor-friendly result required by the hierarchical model."""

    fragment_to_atoms: Any = None
    atom_to_fragment: Any = None
    fragment_edge_index: Any = None
    fragment_edge_to_bond: Any = None
    fragment_edge_to_bond_index: Any = None
    fragment_edge_to_attachment_atoms: Any = None
    fragment_types: Optional[Any] = None
    fragment_smiles: Optional[Any] = None

    @property
    def num_fragments(self):
        return len(self.fragment_to_atoms) if self.fragment_to_atoms is not None else 0

    @property
    def num_fragment_edges(self):
        if self.fragment_edge_index is None:
            return 0
        return int(self.fragment_edge_index.size(1))


class Fragmenter:
    """Base interface implemented by Group Graph and future ablations."""

    def fragment(self, mol) -> FragmentationResult:
        raise NotImplementedError


def validate_fragmentation(result: FragmentationResult, mol):
    """Validate coverage, non-overlap, indices, and original-bond mappings."""
    if result.fragment_to_atoms is None:
        raise ValueError("fragment_to_atoms is required")
    if result.atom_to_fragment is None:
        raise ValueError("atom_to_fragment is required")
    if result.fragment_edge_index is None:
        raise ValueError("fragment_edge_index is required")
    if result.fragment_edge_to_bond is None:
        raise ValueError("fragment_edge_to_bond is required")
    if result.fragment_edge_to_attachment_atoms is None:
        raise ValueError("fragment_edge_to_attachment_atoms is required")

    num_atoms = mol.GetNumAtoms()
    num_fragments = len(result.fragment_to_atoms)

    if num_fragments == 0:
        raise ValueError("fragmentation must contain at least one fragment")

    if result.fragment_types is not None and len(result.fragment_types) != num_fragments:
        raise ValueError("fragment_types must contain one value per fragment")
    if result.fragment_smiles is not None and len(result.fragment_smiles) != num_fragments:
        raise ValueError("fragment_smiles must contain one value per fragment")

    if result.atom_to_fragment.shape != (num_atoms,):
        raise ValueError(
            f"atom_to_fragment must have shape ({num_atoms},), "
            f"got {tuple(result.atom_to_fragment.shape)}"
        )

    if result.fragment_edge_index.dim() != 2 or result.fragment_edge_index.size(0) != 2:
        raise ValueError("fragment_edge_index must have shape [2, num_fragment_edges]")

    num_edges = result.fragment_edge_index.size(1)
    if result.fragment_edge_to_bond.shape != (num_edges,):
        raise ValueError("fragment_edge_to_bond must contain one value per fragment edge")

    if result.fragment_edge_to_attachment_atoms.shape != (2, num_edges):
        raise ValueError(
            "fragment_edge_to_attachment_atoms must have shape [2, num_fragment_edges]"
        )

    if result.fragment_edge_to_bond_index is not None:
        if result.fragment_edge_to_bond_index.shape != (num_edges,):
            raise ValueError("fragment_edge_to_bond_index must contain one value per edge")

    seen_atoms = []
    for fragment_id, atoms in enumerate(result.fragment_to_atoms):
        atom_tensor = torch.as_tensor(atoms, dtype=torch.long)
        if atom_tensor.numel() == 0:
            raise ValueError(f"fragment {fragment_id} is empty")
        if not atoms_are_connected(mol, atom_tensor.tolist()):
            raise ValueError(f"fragment {fragment_id} is disconnected")
        for atom_id in atom_tensor.tolist():
            if atom_id < 0 or atom_id >= num_atoms:
                raise ValueError(f"fragment {fragment_id} contains invalid atom {atom_id}")
            if int(result.atom_to_fragment[atom_id]) != fragment_id:
                raise ValueError(
                    f"atom_to_fragment[{atom_id}] does not point back to fragment {fragment_id}"
                )
        seen_atoms.extend(atom_tensor.tolist())

    if sorted(seen_atoms) != list(range(num_atoms)):
        raise ValueError("fragments must cover every atom exactly once")

    if num_edges == 0:
        return result

    if result.fragment_edge_index.min().item() < 0:
        raise ValueError("fragment_edge_index contains a negative fragment ID")
    if result.fragment_edge_index.max().item() >= num_fragments:
        raise ValueError("fragment_edge_index contains a fragment ID outside the molecule")

    attachment_atoms = result.fragment_edge_to_attachment_atoms
    if attachment_atoms.min().item() < 0 or attachment_atoms.max().item() >= num_atoms:
        raise ValueError("fragment_edge_to_attachment_atoms contains an invalid atom ID")

    for edge_id in range(num_edges):
        src_fragment = int(result.fragment_edge_index[0, edge_id])
        dst_fragment = int(result.fragment_edge_index[1, edge_id])
        src_atom = int(attachment_atoms[0, edge_id])
        dst_atom = int(attachment_atoms[1, edge_id])
        bond_id = int(result.fragment_edge_to_bond[edge_id])

        if src_fragment == dst_fragment:
            raise ValueError("fragment edges must connect two different fragments")
        if int(result.atom_to_fragment[src_atom]) != src_fragment:
            raise ValueError("source attachment atom is not inside the source fragment")
        if int(result.atom_to_fragment[dst_atom]) != dst_fragment:
            raise ValueError("destination attachment atom is not inside the destination fragment")

        bond = mol.GetBondBetweenAtoms(src_atom, dst_atom)
        if bond is None:
            raise ValueError("fragment edge attachment atoms are not bonded")
        if bond.GetIdx() != bond_id:
            raise ValueError(
                f"fragment edge {edge_id} maps to bond {bond_id}, "
                f"but attachment atoms are joined by bond {bond.GetIdx()}"
            )

    return result


def atoms_are_connected(mol, atom_ids):
    """Return True when atom_ids induce one connected component in mol."""
    atom_ids = set(int(atom_id) for atom_id in atom_ids)
    if len(atom_ids) <= 1:
        return True

    start = min(atom_ids)
    visited = {start}
    stack = [start]
    while stack:
        atom = mol.GetAtomWithIdx(int(stack.pop()))
        for neighbor in atom.GetNeighbors():
            neighbor_id = int(neighbor.GetIdx())
            if neighbor_id in atom_ids and neighbor_id not in visited:
                visited.add(neighbor_id)
                stack.append(neighbor_id)

    return visited == atom_ids


def fragment_smiles_from_atom_sets(mol, fragment_to_atoms):
    """Return canonical fragment SMILES strings aligned with fragment_to_atoms."""
    fragment_smiles = []
    for atoms in fragment_to_atoms:
        atom_ids = torch.as_tensor(atoms, dtype=torch.long).tolist()
        fragment_smiles.append(
            mol_fragment_to_smiles(mol, atom_ids)
        )
    return fragment_smiles


def mol_fragment_to_smiles(mol, atom_ids):
    """Build one fragment SMILES without mutating the canonical molecule."""
    from rdkit import Chem

    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=[int(atom_id) for atom_id in atom_ids],
        canonical=True,
        isomericSmiles=True,
    )
