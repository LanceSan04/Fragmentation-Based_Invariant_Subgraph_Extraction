"""BRICS fragmentation adapter using the project's FragmentationResult contract."""

from collections import deque

import torch
from rdkit.Chem import BRICS

from src.fragmentation.base import (
    FragmentationResult,
    Fragmenter,
    fragment_smiles_from_atom_sets,
    validate_fragmentation,
)


class BRICSFragmenter(Fragmenter):
    """Fragment molecules by removing RDKit BRICS bonds."""

    def fragment(self, mol) -> FragmentationResult:
        fragment_to_atoms, broken_bonds = brics_fragment_atom_sets(mol)
        atom_to_fragment = torch.empty(mol.GetNumAtoms(), dtype=torch.long)
        fragment_tensors = []

        for fragment_id, atoms in enumerate(fragment_to_atoms):
            atom_tensor = torch.tensor(atoms, dtype=torch.long)
            fragment_tensors.append(atom_tensor)
            atom_to_fragment[atom_tensor] = fragment_id

        fragment_edge_index, fragment_edge_to_bond, fragment_edge_to_bond_index, attachments = (
            _build_edges_from_broken_bonds(mol, atom_to_fragment, broken_bonds)
        )

        result = FragmentationResult(
            fragment_to_atoms=fragment_tensors,
            atom_to_fragment=atom_to_fragment,
            fragment_edge_index=fragment_edge_index,
            fragment_edge_to_bond=fragment_edge_to_bond,
            fragment_edge_to_bond_index=fragment_edge_to_bond_index,
            fragment_edge_to_attachment_atoms=attachments,
            fragment_types=["brics"] * len(fragment_tensors),
            fragment_smiles=fragment_smiles_from_atom_sets(mol, fragment_tensors),
        )
        return validate_fragmentation(result, mol)


def brics_fragment_atom_sets(mol):
    """Return connected atom components after removing BRICS bonds."""
    broken_bond_specs = list(BRICS.FindBRICSBonds(mol))
    broken_pairs = set()
    broken_bonds = []

    for atom_pair, labels in broken_bond_specs:
        src_atom, dst_atom = int(atom_pair[0]), int(atom_pair[1])
        bond = mol.GetBondBetweenAtoms(src_atom, dst_atom)
        if bond is None:
            continue
        broken_pairs.add(tuple(sorted((src_atom, dst_atom))))
        broken_bonds.append((bond.GetIdx(), src_atom, dst_atom, list(labels)))

    adjacency = {atom.GetIdx(): set() for atom in mol.GetAtoms()}
    for bond in mol.GetBonds():
        src_atom = bond.GetBeginAtomIdx()
        dst_atom = bond.GetEndAtomIdx()
        if tuple(sorted((src_atom, dst_atom))) in broken_pairs:
            continue
        adjacency[src_atom].add(dst_atom)
        adjacency[dst_atom].add(src_atom)

    seen = set()
    components = []
    for atom_id in range(mol.GetNumAtoms()):
        if atom_id in seen:
            continue
        queue = deque([atom_id])
        seen.add(atom_id)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))

    components.sort(key=lambda atoms: (min(atoms), len(atoms), atoms))
    return components, broken_bonds


def _build_edges_from_broken_bonds(mol, atom_to_fragment, broken_bonds):
    edge_pairs = []
    bond_ids = []
    bond_indices = []
    attachment_pairs = []

    bond_index_by_id = {
        bond.GetIdx(): table_index
        for table_index, bond in enumerate(sorted(mol.GetBonds(), key=lambda bond: bond.GetIdx()))
    }

    for bond_id, src_atom, dst_atom, labels in sorted(broken_bonds, key=lambda item: item[0]):
        src_fragment = int(atom_to_fragment[src_atom])
        dst_fragment = int(atom_to_fragment[dst_atom])
        if src_fragment == dst_fragment:
            continue

        if src_fragment <= dst_fragment:
            edge_pairs.append([src_fragment, dst_fragment])
            attachment_pairs.append([src_atom, dst_atom])
        else:
            edge_pairs.append([dst_fragment, src_fragment])
            attachment_pairs.append([dst_atom, src_atom])

        bond_ids.append(int(bond_id))
        bond_indices.append(int(bond_index_by_id[bond_id]))

    if edge_pairs:
        return (
            torch.tensor(edge_pairs, dtype=torch.long).t().contiguous(),
            torch.tensor(bond_ids, dtype=torch.long),
            torch.tensor(bond_indices, dtype=torch.long),
            torch.tensor(attachment_pairs, dtype=torch.long).t().contiguous(),
        )

    return (
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((0,), dtype=torch.long),
        torch.empty((2, 0), dtype=torch.long),
    )
