"""Group Graph fragmentation adapted for the project data contract."""

import torch
from rdkit import Chem

from src.data.atom_graph import ATOM_FDIM, BOND_FDIM, atom_features, bond_features
from src.fragmentation.base import (
    FragmentationResult,
    Fragmenter,
    fragment_smiles_from_atom_sets,
    validate_fragmentation,
)
from src.fragmentation.molecule_fragment import find_clusters


class GroupGraphFragmenter(Fragmenter):
    """Project-facing Group Graph fragmenter."""

    def fragment(self, mol) -> FragmentationResult:
        """Return 0-based fragments and lossless atom/bond provenance."""
        if mol is None:
            raise ValueError("Cannot fragment mol=None")

        canonical_mol = Chem.Mol(mol)
        Chem.SanitizeMol(canonical_mol)

        working_mol = Chem.Mol(canonical_mol)
        cluster, _ = find_clusters(working_mol)
        result = build_fragmentation_result(canonical_mol, cluster)
        return validate_fragmentation(result, canonical_mol)


def build_fragmentation_result(mol, cluster):
    """Convert upstream Group Graph clusters into FragmentationResult."""
    ordered_fragments, ordered_types = order_fragments_deterministically(cluster)
    num_atoms = mol.GetNumAtoms()

    atom_to_fragment = torch.empty(num_atoms, dtype=torch.long)
    fragment_to_atoms = []
    for fragment_id, atoms in enumerate(ordered_fragments):
        atom_tensor = torch.tensor(atoms, dtype=torch.long)
        fragment_to_atoms.append(atom_tensor)
        atom_to_fragment[atom_tensor] = fragment_id

    fragment_edge_index, fragment_edge_to_bond, fragment_edge_to_bond_index, attachments = (
        build_unique_fragment_edges(mol, atom_to_fragment)
    )

    return FragmentationResult(
        fragment_to_atoms=fragment_to_atoms,
        atom_to_fragment=atom_to_fragment,
        fragment_edge_index=fragment_edge_index,
        fragment_edge_to_bond=fragment_edge_to_bond,
        fragment_edge_to_bond_index=fragment_edge_to_bond_index,
        fragment_edge_to_attachment_atoms=attachments,
        fragment_types=ordered_types,
        fragment_smiles=fragment_smiles_from_atom_sets(mol, fragment_to_atoms),
    )


def order_fragments_deterministically(cluster):
    """Return a stable fragment order independent of set iteration order."""
    ordered_items = sorted(
        ((tuple(sorted(atoms)), fragment_type) for atoms, fragment_type in cluster.items()),
        key=lambda item: (min(item[0]), len(item[0]), item[0], item[1]),
    )
    ordered_fragments = [atoms for atoms, _ in ordered_items]
    ordered_types = [fragment_type for _, fragment_type in ordered_items]
    return ordered_fragments, ordered_types


def build_fragment_edge_features(
    mol,
    fragmentation_result: FragmentationResult,
):
    """Build attachment-atom, bond, and fragment-size edge features."""
    num_edges = fragmentation_result.num_fragment_edges
    feature_dim = (2 * ATOM_FDIM) + BOND_FDIM + 2
    if num_edges == 0:
        return torch.empty((0, feature_dim), dtype=torch.float32)

    rows = []
    for edge_id in range(num_edges):
        src_fragment = int(fragmentation_result.fragment_edge_index[0, edge_id])
        dst_fragment = int(fragmentation_result.fragment_edge_index[1, edge_id])
        src_atom = int(fragmentation_result.fragment_edge_to_attachment_atoms[0, edge_id])
        dst_atom = int(fragmentation_result.fragment_edge_to_attachment_atoms[1, edge_id])
        bond_id = int(fragmentation_result.fragment_edge_to_bond[edge_id])

        bond = mol.GetBondWithIdx(bond_id)
        src_size = len(fragmentation_result.fragment_to_atoms[src_fragment])
        dst_size = len(fragmentation_result.fragment_to_atoms[dst_fragment])

        rows.append(
            atom_features(mol.GetAtomWithIdx(src_atom))
            + atom_features(mol.GetAtomWithIdx(dst_atom))
            + bond_features(bond)
            + [float(src_size), float(dst_size)]
        )

    return torch.tensor(rows, dtype=torch.float32)


def build_unique_fragment_edges(mol, atom_to_fragment):
    """Build one undirected fragment edge per original inter-fragment bond."""
    edge_pairs = []
    bond_ids = []
    bond_indices = []
    attachment_pairs = []

    bonds = sorted(mol.GetBonds(), key=lambda bond: bond.GetIdx())
    for bond_table_index, bond in enumerate(bonds):
        src_atom = int(bond.GetBeginAtomIdx())
        dst_atom = int(bond.GetEndAtomIdx())
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

        bond_ids.append(int(bond.GetIdx()))
        bond_indices.append(int(bond_table_index))

    if edge_pairs:
        fragment_edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        fragment_edge_to_bond = torch.tensor(bond_ids, dtype=torch.long)
        fragment_edge_to_bond_index = torch.tensor(bond_indices, dtype=torch.long)
        fragment_edge_to_attachment_atoms = (
            torch.tensor(attachment_pairs, dtype=torch.long).t().contiguous()
        )
    else:
        fragment_edge_index = torch.empty((2, 0), dtype=torch.long)
        fragment_edge_to_bond = torch.empty((0,), dtype=torch.long)
        fragment_edge_to_bond_index = torch.empty((0,), dtype=torch.long)
        fragment_edge_to_attachment_atoms = torch.empty((2, 0), dtype=torch.long)

    return (
        fragment_edge_index,
        fragment_edge_to_bond,
        fragment_edge_to_bond_index,
        fragment_edge_to_attachment_atoms,
    )
