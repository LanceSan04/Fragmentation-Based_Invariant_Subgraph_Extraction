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

FRAGMENT_EDGE_FEATURE_DIM = (2 * ATOM_FDIM) + BOND_FDIM + 2


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
    ordered_fragments, ordered_types = partition_overlapping_clusters(mol, cluster)
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


def partition_overlapping_clusters(mol, cluster):
    """Resolve raw Group Graph cluster overlaps into a molecule-wide partition.

    The copied Group Graph `find_clusters` output can contain overlapping candidate
    cluster atom sets. Original Group Graph collapses those overlaps through
    `atom_cls`, where later matching clusters overwrite earlier assignments. This
    project instead makes the overlap policy explicit because pooling, batching, and
    atom-level reconstruction require every atom to belong to exactly one fragment
    with predictable provenance.

    Higher-priority clusters claim atoms first and lower-priority clusters keep only
    unclaimed atoms. Priority is project-specific: split fragments first, larger
    clusters first, then stable atom-index tie-breakers. Any atom missed by upstream
    rules is added as a connected fallback fragment.
    """
    candidates = sorted(
        ((tuple(sorted(atoms)), fragment_type) for atoms, fragment_type in cluster.items()),
        key=_cluster_priority,
    )

    assigned = set()
    partition = []
    for atoms, fragment_type in candidates:
        residual_atoms = tuple(atom for atom in atoms if atom not in assigned)
        if not residual_atoms:
            continue
        for component in connected_components_for_atoms(mol, residual_atoms):
            partition.append((component, fragment_type))
            assigned.update(component)

    missing_atoms = set(range(mol.GetNumAtoms())).difference(assigned)
    for component in connected_components_for_atoms(mol, missing_atoms):
        partition.append((component, "frag"))

    partition.sort(key=lambda item: (min(item[0]), len(item[0]), item[0], item[1]))
    ordered_fragments = [atoms for atoms, _ in partition]
    ordered_types = [fragment_type for _, fragment_type in partition]
    return ordered_fragments, ordered_types


def _cluster_priority(item):
    atoms, fragment_type = item
    type_priority = 0 if fragment_type == "split_frag" else 1
    return (type_priority, -len(atoms), min(atoms), atoms, fragment_type)


def connected_components_for_atoms(mol, atom_ids):
    """Return connected components induced by atom_ids."""
    remaining = set(atom_ids)
    components = []
    while remaining:
        start = min(remaining)
        stack = [start]
        remaining.remove(start)
        component = []
        while stack:
            atom_id = stack.pop()
            component.append(atom_id)
            atom = mol.GetAtomWithIdx(int(atom_id))
            for neighbor in atom.GetNeighbors():
                neighbor_id = int(neighbor.GetIdx())
                if neighbor_id in remaining:
                    remaining.remove(neighbor_id)
                    stack.append(neighbor_id)
        components.append(tuple(sorted(component)))
    return sorted(components, key=lambda atoms: (min(atoms), len(atoms), atoms))


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
    if num_edges == 0:
        return torch.empty((0, FRAGMENT_EDGE_FEATURE_DIM), dtype=torch.float32)

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
