"""RECAP fragmentation adapter using the project's FragmentationResult contract.

RDKit's RecapDecompose returns a decomposition hierarchy whose product
fragments can contain dummy atoms and can omit linker atoms from some cleavage
rules. This adapter instead projects the RECAP cleavage rules back onto the
original molecule: every matched RECAP boundary bond is removed, then connected
components over the original atoms become fragments. That keeps the
non-overlapping, all-atoms-covered invariant required by reconstruction,
batching, and fragment-level diagnostics.
"""

from collections import deque

import torch
from rdkit import Chem

from src.fragmentation.base import (
    FragmentationResult,
    Fragmenter,
    fragment_smiles_from_atom_sets,
    validate_fragmentation,
)


RECAP_FRAGMENT_TYPE = "recap"


# Bond-level projection of the RDKit RECAP reaction SMARTS. Each cut pair uses
# query-atom positions inside the SMARTS pattern. Rules such as urea, amines,
# and ethers omit central linker atoms in product-side RECAP; here we cut the
# adjacent original bonds and keep those linker atoms as their own components
# when necessary.
RECAP_CLEAVAGE_RULES = (
    (
        "urea",
        "[#7;+0;D2,D3]!@C(!@=O)!@[#7;+0;D2,D3]",
        ((0, 1), (1, 3)),
    ),
    (
        "amide",
        "[C;!$(C([#7])[#7])](=!@[O])!@[#7;+0;!D1]",
        ((0, 2),),
    ),
    (
        "ester",
        "[C](=!@[O])!@[O;+0]",
        ((0, 2),),
    ),
    (
        "amine",
        "[N;!D1;+0;!$(N-C=[#7,#8,#15,#16])](-!@[*])-!@[*]",
        ((0, 1), (0, 2)),
    ),
    (
        "cyclic_amine",
        "[#7;R;D3;+0]-!@[*]",
        ((0, 1),),
    ),
    (
        "ether",
        "[#6]-!@[O;+0]-!@[#6]",
        ((0, 1), (1, 2)),
    ),
    (
        "olefin",
        "[C]=!@[C]",
        ((0, 1),),
    ),
    (
        "aromatic_n_aliphatic_c",
        "[n;+0]-!@[C]",
        ((0, 1),),
    ),
    (
        "lactam_n_aliphatic_c",
        "[O]=[C]-@[N;+0]-!@[C]",
        ((2, 3),),
    ),
    (
        "aryl_aryl",
        "[c]-!@[c]",
        ((0, 1),),
    ),
    (
        "aromatic_n_aryl_c",
        "[n;+0]-!@[c]",
        ((0, 1),),
    ),
    (
        "sulphonamide",
        "[#7;+0;D2,D3]-!@[S](=[O])=[O]",
        ((0, 1),),
    ),
)
_COMPILED_RECAP_QUERIES = None


class RECAPFragmenter(Fragmenter):
    """Fragment molecules by removing bonds matched by RECAP cleavage rules."""

    def fragment(self, mol) -> FragmentationResult:
        fragment_to_atoms, broken_bonds = recap_fragment_atom_sets(mol)
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
            fragment_types=[RECAP_FRAGMENT_TYPE] * len(fragment_tensors),
            fragment_smiles=fragment_smiles_from_atom_sets(mol, fragment_tensors),
        )
        return validate_fragmentation(result, mol)


def recap_fragment_atom_sets(mol):
    """Return connected atom components after removing RECAP boundary bonds."""
    broken_bonds = find_recap_bonds(mol)
    broken_pairs = {
        tuple(sorted((src_atom, dst_atom)))
        for _, src_atom, dst_atom, _ in broken_bonds
    }

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


def find_recap_bonds(mol):
    """Find original molecule bonds matched by RECAP cleavage SMARTS."""
    broken_by_bond_id = {}
    for rule_name, query, cut_pairs in compiled_recap_queries():
        for match in mol.GetSubstructMatches(query, uniquify=True):
            for query_src, query_dst in cut_pairs:
                if query_src >= len(match) or query_dst >= len(match):
                    continue
                src_atom = int(match[query_src])
                dst_atom = int(match[query_dst])
                bond = mol.GetBondBetweenAtoms(src_atom, dst_atom)
                if bond is None or bond.IsInRing():
                    continue
                bond_id = int(bond.GetIdx())
                broken_by_bond_id.setdefault(
                    bond_id,
                    (bond_id, src_atom, dst_atom, rule_name),
                )

    return [
        broken_by_bond_id[bond_id]
        for bond_id in sorted(broken_by_bond_id)
    ]


def compiled_recap_queries():
    """Compile RECAP SMARTS once per process."""
    global _COMPILED_RECAP_QUERIES
    if _COMPILED_RECAP_QUERIES is None:
        queries = []
        for rule_name, smarts, cut_pairs in RECAP_CLEAVAGE_RULES:
            query = Chem.MolFromSmarts(smarts)
            if query is None:
                raise ValueError(f"Invalid RECAP SMARTS for rule {rule_name}: {smarts}")
            queries.append((rule_name, query, cut_pairs))
        _COMPILED_RECAP_QUERIES = tuple(queries)
    return _COMPILED_RECAP_QUERIES


def _build_edges_from_broken_bonds(mol, atom_to_fragment, broken_bonds):
    edge_pairs = []
    bond_ids = []
    bond_indices = []
    attachment_pairs = []

    bond_index_by_id = {
        bond.GetIdx(): table_index
        for table_index, bond in enumerate(sorted(mol.GetBonds(), key=lambda bond: bond.GetIdx()))
    }

    for bond_id, src_atom, dst_atom, rule_name in sorted(broken_bonds, key=lambda item: item[0]):
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
