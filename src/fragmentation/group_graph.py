"""Group Graph construction adapted from the upstream implementation."""

import networkx as nx
import torch
from rdkit import Chem

from src.data.atom_graph import ATOM_FDIM, BOND_FDIM, atom_features, bond_features
from src.fragmentation.base import FragmentationResult, Fragmenter, validate_fragmentation
from src.fragmentation.molecule_fragment import (
    find_clusters,
    get_inter,
    get_mol,
    get_smiles,
    get_sub_mol,
)


class MolSplit(object):
    # 元素及其最大成键电子数
    def __init__(self, smiles):

        self.mol = get_mol(smiles)
        self.smiles = get_smiles(self.mol)

        # print(self.smiles)
        # self.mol_graph = self.build_mol_graph()
        self.cluster, self.atom_cls = find_clusters(self.mol)
        # print(self.cluster, self.atom_cls)
        self.inter_pair = get_inter(self.mol)
        self.max_inter_num = 4

        # self.vocab_mol_list = Vocab.get_vocab_mol()

    def get_mol_graph(self):
        g1 = nx.MultiGraph()
        g1.graph['mol'] = self.mol
        g1.graph['smiles'] = self.smiles
        mol = self.mol

        cluster, atom_cls = self.cluster, self.atom_cls
        g1.graph['atom_cls'] = atom_cls
        cls = list(cluster.keys())

        inter_pair = self.inter_pair
        Chem.Kekulize(self.mol)
        vocab_dict = {}
        if len(inter_pair) == 0 or len(cls) == 1:
            # print(self.smiles,'no inter',cls, get_smiles(self.mol))

            # print( [ i for i in range(len(self.atom_cls))])

            g1.add_node(0, frag_idx=[i + 1 for i in range(len(self.atom_cls))], sub_smiles=self.smiles,
                        sub_mol=self.mol)
            if self.smiles not in vocab_dict:
                vocab_dict[self.smiles] = 1
            else:
                vocab_dict[self.smiles] = vocab_dict[self.smiles] + 1
        else:
            for i, c in enumerate(cls):
                new_c = [idx + 1 for idx in c]
                sub_mol = get_sub_mol(mol, c)
                sub_smiles = get_smiles(sub_mol, isomeric=False)

                if get_mol(sub_smiles) != None:
                    g1.add_node(i, frag_idx=new_c, sub_smiles=sub_smiles, sub_mol=sub_mol)
                    if sub_smiles not in vocab_dict:
                        vocab_dict[sub_smiles] = 1
                    else:
                        vocab_dict[sub_smiles] = vocab_dict[sub_smiles] + 1
                else:
                    print(self.smiles, sub_smiles, c)
                    print(cls)

            for x, y in inter_pair:
                cls_x, cls_y = atom_cls[x], atom_cls[y]
                sub_mol_x, sub_mol_y = g1.nodes[cls_x]['sub_mol'], g1.nodes[cls_y]['sub_mol']
                # sub_smiles_x,sub_smiles_y = frags_label[atom_cls[x]]['sub_smiles'], frags_label[atom_cls[y]]['sub_smiles']
                inter_x = []
                inter_tensor_x = []
                for a in sub_mol_x.GetAtoms():
                    if int(a.GetProp('mol_idx')) == x:
                        inter_tensor_x.append(atom_features(a))
                        inter_x.append(a.GetIdx())
                if len(inter_x) == 0:
                    print(g1.nodes[cls_x]['sub_smiles'], 'do not match vocab_mol_list')
                    inter_x[0] = 0

                inter_y = []
                inter_tensor_y = []
                for a in sub_mol_y.GetAtoms():
                    if int(a.GetProp('mol_idx')) == y:
                        inter_tensor_y.append(atom_features(a))
                        inter_y.append(a.GetIdx())
                if len(inter_y) == 0:
                    print(g1.nodes[cls_x]['sub_smiles'], 'do not match vocab_mol_list')
                    inter_y[0] = 0

                if cls_x <= cls_y:
                    edge_attr = [sub_mol_x.GetNumAtoms(), inter_x[0],
                                 sub_mol_y.GetNumAtoms(), inter_y[0]]

                    inter_tensor = [inter_tensor_x[0], inter_tensor_y[0]]
                    g1.add_edge(cls_x, cls_y, edge_attr=edge_attr, inter_tensor=inter_tensor)

                else:
                    edge_attr = [sub_mol_y.GetNumAtoms(), inter_y[0],
                                 sub_mol_x.GetNumAtoms(), inter_x[0]]
                    inter_tensor = [inter_tensor_y[0], inter_tensor_x[0]]

                    g1.add_edge(cls_y, cls_x, edge_attr=edge_attr, inter_tensor=inter_tensor)

        g1.graph['vocab_dict'] = vocab_dict
        return g1


class GroupGraphFragmenter(Fragmenter):
    """Project-facing Group Graph fragmenter."""

    def fragment(self, mol) -> FragmentationResult:
        """Return 0-based fragments and lossless atom/bond provenance."""
        if mol is None:
            raise ValueError("Cannot fragment mol=None")

        canonical_mol = Chem.Mol(mol)
        Chem.SanitizeMol(canonical_mol)

        working_mol = Chem.Mol(canonical_mol)
        cluster, atom_cls = find_clusters(working_mol)
        result = build_fragmentation_result(canonical_mol, cluster, atom_cls, inter_pair=None)
        return validate_fragmentation(result, canonical_mol)


def build_fragmentation_result(mol, cluster, atom_cls, inter_pair):
    """Convert upstream Group Graph outputs into FragmentationResult."""
    ordered_fragments, ordered_types, old_to_new = order_fragments_deterministically(cluster)
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
    )


def map_fragment_edges_to_bonds(mol, atom_cls, inter_pair):
    """Map every unique fragment edge to its original RDKit bond ID."""
    mappings = []
    for src_atom, dst_atom in inter_pair:
        bond = mol.GetBondBetweenAtoms(int(src_atom), int(dst_atom))
        if bond is None:
            raise ValueError(f"Atoms {src_atom} and {dst_atom} are not bonded")
        mappings.append(
            {
                "fragment_edge": tuple(sorted((atom_cls[src_atom], atom_cls[dst_atom]))),
                "attachment_atoms": (int(src_atom), int(dst_atom)),
                "bond_id": int(bond.GetIdx()),
            }
        )
    return mappings


def order_fragments_deterministically(cluster):
    """Return a stable fragment order independent of set iteration order."""
    ordered_items = sorted(
        ((tuple(sorted(atoms)), fragment_type) for atoms, fragment_type in cluster.items()),
        key=lambda item: (min(item[0]), len(item[0]), item[0], item[1]),
    )
    ordered_fragments = [atoms for atoms, _ in ordered_items]
    ordered_types = [fragment_type for _, fragment_type in ordered_items]

    old_to_new = {}
    for new_id, atoms in enumerate(ordered_fragments):
        old_to_new[atoms] = new_id

    return ordered_fragments, ordered_types, old_to_new


def graph_to_fragmentation_result(graph) -> FragmentationResult:
    """Convert an upstream NetworkX group graph into project mappings."""
    fragment_to_atoms = []
    fragment_types = []
    for node_id, attrs in sorted(graph.nodes(data=True), key=lambda item: item[0]):
        atoms = [int(atom_id) - 1 for atom_id in attrs.get("frag_idx", [])]
        fragment_to_atoms.append(torch.tensor(atoms, dtype=torch.long))
        fragment_types.append(attrs.get("sub_smiles"))

    num_atoms = sum(len(atoms) for atoms in fragment_to_atoms)
    atom_to_fragment = torch.empty(num_atoms, dtype=torch.long)
    for fragment_id, atoms in enumerate(fragment_to_atoms):
        atom_to_fragment[atoms] = fragment_id

    edge_pairs = []
    attachments = []
    for src, dst, attrs in graph.edges(data=True):
        edge_pairs.append([int(src), int(dst)])
        edge_attr = attrs.get("edge_attr", [0, 0, 0, 0])
        src_local = int(edge_attr[1])
        dst_local = int(edge_attr[3])
        src_atom = int(fragment_to_atoms[int(src)][src_local])
        dst_atom = int(fragment_to_atoms[int(dst)][dst_local])
        attachments.append([src_atom, dst_atom])

    if edge_pairs:
        fragment_edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        fragment_edge_to_attachment_atoms = torch.tensor(attachments, dtype=torch.long).t().contiguous()
    else:
        fragment_edge_index = torch.empty((2, 0), dtype=torch.long)
        fragment_edge_to_attachment_atoms = torch.empty((2, 0), dtype=torch.long)

    missing = torch.full((fragment_edge_index.size(1),), -1, dtype=torch.long)
    return FragmentationResult(
        fragment_to_atoms=fragment_to_atoms,
        atom_to_fragment=atom_to_fragment,
        fragment_edge_index=fragment_edge_index,
        fragment_edge_to_bond=missing,
        fragment_edge_to_bond_index=missing.clone(),
        fragment_edge_to_attachment_atoms=fragment_edge_to_attachment_atoms,
        fragment_types=fragment_types,
    )


def build_fragment_edge_features(
    mol,
    fragmentation_result: FragmentationResult,
):
    """Build attachment-atom, bond, and optional fragment-size edge features."""
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


def handle_single_fragment_molecule(mol) -> FragmentationResult:
    """Represent a molecule with one fragment without introducing a fake edge."""
    atom_to_fragment = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
    return FragmentationResult(
        fragment_to_atoms=[torch.arange(mol.GetNumAtoms(), dtype=torch.long)],
        atom_to_fragment=atom_to_fragment,
        fragment_edge_index=torch.empty((2, 0), dtype=torch.long),
        fragment_edge_to_bond=torch.empty((0,), dtype=torch.long),
        fragment_edge_to_bond_index=torch.empty((0,), dtype=torch.long),
        fragment_edge_to_attachment_atoms=torch.empty((2, 0), dtype=torch.long),
        fragment_types=["single_fragment"],
    )
