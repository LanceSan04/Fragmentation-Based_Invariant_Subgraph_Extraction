"""Group Graph construction adapted from the upstream implementation."""

import networkx as nx
from rdkit import Chem

from src.data.atom_graph import atom_features
from src.fragmentation.base import FragmentationResult, Fragmenter
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
        raise NotImplementedError


def build_fragmentation_result(mol, cluster, atom_cls, inter_pair):
    """Convert upstream Group Graph outputs into FragmentationResult."""
    raise NotImplementedError


def map_fragment_edges_to_bonds(mol, atom_cls, inter_pair):
    """Map every unique fragment edge to its original RDKit bond ID."""
    raise NotImplementedError


def order_fragments_deterministically(cluster):
    """Return a stable fragment order independent of set iteration order."""
    raise NotImplementedError


def graph_to_fragmentation_result(graph) -> FragmentationResult:
    """Convert an upstream NetworkX group graph into project mappings."""
    raise NotImplementedError


def build_fragment_edge_features(
    mol,
    fragmentation_result: FragmentationResult,
):
    """Build attachment-atom, bond, and optional fragment-size edge features."""
    raise NotImplementedError


def build_unique_fragment_edges(mol, atom_to_fragment):
    """Build one undirected fragment edge per original inter-fragment bond."""
    raise NotImplementedError


def handle_single_fragment_molecule(mol) -> FragmentationResult:
    """Represent a molecule with one fragment without introducing a fake edge."""
    raise NotImplementedError
