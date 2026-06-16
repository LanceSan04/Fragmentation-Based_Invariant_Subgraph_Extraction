"""Atom and bond feature utilities adapted from Group Graph.

The feature encoders below are copied from group-graph/prepare_group_graph.py.
The atom-graph construction code is project-specific: it preserves a canonical
undirected bond table for provenance and creates directed PyG edges only for
message passing.
"""

from dataclasses import dataclass

import torch
from rdkit import Chem


MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


@dataclass
class AtomGraphData:
    """Tensor-backed atom graph with original atom and bond provenance."""

    atom_x: torch.Tensor
    bond_index: torch.Tensor
    bond_attr: torch.Tensor
    atom_edge_index: torch.Tensor
    atom_edge_attr: torch.Tensor
    atom_ids: torch.Tensor
    bond_ids: torch.Tensor
    directed_edge_to_bond: torch.Tensor

    @property
    def num_atoms(self):
        return int(self.atom_x.size(0))

    @property
    def num_bonds(self):
        return int(self.bond_attr.size(0))

    @property
    def num_directed_edges(self):
        return int(self.atom_edge_attr.size(0))

    def validate(self):
        """Validate feature dimensions and provenance tensor alignment."""
        if self.atom_x.dim() != 2 or self.atom_x.size(1) != ATOM_FDIM:
            raise ValueError(
                f"atom_x must have shape [num_atoms, {ATOM_FDIM}], "
                f"got {tuple(self.atom_x.shape)}"
            )

        if self.bond_index.shape != (2, self.num_bonds):
            raise ValueError(
                "bond_index must have shape [2, num_bonds], "
                f"got {tuple(self.bond_index.shape)}"
            )

        if self.bond_attr.dim() != 2 or self.bond_attr.size(1) != BOND_FDIM:
            raise ValueError(
                f"bond_attr must have shape [num_bonds, {BOND_FDIM}], "
                f"got {tuple(self.bond_attr.shape)}"
            )

        if self.atom_edge_index.shape != (2, self.num_directed_edges):
            raise ValueError(
                "atom_edge_index must have shape [2, num_directed_edges], "
                f"got {tuple(self.atom_edge_index.shape)}"
            )

        if self.atom_edge_attr.dim() != 2 or self.atom_edge_attr.size(1) != BOND_FDIM:
            raise ValueError(
                f"atom_edge_attr must have shape [num_directed_edges, {BOND_FDIM}], "
                f"got {tuple(self.atom_edge_attr.shape)}"
            )

        if self.atom_ids.numel() != self.num_atoms:
            raise ValueError("atom_ids must contain one ID per atom")

        if self.bond_ids.numel() != self.num_bonds:
            raise ValueError("bond_ids must contain one ID per original bond")

        if self.directed_edge_to_bond.numel() != self.num_directed_edges:
            raise ValueError("directed_edge_to_bond must contain one entry per directed edge")

        if self.num_bonds == 0:
            if self.num_directed_edges != 0:
                raise ValueError("bondless molecules cannot have directed bond edges")
            return self

        if self.bond_index.min().item() < 0 or self.bond_index.max().item() >= self.num_atoms:
            raise ValueError("bond_index contains an atom index outside the molecule")

        if (
            self.atom_edge_index.min().item() < 0
            or self.atom_edge_index.max().item() >= self.num_atoms
        ):
            raise ValueError("atom_edge_index contains an atom index outside the molecule")

        if (
            self.directed_edge_to_bond.min().item() < 0
            or self.directed_edge_to_bond.max().item() >= self.num_bonds
        ):
            raise ValueError("directed_edge_to_bond contains a bond index outside the bond table")

        return self


def onek_encoding_unk(value, choices):
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups=None):
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features


def bond_features(bond: Chem.rdchem.Bond):
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


def _empty_long_edge_index():
    return torch.empty((2, 0), dtype=torch.long)


def _empty_float_features(width):
    return torch.empty((0, width), dtype=torch.float32)


def _sanitize_mol_copy(mol):
    if mol is None:
        raise ValueError("Cannot build an atom graph from mol=None")

    mol_copy = Chem.Mol(mol)
    Chem.SanitizeMol(mol_copy)

    if mol_copy.GetNumAtoms() == 0:
        raise ValueError("Cannot build an atom graph for a molecule with no atoms")

    return mol_copy


def mol_to_atom_graph(mol):
    """Build the project's atom-level graph tensors and provenance mappings.

    The returned record stores every original RDKit bond once in ``bond_index``
    and creates two directed copies in ``atom_edge_index`` for PyG message
    passing. ``directed_edge_to_bond`` indexes the canonical bond table.
    """
    mol = _sanitize_mol_copy(mol)

    atom_x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float32,
    )
    atom_ids = torch.arange(mol.GetNumAtoms(), dtype=torch.long)

    bond_pairs = []
    bond_feature_rows = []
    bond_ids = []
    directed_edges = []
    directed_edge_features = []
    directed_edge_to_bond = []

    bonds = sorted(mol.GetBonds(), key=lambda bond: bond.GetIdx())
    for bond_table_index, bond in enumerate(bonds):
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        features = bond_features(bond)

        bond_pairs.append([begin_idx, end_idx])
        bond_feature_rows.append(features)
        bond_ids.append(bond.GetIdx())

        directed_edges.append([begin_idx, end_idx])
        directed_edge_features.append(features)
        directed_edge_to_bond.append(bond_table_index)

        directed_edges.append([end_idx, begin_idx])
        directed_edge_features.append(features)
        directed_edge_to_bond.append(bond_table_index)

    if bond_pairs:
        bond_index = torch.tensor(bond_pairs, dtype=torch.long).t().contiguous()
        bond_attr = torch.tensor(bond_feature_rows, dtype=torch.float32)
        bond_ids = torch.tensor(bond_ids, dtype=torch.long)
        atom_edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        atom_edge_attr = torch.tensor(directed_edge_features, dtype=torch.float32)
        directed_edge_to_bond = torch.tensor(directed_edge_to_bond, dtype=torch.long)
    else:
        bond_index = _empty_long_edge_index()
        bond_attr = _empty_float_features(BOND_FDIM)
        bond_ids = torch.empty((0,), dtype=torch.long)
        atom_edge_index = _empty_long_edge_index()
        atom_edge_attr = _empty_float_features(BOND_FDIM)
        directed_edge_to_bond = torch.empty((0,), dtype=torch.long)

    return AtomGraphData(
        atom_x=atom_x,
        bond_index=bond_index,
        bond_attr=bond_attr,
        atom_edge_index=atom_edge_index,
        atom_edge_attr=atom_edge_attr,
        atom_ids=atom_ids,
        bond_ids=bond_ids,
        directed_edge_to_bond=directed_edge_to_bond,
    ).validate()


def smiles_to_atom_graph(smiles):
    """Parse a SMILES string and return the project's atom-level graph record."""
    if smiles is None:
        raise ValueError("smiles cannot be None")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    return mol_to_atom_graph(mol)
