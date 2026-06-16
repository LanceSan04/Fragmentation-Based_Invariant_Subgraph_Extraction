"""Run Group Graph fragmentation on one molecule and save a visual summary.

Example:
    python scripts/run_group_graph_visualization.py --smiles "CCOC(=O)C"
"""

import argparse
import re
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fragmentation.group_graph import GroupGraphFragmenter


FRAGMENT_COLORS = [
    (0.90, 0.20, 0.20),
    (0.20, 0.55, 0.90),
    (0.20, 0.75, 0.35),
    (0.95, 0.65, 0.15),
    (0.65, 0.35, 0.90),
    (0.20, 0.75, 0.75),
    (0.95, 0.35, 0.65),
    (0.55, 0.55, 0.55),
]


def main():
    args = parse_args()
    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {args.smiles!r}")

    result = GroupGraphFragmenter().fragment(mol)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = safe_filename(args.smiles)
    image_path = output_dir / f"{stem}_group_graph_fragments.png"
    summary_path = output_dir / f"{stem}_group_graph_fragments.txt"

    lines = build_summary_lines(mol, result, args.smiles)
    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    image = build_fragment_grid_image(mol, result)
    image.save(image_path)

    print(summary_text)
    print()
    print(f"Saved fragment image: {image_path}")
    print(f"Saved fragment summary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Group Graph fragments for one SMILES string."
    )
    parser.add_argument(
        "--smiles",
        default="CCOC(=O)C",
        help="SMILES string to fragment. Default: CCOC(=O)C",
    )
    parser.add_argument(
        "--output-dir",
        default="scripts/group_graph_visuals/fragment_visualizations",
        help="Directory for the PNG and text summary.",
    )
    return parser.parse_args()


def build_summary_lines(mol, result, smiles):
    lines = [
        "Group Graph Fragmentation Run",
        f"SMILES: {smiles}",
        f"Canonical SMILES: {Chem.MolToSmiles(mol, isomericSmiles=True)}",
        f"Atoms: {mol.GetNumAtoms()}",
        f"Bonds: {mol.GetNumBonds()}",
        f"Fragments: {result.num_fragments}",
        f"Fragment edges: {result.num_fragment_edges}",
        "",
        "Fragments:",
    ]

    for fragment_id, atoms in enumerate(result.fragment_to_atoms):
        atom_ids = atoms.tolist()
        fragment_type = (
            result.fragment_types[fragment_id]
            if result.fragment_types is not None
            else None
        )
        atom_symbols = [mol.GetAtomWithIdx(atom_id).GetSymbol() for atom_id in atom_ids]
        fragment_smiles = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=atom_ids,
            canonical=True,
            isomericSmiles=True,
        )
        internal_bonds = [
            bond.GetIdx()
            for bond in mol.GetBonds()
            if bond.GetBeginAtomIdx() in atom_ids and bond.GetEndAtomIdx() in atom_ids
        ]

        lines.append(
            f"  F{fragment_id}: "
            f"atoms={atom_ids}, "
            f"symbols={atom_symbols}, "
            f"type={fragment_type}, "
            f"fragment_smiles={fragment_smiles}, "
            f"internal_bonds={internal_bonds}"
        )

    lines.extend(["", "Fragment edges:"])
    if result.num_fragment_edges == 0:
        lines.append("  none")
    else:
        for edge_id in range(result.num_fragment_edges):
            src_fragment = int(result.fragment_edge_index[0, edge_id])
            dst_fragment = int(result.fragment_edge_index[1, edge_id])
            src_atom = int(result.fragment_edge_to_attachment_atoms[0, edge_id])
            dst_atom = int(result.fragment_edge_to_attachment_atoms[1, edge_id])
            bond_id = int(result.fragment_edge_to_bond[edge_id])
            bond = mol.GetBondWithIdx(bond_id)
            lines.append(
                f"  E{edge_id}: "
                f"F{src_fragment}-F{dst_fragment}, "
                f"bond={bond_id}, "
                f"attachment_atoms={[src_atom, dst_atom]}, "
                f"attachment_symbols="
                f"{[mol.GetAtomWithIdx(src_atom).GetSymbol(), mol.GetAtomWithIdx(dst_atom).GetSymbol()]}, "
                f"bond_type={bond.GetBondType()}"
            )

    return lines


def build_fragment_grid_image(mol, result):
    mols = []
    legends = []
    highlight_atom_lists = []
    highlight_bond_lists = []

    for fragment_id, atoms in enumerate(result.fragment_to_atoms):
        atom_ids = atoms.tolist()
        mols.append(mol_with_atom_indices(mol))
        legends.append(f"F{fragment_id}: atoms {atom_ids}")
        highlight_atom_lists.append(atom_ids)
        highlight_bond_lists.append(internal_bond_ids(mol, atom_ids))

    return Draw.MolsToGridImage(
        mols,
        molsPerRow=2,
        subImgSize=(450, 350),
        legends=legends,
        highlightAtomLists=highlight_atom_lists,
        highlightBondLists=highlight_bond_lists,
        useSVG=False,
    )


def mol_with_atom_indices(mol):
    mol_copy = Chem.Mol(mol)
    for atom in mol_copy.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    return mol_copy


def internal_bond_ids(mol, atom_ids):
    atom_set = set(atom_ids)
    return [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set
    ]


def safe_filename(smiles):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", smiles).strip("_")
    return cleaned or "molecule"


if __name__ == "__main__":
    main()
