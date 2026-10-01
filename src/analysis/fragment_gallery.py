"""Run molecular fragmentation on one molecule and save a visual summary.

CLI options:
    --smiles SMILES
        SMILES string to fragment. Default: CCOC(=O)C.
    --output-dir PATH
        Directory for the PNG and text summary. Default:
        results/fragment_visualizations.
    --fragmentation NAME
        Fragmentation method to visualize. Choices are sorted(FRAGMENTERS).
        Default: group_graph.

Example:
    python -m src.analysis.fragment_gallery --smiles "CCOC(=O)C"
    python -m src.analysis.fragment_gallery --smiles "CCOC(=O)C" --fragmentation brics
"""

import argparse
import re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw


from src.fragmentation import FRAGMENTERS, build_fragmenter

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


def main(argv=None):
    args = parse_args(argv)
    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {args.smiles!r}")

    fragmenter = build_fragmenter(args.fragmentation)
    result = fragmenter.fragment(mol)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = safe_filename(args.smiles)
    fragmentation_name = safe_filename(args.fragmentation)
    image_path = output_dir / f"{stem}_{fragmentation_name}_fragments.png"
    summary_path = output_dir / f"{stem}_{fragmentation_name}_fragments.txt"

    lines = build_summary_lines(mol, result, args.smiles, args.fragmentation)
    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    image = build_fragment_grid_image(mol, result)
    image.save(image_path)

    print(summary_text)
    print()
    print(f"Saved fragment image: {image_path}")
    print(f"Saved fragment summary: {summary_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smiles",
        default="CCOC(=O)C",
        help="SMILES string to fragment. Default: CCOC(=O)C",
    )
    parser.add_argument(
        "--output-dir",
        default="results/fragment_visualizations",
        help="Directory for the PNG and text summary.",
    )
    parser.add_argument(
        "--fragmentation",
        default="group_graph",
        choices=sorted(FRAGMENTERS),
        help="Fragmentation method to visualize. Default: group_graph",
    )
    return parser.parse_args(argv)


def build_summary_lines(mol, result, smiles, fragmentation):
    lines = [
        "Fragmentation Run",
        f"Fragmentation method: {fragmentation}",
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
            result.fragment_types[fragment_id] if result.fragment_types is not None else None
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
