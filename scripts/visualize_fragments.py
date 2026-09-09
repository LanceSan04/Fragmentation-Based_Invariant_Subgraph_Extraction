#!/usr/bin/env python
"""Visualize one molecule's fragmentation: molecule, per-fragment atoms, fragment graph.

Loads a single molecule directly from a processed split .pt file (created by
``src/data/preprocess.py``) and renders a three-panel figure:

  1. the plain molecule with RDKit atom indices,
  2. the same molecule with every atom/bond colored by the fragment it belongs
     to (matches the per-fragment coloring style used in analyze_subgraphs.py),
  3. the fragment graph (fragments as nodes, force-directed spring layout,
     edges from ``fragment_edge_index``), colored to match panel 2.

No trained model or checkpoint is needed here -- this only reads the
fragmentation already stored in the processed split.

CLI arguments:
    processed_dir
        Directory containing split .pt files, e.g.
        data/processed/bbbp/group_graph
    --split NAME
        Split file to read (without .pt). Default: test.
    --index N
        Molecule index within the split. Default: 0.
    --method NAME
        Fragmentation method to visualize for mixed-view splits (the
        fragment fields are stored as ``{method}_fragment_x_static`` etc.).
        Default: the first method found on the record. Legacy single-method
        splits store unprefixed fields and are reported as "default".
    --output_dir PATH
        Directory for the output PNG. Default: results/fragment_visualizations.

Example:
    python scripts/visualize_fragments.py data/processed/bbbp/group_graph --split test --index 0
"""

import argparse
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_molecule(processed_dir, split, index):
    """Load one raw (unbatched) molecule record straight from a processed split .pt."""
    split_path = Path(processed_dir) / f"{split}.pt"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    payload = torch.load(split_path, map_location="cpu", weights_only=False)
    molecules = payload["molecule_data"]
    if index < 0 or index >= len(molecules):
        raise IndexError(f"--index must be between 0 and {len(molecules) - 1} for {split_path}")
    return molecules[index]


def record_keys(example):
    keys = example.keys() if callable(getattr(example, "keys", None)) else example.keys
    return sorted(str(key) for key in keys)


def available_methods(example):
    """Fragmentation methods stored on this record (None = unprefixed default)."""
    keys = record_keys(example)
    methods = []
    if "fragment_x_static" in keys:
        methods.append(None)
    suffix = "_fragment_x_static"
    for key in keys:
        if key.endswith(suffix) and key != "fragment_x_static":
            methods.append(key[: -len(suffix)])
    return methods


def field(example, method, base_name):
    name = base_name if method is None else f"{method}_{base_name}"
    return getattr(example, name, None)


def _lighten(color, amount=0.35):
    """Blend an RGB color toward white by ``amount`` (0 = unchanged, 1 = white)."""
    return tuple(c + (1.0 - c) * amount for c in color)


FRAGMENT_PALETTE = [
    _lighten(matplotlib.colors.to_rgb("red")),
    _lighten(matplotlib.colors.to_rgb("orange")),
    _lighten(matplotlib.colors.to_rgb("blue")),
    _lighten(matplotlib.colors.to_rgb("gray")),
    _lighten(matplotlib.colors.to_rgb("purple")),
    _lighten(matplotlib.colors.to_rgb("cyan")),
    _lighten(matplotlib.colors.to_rgb("pink")),
    _lighten(matplotlib.colors.to_rgb("brown")),
    _lighten(matplotlib.colors.to_rgb("yellow")),
]


def fragment_signatures(mol, num_fragments, atom_to_fragment):
    """Canonical fragment SMILES per fragment id, used to identify identical fragments."""
    fragment_atoms = {i: [] for i in range(num_fragments)}
    for atom_idx, frag_id in enumerate(atom_to_fragment):
        if 0 <= frag_id < num_fragments:
            fragment_atoms[frag_id].append(atom_idx)
    signatures = {}
    for frag_id, atom_ids in fragment_atoms.items():
        if not atom_ids:
            signatures[frag_id] = f"fragment_{frag_id}"
            continue
        signatures[frag_id] = Chem.MolFragmentToSmiles(
            mol, atomsToUse=atom_ids, canonical=True, isomericSmiles=True
        )
    return signatures


def fragment_colors(signatures):
    """RGB color per fragment id, cycled through the palette by fragment signature.

    Fragments with the same canonical SMILES (e.g. two separate methyl groups)
    share a color instead of getting distinct colors just because they have
    different fragment ids.
    """
    color_by_signature = {}
    frag_colors = {}
    for frag_id, signature in signatures.items():
        if signature not in color_by_signature:
            color_by_signature[signature] = FRAGMENT_PALETTE[len(color_by_signature) % len(FRAGMENT_PALETTE)]
        frag_colors[frag_id] = color_by_signature[signature]
    return frag_colors


def highlight_colors(mol, atom_to_fragment, frag_colors):
    """Color every atom by its fragment; color every bond by its fragment(s)."""
    atom_colors = {
        atom_idx: frag_colors[frag_id]
        for atom_idx, frag_id in enumerate(atom_to_fragment)
        if frag_id in frag_colors
    }
    bond_colors = {}
    for bond in mol.GetBonds():
        c1 = atom_colors.get(bond.GetBeginAtomIdx())
        c2 = atom_colors.get(bond.GetEndAtomIdx())
        if c1 is None and c2 is None:
            continue
        c1 = c1 if c1 is not None else c2
        c2 = c2 if c2 is not None else c1
        bond_colors[bond.GetIdx()] = tuple((x + y) / 2 for x, y in zip(c1, c2))
    return atom_colors, bond_colors


def render_mol_highlight(mol, highlight_atoms, highlight_bonds, atom_colors, bond_colors, size=(900, 720)):
    """Render one molecule with the given atoms/bonds highlighted as a PIL image."""
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=[int(a) for a in highlight_atoms],
        highlightBonds=[int(b) for b in highlight_bonds],
        highlightAtomColors=atom_colors or None,
        highlightBondColors=bond_colors or None,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))


def fragment_layout(mol, num_fragments, atom_to_fragment):
    """Fragment node positions from the molecule's own 2D depiction.

    Each fragment is placed at the centroid of its member atoms' 2D coordinates,
    so a fragment drawn in the upper-right of the atom panel stays in the
    upper-right of the fragment graph instead of moving under a force-directed
    layout.
    """
    if num_fragments <= 0:
        return {}
    conf = mol.GetConformer()
    sums = {i: [0.0, 0.0, 0] for i in range(num_fragments)}
    for atom_idx, frag_id in enumerate(atom_to_fragment):
        if 0 <= frag_id < num_fragments:
            pos = conf.GetAtomPosition(atom_idx)
            entry = sums[frag_id]
            entry[0] += pos.x
            entry[1] += pos.y
            entry[2] += 1
    positions = {
        frag_id: (entry[0] / entry[2], entry[1] / entry[2])
        for frag_id, entry in sums.items()
        if entry[2] > 0
    }
    for frag_id in range(num_fragments):
        positions.setdefault(frag_id, (0.0, 0.0))
    return _normalize_positions(positions)


def _normalize_positions(positions, pad=0.15):
    """Rescale positions to fit within [-1, 1], preserving aspect ratio and orientation."""
    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min, 1e-6)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    scale = (1.0 - pad) / (span / 2)
    return {frag_id: ((x - cx) * scale, (y - cy) * scale) for frag_id, (x, y) in positions.items()}


def draw_fragment_graph(ax, mol, num_fragments, edges, atom_to_fragment, frag_colors):
    """Draw every fragment as a colored node and every fragment edge, laid out to match the atom graph."""
    pos = fragment_layout(mol, num_fragments, atom_to_fragment)
    for u, v in edges:
        if u not in pos or v not in pos or u == v:
            continue
        color = tuple((a + b) / 2 for a, b in zip(frag_colors[u], frag_colors[v]))
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=color, linewidth=2.5, zorder=1)

    for frag_id, (x, y) in pos.items():
        ax.scatter([x], [y], s=340, c=[frag_colors[frag_id]], edgecolors="black", linewidths=1.2, zorder=2)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def edge_pairs(edge_index):
    """Convert a [2, E] fragment_edge_index tensor into a list of (u, v) int pairs."""
    if edge_index is None or not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
        return []
    ei = edge_index.detach().cpu()
    return [(int(ei[0, k]), int(ei[1, k])) for k in range(ei.size(1))]


def build_figure(example, method):
    """Build the 3-panel (molecule, atoms-by-fragment, fragment graph) figure."""
    smiles = getattr(example, "smiles", None)
    if not smiles:
        raise ValueError("Record has no smiles field to render.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    AllChem.Compute2DCoords(mol)

    atom_to_fragment_tensor = field(example, method, "atom_to_fragment")
    atom_to_fragment = (
        [int(a) for a in atom_to_fragment_tensor.reshape(-1).tolist()]
        if atom_to_fragment_tensor is not None
        else []
    )
    frag_x = field(example, method, "fragment_x_static")
    num_fragments = (
        int(frag_x.size(0)) if frag_x is not None else (max(atom_to_fragment, default=-1) + 1)
    )
    edges = edge_pairs(field(example, method, "fragment_edge_index"))
    signatures = fragment_signatures(mol, num_fragments, atom_to_fragment)
    frag_colors = fragment_colors(signatures)

    all_bonds = list(range(mol.GetNumBonds()))
    atom_colors, bond_colors = highlight_colors(mol, atom_to_fragment, frag_colors)

    plain_image = render_mol_highlight(mol, [], [], {}, {})
    colored_image = render_mol_highlight(
        mol, range(mol.GetNumAtoms()), all_bonds, atom_colors, bond_colors
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    axes[0].imshow(np.asarray(plain_image))
    axes[0].axis("off")

    axes[1].imshow(np.asarray(colored_image))
    axes[1].axis("off")

    draw_fragment_graph(axes[2], mol, num_fragments, edges, atom_to_fragment, frag_colors)

    fig.tight_layout()
    _add_captions(
        fig,
        axes,
        [
            f"Atom graph",
            f"Atoms by fragment",
            f"Fragment graph",
        ],
    )
    return fig


def _add_captions(fig, axes, captions):
    """Draw one caption per panel, aligned to a shared baseline below all panels.

    The fragment graph panel has aspect="equal" data, so matplotlib shrinks its
    axes box to preserve that aspect while the two image panels keep their full
    box height; anchoring captions to each axes' own transAxes coordinates put
    the fragment-graph caption far below the others. Using the lowest of the
    three post-layout box bottoms as a shared baseline keeps every caption
    level regardless of per-panel aspect shrinking.
    """
    boxes = [ax.get_position() for ax in axes]
    baseline = min(box.y0 for box in boxes)
    for box, caption in zip(boxes, captions):
        x_center = (box.x0 + box.x1) / 2
        fig.text(
            x_center, baseline, caption, ha="center", va="top", fontsize=20, fontweight="bold"
        )


def safe_filename(text):
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text)).strip("_")
    return cleaned[:60] or "molecule"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("processed_dir", help="Directory containing split .pt files.")
    parser.add_argument("--split", default="test", help="Split file to read (without .pt). Default: test.")
    parser.add_argument("--index", type=int, default=0, help="Molecule index within the split. Default: 0.")
    parser.add_argument(
        "--method",
        default=None,
        help="Fragmentation method to visualize for mixed-view splits. Default: first found.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Output directory for the PNG.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    example = load_molecule(args.processed_dir, args.split, args.index)

    methods = available_methods(example)
    if not methods:
        raise ValueError("Record has no fragment_x_static field under any known method.")
    method = args.method
    if method is not None and method not in methods:
        available = ", ".join(m or "default" for m in methods)
        raise ValueError(f"Method {method!r} not found on this record. Available: {available}")
    if method is None:
        method = methods[0]

    fig = build_figure(example, method)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    molecule_id = getattr(example, "molecule_id", None)
    label = safe_filename(molecule_id if molecule_id is not None else args.index)
    method_label = safe_filename(method or "default")
    path = output_dir / f"molecule_{args.index}_{label}_{method_label}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
