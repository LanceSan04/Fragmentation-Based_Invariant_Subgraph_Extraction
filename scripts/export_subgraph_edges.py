#!/usr/bin/env python
"""Export the REU (FragmentCIGA) ensemble's aggregated extracted subgraph per molecule.

Stage 1 of the two-stage REU-vs-SuGAr subgraph visualization. This script runs the
trained ``SuGAREnsemble`` on a chosen split, and for each molecule writes the atoms and
bonds of the *aggregated* reconstructed subgraph (the invariant subgraph the ensemble's
classifiers actually vote on) to a small JSON file. A separate, RDKit-only renderer
(``render_subgraph_comparison.py`` at the repo root) reads this JSON together with the
SuGAr exporter's JSON and draws the side-by-side molecule highlights.

Bonds are emitted as ``[begin_atom, end_atom, weight]`` triples in this molecule's own
RDKit atom indexing (the atom order of ``Chem.MolFromSmiles(smiles)``, which the
preprocessing preserves), plus the raw SMILES string. No canonicalization happens here —
the renderer owns all cross-model alignment so it is robust to RDKit-version differences
between the two repos' environments.

Reuses ``load_ensemble`` (checkpoint + dataloaders) and ``SuGAREnsemble.extract_subgraphs``
from ``scripts/analyze_subgraphs.py``.

Example:
    python scripts/export_subgraph_edges.py \
        --config configs/experiment.yaml \
        --checkpoint outputs/.../checkpoints/best_model.pt \
        --processed-dir data/processed/drugood_lbap_core_ec50_assay/group_graph \
        --split ood_test --num-molecules 200 --out reu_edges.json --device cpu
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
for p in (str(ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ``analyze_subgraphs`` lives next to this file; importing it only defines functions.
from analyze_subgraphs import load_ensemble  # noqa: E402
from train_sugar import build_config, resolve_device  # noqa: E402


def _scalar(value):
    """Best-effort convert a possibly-tensor scalar field to a plain Python value."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.reshape(-1)[0].item()
        return value.reshape(-1).tolist()
    return value


def _molecule_record(full, agg):
    """Build one JSON record from an original molecule and its aggregated reconstruction."""
    # ``selected_original_atom_indices`` maps each local reconstructed node back to the
    # original (RDKit) atom index; ``bond_index`` holds one column per undirected bond.
    orig_atom = agg.selected_original_atom_indices.reshape(-1).tolist()
    bond_index = agg.bond_index
    weights = getattr(agg, "reconstructed_bond_weight", None)

    selected_bonds = []
    for c in range(bond_index.size(1)):
        a = int(orig_atom[int(bond_index[0, c])])
        b = int(orig_atom[int(bond_index[1, c])])
        w = float(weights[c]) if weights is not None else 1.0
        selected_bonds.append([a, b, w])

    return {
        "smiles": _scalar(getattr(full, "smiles", None)),
        "molecule_id": _scalar(getattr(full, "molecule_id", None)),
        "label": _scalar(getattr(full, "y", None)),
        "selected_atoms": [int(a) for a in orig_atom],
        "selected_bonds": selected_bonds,
    }


@torch.no_grad()
def export_edges(ensemble, loader, device, num_molecules=None):
    """Iterate the split and collect one record per molecule (up to ``num_molecules``)."""
    molecules = []
    for batch in loader:
        batch = batch.to(device)
        full_list = batch.to_data_list()
        result = ensemble.extract_subgraphs(batch)
        agg_list = result["aggregated"].to_data_list()

        for full, agg in zip(full_list, agg_list):
            if num_molecules is not None and len(molecules) >= num_molecules:
                return molecules
            if getattr(full, "smiles", None) is None:
                continue
            molecules.append(_molecule_record(full, agg))
    return molecules


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config (model/data defaults).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained REU checkpoint .pt.")
    parser.add_argument(
        "--processed-dir", dest="processed_dir", type=str, default=None,
        help="Processed split directory (defaults to processed_dir from --config).",
    )
    parser.add_argument("--split", type=str, default="ood_test", help="Split to export.")
    parser.add_argument(
        "--num-molecules", dest="num_molecules", type=int, default=200,
        help="Max molecules to export (<=0 for all).",
    )
    parser.add_argument("--out", type=str, default="reu_edges.json", help="Output JSON path.")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda (default: config).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build the effective config the same way the training/analysis entrypoints do.
    config_args = SimpleNamespace(config=args.config)
    if args.processed_dir is not None:
        config_args.processed_dir = args.processed_dir
    if args.device is not None:
        config_args.device = args.device
    config = build_config(config_args)

    device = resolve_device(config)
    print(f"Using device: {device}")

    ensemble, dataloaders = load_ensemble(config, args.checkpoint, device)
    loader = dataloaders.get(args.split)
    if loader is None:
        available = ", ".join(sorted(k for k, v in dataloaders.items() if v is not None)) or "none"
        raise ValueError(f"Split {args.split!r} not found. Available: {available}")

    cap = None if args.num_molecules is not None and args.num_molecules <= 0 else args.num_molecules
    print(f"Exporting aggregated subgraphs from split {args.split!r} (cap={cap})...")
    molecules = export_edges(ensemble, loader, device, num_molecules=cap)

    payload = {
        "model": "REU",
        "dataset": config.get("dataset"),
        "split": args.split,
        "aggregation": config.get("aggregation_mode", config.get("aggregation", "mean")),
        "checkpoint": str(args.checkpoint),
        "molecules": molecules,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f)

    non_empty = sum(1 for m in molecules if m["selected_bonds"])
    print(f"Wrote {len(molecules)} molecules ({non_empty} with non-empty subgraphs) to {out_path}")


if __name__ == "__main__":
    main()
