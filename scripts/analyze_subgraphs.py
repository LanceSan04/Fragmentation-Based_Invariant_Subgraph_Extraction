#!/usr/bin/env python
"""Analyze invariant subgraphs extracted by a trained SuGAR ensemble.

For a chosen split this script loads a trained checkpoint, extracts — for every
molecule — each view's own invariant subgraph plus the shared aggregated subgraph
that the ensemble votes on, and reports:

  * pairwise inter-view agreement (Jaccard IoU over selected atoms),
  * a view-agreement heatmap,
  * per-molecule figures pairing, for each view and the aggregate, the highlighted
    molecular subgraph with the fragment graph (fragments as nodes, laid out with a
    force-directed spring layout) with the extracted fragment edges highlighted;
    each fragment gets its own color, shared between the two panels,
  * raw fragment and fragment-pair selection metrics,
  * support-filtered interpretation tables for top fragments, top fragment-pairs,
    and ranked functional-group rows.

A checkpoint must be produced first via ``train_sugar.py --save_model ...``; both the
rich ``save_training_checkpoint`` format and the bare ``{view_index: state_dict}`` map
are accepted. The checkpoint's view count / architecture must match the data.

The checkpoint and data directory can be given on the command line or, if omitted,
are read from ``--config``: ``--model_path`` falls back to the config's ``model_path``
or ``save_model``, and ``--processed_dir`` falls back to the config's ``processed_dir``.
CLI flags override the config when both are present.

CLI options:
    --config PATH
        Optional YAML config. Used for model/data defaults and normal training
        config values.
    --model_path PATH
        Trained checkpoint path. Defaults to model_path or save_model from config.
    --processed_dir PATH
        Directory containing processed split .pt files. Defaults to processed_dir
        from config.
    --split NAME
        Split to analyze. Defaults to the config test_split or "test".
    --num_samples N
        Optional cap on molecules analyzed. Default: all molecules in the split.
    --num_visualize N
        Number of molecule visualization PNGs to write. Default: 3.
    --output_dir PATH
        Output directory for CSVs and plots. Default: results/subgraph_analysis.
    --metric_min_available N
        Minimum available_count required for interpretation tables. Default: 10.
    --metric_top_k N
        Rows to keep in each ranked interpretation table. Use <=0 to keep all.
        Default: 25.
    --batch_size N
        Override analysis DataLoader batch size.
    --device {cpu,cuda}
        Override device. Defaults to config/device resolution.

Example with explicit paths:
    python scripts/analyze_subgraphs.py \
        --model_path checkpoints/smoke_sugar.pt \
        --processed_dir data/processed/bbbp/group_graph \
        --split test --num_samples 20 --num_visualize 3 --device cpu

Example reusing a training config for both paths:
    python scripts/analyze_subgraphs.py \
        --config configs/experiment.yaml \
        --split test --num_samples 20 --num_visualize 3 --device cpu
"""

import argparse
import csv
import io
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataloader import build_split_dataloaders
from src.models.sugar_ensemble import SuGAREnsemble
from src.training.checkpoint import (
    checkpoint_feature_dims,
    load_checkpoint,
    load_view_state_dicts,
)
from src.evaluation.rationale_metrics import (
    pairwise_view_agreement,
    summarize,
)
from train_sugar import (
    align_num_views_to_checkpoint,
    apply_checkpoint_defaults,
    build_config,
    build_views_from_feature_dims,
    infer_feature_dimensions,
    resolve_device,
)


# Fields that are unused by the model/reconstruction and are either bulky or ragged,
# so PyG should not try to collate them. Unlike training we KEEP ``smiles`` (needed for
# RDKit rendering) and ``molecule_id`` (needed for labeling).
UNUSED_BATCH_FIELDS = ["fragment_to_atoms", "metadata"]


def strip_unused_batch_fields(dataloaders):
    """Drop bulky/ragged metadata that is not needed for subgraph analysis."""
    for loader in dataloaders.values():
        if loader is None or not hasattr(loader, "dataset"):
            continue
        for data in loader.dataset:
            for key in UNUSED_BATCH_FIELDS:
                if hasattr(data, key):
                    delattr(data, key)
                if key in data._store:
                    del data._store[key]


def load_ensemble(config, checkpoint_path, device):
    """Load a trained SuGAR ensemble and the configured split loaders."""
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    apply_checkpoint_defaults(config, payload)
    align_num_views_to_checkpoint(config, payload)

    print(f"Loading data from {config.processed_dir}...")
    dataloaders = build_split_dataloaders(config.processed_dir, config)
    strip_unused_batch_fields(dataloaders)

    feature_dims = checkpoint_feature_dims(payload)
    if not feature_dims:
        first_loader = next(iter(dataloaders.values()))
        feature_dims = infer_feature_dimensions(first_loader)

    views = build_views_from_feature_dims(config, feature_dims)
    load_view_state_dicts(views, payload)
    for view in views:
        view.to(device)

    ensemble = SuGAREnsemble(
        views,
        aggregation_mode=config.get("aggregation_mode", config.get("aggregation", "mean")),
    ).eval()
    return ensemble, dataloaders


@torch.no_grad()
def collect_subgraph_records(ensemble, loader, device, num_samples=None):
    """Extract per-view + aggregated subgraphs and build per-molecule records.

    Returns a list of dicts, one per molecule, each holding the metric row plus the
    raw atom-index selections (used later for visualization and averaged agreement).
    """
    records = []
    for batch in loader:
        batch = batch.to(device)
        full_list = batch.to_data_list()
        result = ensemble.extract_subgraphs(batch)

        # per_view[v] and aggregated are batched; unbatch to align with full_list order.
        per_view_lists = [pv.to_data_list() for pv in result["per_view"]]
        agg_list = result["aggregated"].to_data_list()
        num_views = len(per_view_lists)

        for i in range(len(full_list)):
            if num_samples is not None and len(records) >= num_samples:
                return records

            full = full_list[i]
            agg_sub = agg_list[i]
            per_view_subs = [per_view_lists[v][i] for v in range(num_views)]
            per_view_atoms = [
                _atom_indices(sub.selected_original_atom_indices) for sub in per_view_subs
            ]
            agg_atoms = _atom_indices(agg_sub.selected_original_atom_indices)

            # Fragment graph (fragments as nodes, fragment edges as edges) plus the
            # fragment edges/nodes each subgraph extracted. Taken from the molecule's
            # own fragment structure so indices align with the reconstructions.
            fragment_edges = _edge_pairs(getattr(full, "fragment_edge_index", None))
            num_fragments = _num_fragments(full)
            per_view_edge_ids = [
                _id_list(getattr(sub, "selected_fragment_edge_ids", None))
                for sub in per_view_subs
            ]
            per_view_frag_ids = [
                _id_list(getattr(sub, "selected_fragment_ids", None)) for sub in per_view_subs
            ]
            agg_edge_ids = _id_list(getattr(agg_sub, "selected_fragment_edge_ids", None))
            agg_frag_ids = _id_list(getattr(agg_sub, "selected_fragment_ids", None))
            atom_to_fragment = _int_list(getattr(full, "atom_to_fragment", None))
            fragment_signatures = _fragment_signatures(full)
            label = _label_value(getattr(full, "y", None))

            agreement = pairwise_view_agreement(per_view_atoms)

            row = {
                "molecule_idx": len(records),
                "molecule_id": _scalar(getattr(full, "molecule_id", None)),
                "smiles": _scalar(getattr(full, "smiles", None)),
                "label": label,
                "mean_view_agreement": round(_offdiag_mean(agreement), 4),
            }

            records.append(
                {
                    "row": row,
                    "num_views": num_views,
                    "smiles": row["smiles"],
                    "molecule_id": row["molecule_id"],
                    "per_view_atoms": per_view_atoms,
                    "agg_atoms": agg_atoms,
                    "agreement": agreement,
                    "num_fragments": num_fragments,
                    "fragment_signatures": fragment_signatures,
                    "fragment_edges": fragment_edges,
                    "per_view_edge_ids": per_view_edge_ids,
                    "per_view_frag_ids": per_view_frag_ids,
                    "agg_edge_ids": agg_edge_ids,
                    "agg_frag_ids": agg_frag_ids,
                    "atom_to_fragment": atom_to_fragment,
                }
            )

        if num_samples is not None and len(records) >= num_samples:
            break
    return records


def _atom_indices(tensor):
    """Convert a selected_original_atom_indices tensor into a sorted list of ints."""
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return []
    return sorted(int(a) for a in tensor.reshape(-1).tolist())


def _id_list(tensor):
    """Convert a 1-D index tensor into a sorted list of ints (empty if absent)."""
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return []
    return sorted(int(a) for a in tensor.reshape(-1).tolist())


def _int_list(tensor):
    """Convert a 1-D tensor into a list of ints preserving order (empty if absent)."""
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return []
    return [int(a) for a in tensor.reshape(-1).tolist()]


def _edge_pairs(edge_index):
    """Convert a [2, E] fragment_edge_index tensor into a list of (u, v) int pairs."""
    if edge_index is None or not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
        return []
    ei = edge_index.detach().cpu()
    return [(int(ei[0, k]), int(ei[1, k])) for k in range(ei.size(1))]


def _label_value(value):
    """Normalize a graph label into a compact string for grouped metrics."""
    if value is None:
        return "unknown"
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return "unknown"
        value = value.reshape(-1)[0].item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _fragment_signatures(full_data):
    """Return stable fragment identity tuples aligned to fragment ids.

    The first implementation uses the metadata already preserved by preprocessing:
    ``(fragment_type, fragment_smiles)``. This is specific enough to distinguish
    Group Graph split fragments from ordinary fragments while still aggregating
    the same chemistry across molecules.
    """
    num_fragments = _num_fragments(full_data)
    fragment_types = list(getattr(full_data, "fragment_types", None) or [])
    fragment_smiles = list(getattr(full_data, "fragment_smiles", None) or [])
    signatures = []
    for fragment_id in range(num_fragments):
        fragment_type = (
            str(fragment_types[fragment_id])
            if fragment_id < len(fragment_types)
            else "unknown"
        )
        fragment_smiles_value = (
            str(fragment_smiles[fragment_id])
            if fragment_id < len(fragment_smiles)
            else f"fragment_{fragment_id}"
        )
        signatures.append((fragment_type, fragment_smiles_value))
    return signatures


def _num_fragments(full_data):
    """Number of fragments in a molecule (rows of fragment_x_static)."""
    frag_x = getattr(full_data, "fragment_x_static", None)
    if frag_x is not None:
        return int(frag_x.size(0))
    atom_to_fragment = getattr(full_data, "atom_to_fragment", None)
    if atom_to_fragment is not None and atom_to_fragment.numel() > 0:
        return int(atom_to_fragment.max()) + 1
    return 0


def _scalar(value):
    """Normalize a possibly-tensor/1-element scalar field into a plain Python value."""
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    return value


def _offdiag_mean(matrix):
    """Mean of the off-diagonal entries of a square agreement matrix (1.0 if <2 views)."""
    n = len(matrix)
    if n < 2:
        return 1.0
    total = sum(matrix[i][j] for i in range(n) for j in range(n) if i != j)
    return total / (n * (n - 1))


def _source_names(num_views):
    return [f"view_{view_idx}" for view_idx in range(num_views)] + [
        "any_view",
        "aggregated",
    ]


def _selected_ids_by_source(record, selected_key):
    """Return selected fragment/edge ids grouped by view, any-view, and aggregate."""
    per_view = record[selected_key]
    by_source = {
        f"view_{view_idx}": set(ids)
        for view_idx, ids in enumerate(per_view)
    }
    any_view = set()
    for ids in per_view:
        any_view.update(ids)
    by_source["any_view"] = any_view
    aggregate_key = "agg_frag_ids" if selected_key == "per_view_frag_ids" else "agg_edge_ids"
    by_source["aggregated"] = set(record[aggregate_key])
    return by_source


def _canonical_fragment_pair(sig_a, sig_b):
    """Return an order-independent pair of fragment signatures."""
    return tuple(sorted((sig_a, sig_b)))


def build_fragment_selection_metric_rows(records):
    """Build label-aware fragment and fragment-pair selection metric rows.

    Counts are occurrence-based: every fragment occurrence and every fragment-edge
    occurrence contributes to availability. Selection counts are grouped by view,
    any-view, and aggregated-subgraph source.
    """
    if not records:
        return [], []

    num_views = records[0]["num_views"]
    sources = _source_names(num_views)
    fragment_available = Counter()
    fragment_selected = Counter()
    edge_available = Counter()
    edge_selected = Counter()

    for record in records:
        label = str(record["row"].get("label", "unknown"))
        fragment_signatures = record["fragment_signatures"]

        for fragment_id, signature in enumerate(fragment_signatures):
            fragment_available[(label, signature)] += 1

        fragment_ids_by_source = _selected_ids_by_source(record, "per_view_frag_ids")
        for source, selected_ids in fragment_ids_by_source.items():
            for fragment_id in selected_ids:
                if 0 <= fragment_id < len(fragment_signatures):
                    fragment_selected[(source, label, fragment_signatures[fragment_id])] += 1

        edge_signatures = []
        for edge_id, (src, dst) in enumerate(record["fragment_edges"]):
            if src >= len(fragment_signatures) or dst >= len(fragment_signatures):
                continue
            edge_signature = _canonical_fragment_pair(
                fragment_signatures[src],
                fragment_signatures[dst],
            )
            edge_signatures.append((edge_id, edge_signature))
            edge_available[(label, edge_signature)] += 1

        edge_ids_by_source = _selected_ids_by_source(record, "per_view_edge_ids")
        edge_signature_by_id = dict(edge_signatures)
        for source, selected_ids in edge_ids_by_source.items():
            for edge_id in selected_ids:
                edge_signature = edge_signature_by_id.get(edge_id)
                if edge_signature is not None:
                    edge_selected[(source, label, edge_signature)] += 1

    fragment_rows = []
    for label, signature in sorted(fragment_available):
        available = fragment_available[(label, signature)]
        fragment_type, fragment_smiles = signature
        for source in sources:
            selected = fragment_selected[(source, label, signature)]
            fragment_rows.append(
                {
                    "label": label,
                    "selection_source": source,
                    "fragment_type": fragment_type,
                    "fragment_smiles": fragment_smiles,
                    "available_count": available,
                    "selected_count": selected,
                    "selection_rate": _rate(selected, available),
                }
            )

    edge_rows = []
    for label, edge_signature in sorted(edge_available):
        available = edge_available[(label, edge_signature)]
        (type_a, smiles_a), (type_b, smiles_b) = edge_signature
        for source in sources:
            selected = edge_selected[(source, label, edge_signature)]
            edge_rows.append(
                {
                    "label": label,
                    "selection_source": source,
                    "fragment_a_type": type_a,
                    "fragment_a_smiles": smiles_a,
                    "fragment_b_type": type_b,
                    "fragment_b_smiles": smiles_b,
                    "available_count": available,
                    "selected_count": selected,
                    "selection_rate": _rate(selected, available),
                }
            )

    return fragment_rows, edge_rows


def _rate(part, whole):
    if whole <= 0:
        return 0.0
    return round(float(part) / float(whole), 6)


def write_csv_rows(path, rows, fieldnames):
    """Write rows even when empty so downstream scripts get stable headers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_fragment_selection_metrics(records, output_dir):
    """Write sparse vector/matrix-friendly fragment selection metrics."""
    fragment_rows, edge_rows = build_fragment_selection_metric_rows(records)
    fragment_path = write_csv_rows(
        output_dir / "fragment_selection_metrics.csv",
        fragment_rows,
        [
            "label",
            "selection_source",
            "fragment_type",
            "fragment_smiles",
            "available_count",
            "selected_count",
            "selection_rate",
        ],
    )
    edge_path = write_csv_rows(
        output_dir / "fragment_pair_selection_metrics.csv",
        edge_rows,
        [
            "label",
            "selection_source",
            "fragment_a_type",
            "fragment_a_smiles",
            "fragment_b_type",
            "fragment_b_smiles",
            "available_count",
            "selected_count",
            "selection_rate",
        ],
    )
    print(f"Wrote fragment selection metrics: {fragment_path}")
    print(f"Wrote fragment-pair selection metrics: {edge_path}")
    return fragment_rows, edge_rows


def _metric_value(row, key, default=0.0):
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def _metric_int(row, key):
    return int(_metric_value(row, key, 0.0))


def _passes_support(row, min_available):
    return _metric_int(row, "available_count") >= min_available


def _limited_sorted(rows, primary_key, top_k):
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -_metric_value(row, primary_key),
            -_metric_int(row, "selected_count"),
            -_metric_int(row, "available_count"),
            tuple(str(row.get(key, "")) for key in sorted(row)),
        ),
    )
    if top_k is None or top_k <= 0:
        return sorted_rows
    return sorted_rows[:top_k]


def _top_within_groups(rows, group_fields, primary_key, top_k):
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    ranked = []
    for key in sorted(grouped):
        group_rows = _limited_sorted(grouped[key], primary_key, top_k)
        for rank, row in enumerate(group_rows, start=1):
            ranked.append(
                {
                    **dict(zip(group_fields, key)),
                    "rank_within_group": rank,
                    **row,
                }
            )
    return ranked


def _filter_summary_rows(name, rows, min_available):
    kept = sum(1 for row in rows if _passes_support(row, min_available))
    return {
        "metric_table": name,
        "min_available": min_available,
        "total_rows": len(rows),
        "kept_rows": kept,
        "dropped_low_support_rows": len(rows) - kept,
    }


def build_fragment_interpretation_tables(fragment_rows, edge_rows, min_available=10, top_k=25):
    """Build mentor-facing interpretation tables from raw selection metrics.

    Raw metric CSVs stay complete. These tables apply a support filter so rare
    one-off fragments do not dominate interpretation.
    """
    supported_fragments = [
        row for row in fragment_rows
        if _passes_support(row, min_available)
    ]
    supported_edges = [
        row for row in edge_rows
        if _passes_support(row, min_available)
    ]

    tables = {
        "interpretation_filter_summary.csv": [
            _filter_summary_rows("fragment_selection_metrics", fragment_rows, min_available),
            _filter_summary_rows("fragment_pair_selection_metrics", edge_rows, min_available),
        ],
        "top_fragments_by_selection_rate.csv": _limited_sorted(
            supported_fragments,
            "selection_rate",
            top_k,
        ),
        "top_fragments_by_selected_count.csv": _limited_sorted(
            supported_fragments,
            "selected_count",
            top_k,
        ),
        "top_fragment_pairs_by_selection_rate.csv": _limited_sorted(
            supported_edges,
            "selection_rate",
            top_k,
        ),
        "top_fragment_pairs_by_selected_count.csv": _limited_sorted(
            supported_edges,
            "selected_count",
            top_k,
        ),
        "top_functional_group_fragments.csv": _top_within_groups(
            supported_fragments,
            ["label", "selection_source", "fragment_type"],
            "selection_rate",
            top_k,
        ),
        "top_functional_group_pairs.csv": _top_within_groups(
            supported_edges,
            ["label", "selection_source", "fragment_a_type", "fragment_b_type"],
            "selection_rate",
            top_k,
        ),
    }
    return tables


def write_fragment_interpretation_tables(
    fragment_rows,
    edge_rows,
    output_dir,
    min_available=10,
    top_k=25,
):
    """Write support-filtered interpretation tables derived from raw metrics."""
    tables = build_fragment_interpretation_tables(
        fragment_rows,
        edge_rows,
        min_available=min_available,
        top_k=top_k,
    )
    interpretation_dir = output_dir / "interpretation"
    paths = []
    for filename, rows in tables.items():
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = ["empty"]
        path = write_csv_rows(interpretation_dir / filename, rows, fieldnames)
        paths.append(path)
    print(f"Wrote {len(paths)} interpretation table(s): {interpretation_dir}")
    return paths


def print_summary(records):
    """Print mean/median/min/max/stdev of the inter-view agreement metric."""
    if not records:
        print("No molecules analyzed.")
        return
    num_views = records[0]["num_views"]

    print("\n" + "=" * 78)
    print("INVARIANT SUBGRAPH ANALYSIS SUMMARY")
    print("=" * 78)
    print(f"Analyzed {len(records)} molecules across {num_views} view(s)\n")

    def _report(name, values):
        s = summarize(values)
        print(
            f"{name:<28} mean={s['mean']:6.2f}  median={s['median']:6.2f}  "
            f"min={s['min']:6.2f}  max={s['max']:6.2f}  stdev={s['stdev']:6.2f}"
        )

    if num_views >= 2:
        _report("Mean view agreement (IoU)", [r["row"]["mean_view_agreement"] for r in records])


def plot_agreement_heatmap(records, output_dir, num_views):
    """Heatmap of the average pairwise view-agreement (IoU) matrix."""
    if num_views < 2:
        print("Only one view; skipping agreement heatmap.")
        return
    avg = [[0.0] * num_views for _ in range(num_views)]
    for r in records:
        for i in range(num_views):
            for j in range(num_views):
                avg[i][j] += r["agreement"][i][j]
    for i in range(num_views):
        for j in range(num_views):
            avg[i][j] /= len(records)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(avg, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(num_views))
    ax.set_yticks(range(num_views))
    ax.set_xticklabels([f"V{v}" for v in range(num_views)])
    ax.set_yticklabels([f"V{v}" for v in range(num_views)])
    ax.set_title("Mean pairwise view agreement (atom IoU)")
    for i in range(num_views):
        for j in range(num_views):
            ax.text(j, i, f"{avg[i][j]:.2f}", ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax)
    path = output_dir / "view_agreement_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote view-agreement heatmap: {path}")


def mol_with_atom_indices(mol):
    """Copy a mol and annotate each atom with its index (matches viz script style)."""
    mol_copy = Chem.Mol(mol)
    for atom in mol_copy.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    return mol_copy


def internal_bond_ids(mol, atom_ids, atom_to_fragment=None, connected_fragments=None):
    """Bond indices whose both endpoints are within ``atom_ids``.

    An inter-fragment bond (its two atoms belong to different fragments) is only
    kept when that fragment pair is in ``connected_fragments`` — the set of
    fragment edges this panel actually extracted. This keeps the molecule panel
    consistent with the fragment graph: an edge is drawn between two fragments in
    the molecule only where the fragment graph highlights an edge between them.
    Intra-fragment bonds are always kept. When the fragment mapping is
    unavailable, every internal bond is kept (previous behavior).
    """
    atom_set = set(int(a) for a in atom_ids)
    bond_ids = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in atom_set or b not in atom_set:
            continue
        if atom_to_fragment and connected_fragments is not None:
            fa = atom_to_fragment[a] if a < len(atom_to_fragment) else None
            fb = atom_to_fragment[b] if b < len(atom_to_fragment) else None
            if fa is not None and fb is not None and fa != fb:
                if (min(fa, fb), max(fa, fb)) not in connected_fragments:
                    continue
        bond_ids.append(bond.GetIdx())
    return bond_ids


def _selected_fragment_pairs(edges, selected_edge_ids):
    """Unordered fragment pairs for the extracted (selected) fragment edges.

    ``selected_edge_ids`` index into ``edges`` (the fragment_edge_index columns),
    so each maps to the (u, v) fragment pair joined by that extracted edge.
    """
    pairs = set()
    for edge_id in selected_edge_ids:
        if 0 <= edge_id < len(edges):
            u, v = edges[edge_id]
            if u != v:
                pairs.add((min(u, v), max(u, v)))
    return pairs


def render_mol_highlight(
    mol, highlight_atoms, highlight_bonds, atom_colors=None, bond_colors=None, size=(420, 340)
):
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


def _fragment_colors(num_fragments):
    """Distinct RGB color per fragment id (tab20 palette, cycled beyond 20)."""
    cmap = plt.get_cmap("tab20")
    return {i: cmap(i % 20)[:3] for i in range(num_fragments)}


def _highlight_colors(mol, atom_ids, bond_ids, atom_to_fragment, frag_colors):
    """Map highlighted atoms/bonds to their fragment's color.

    Atoms take the color of the fragment they belong to; bonds take their
    fragment's color, or the average of the two endpoint colors when the bond
    bridges two fragments. Returns (atom_colors, bond_colors) dicts for RDKit.
    """
    atom_colors = {}
    for a in atom_ids:
        a = int(a)
        if a < len(atom_to_fragment) and atom_to_fragment[a] in frag_colors:
            atom_colors[a] = frag_colors[atom_to_fragment[a]]
    bond_colors = {}
    for b in bond_ids:
        bond = mol.GetBondWithIdx(int(b))
        c1 = atom_colors.get(bond.GetBeginAtomIdx())
        c2 = atom_colors.get(bond.GetEndAtomIdx())
        if c1 is None and c2 is None:
            continue
        c1 = c1 if c1 is not None else c2
        c2 = c2 if c2 is not None else c1
        bond_colors[int(b)] = tuple((x + y) / 2 for x, y in zip(c1, c2))
    return atom_colors, bond_colors


def _fragment_layout(num_fragments, edges):
    """Force-directed positions for fragment nodes, keyed by fragment id.

    Uses a seeded spring layout so the drawing follows the graph's connectivity
    (chains stretch out, rings stay round) and is identical across runs.
    Positions fall within [-1, 1] on both axes.
    """
    if num_fragments <= 0:
        return {}
    if num_fragments == 1:
        return {0: (0.0, 0.0)}
    graph = nx.Graph()
    graph.add_nodes_from(range(num_fragments))
    graph.add_edges_from((u, v) for u, v in edges if u != v)
    pos = nx.spring_layout(graph, seed=0)
    return {i: (float(x), float(y)) for i, (x, y) in pos.items()}


def draw_fragment_graph(ax, num_fragments, edges, selected_edge_ids, selected_frag_ids, frag_colors):
    """Draw the fragment graph on ``ax`` with the extracted edges/nodes highlighted.

    Nodes are fragments; edges are fragment_edge_index columns. Edges whose column
    index is in ``selected_edge_ids`` (the extracted fragment edges) are drawn bold in
    the blend of their endpoint fragments' colors; selected fragment nodes are filled
    with their fragment's color (matching the molecule-panel highlights).
    """
    pos = _fragment_layout(num_fragments, edges)
    selected_edges = set(selected_edge_ids)
    selected_nodes = set(selected_frag_ids)

    for edge_id, (u, v) in enumerate(edges):
        if u not in pos or v not in pos:
            continue
        chosen = edge_id in selected_edges
        if chosen:
            color = tuple((a + b) / 2 for a, b in zip(frag_colors[u], frag_colors[v]))
        else:
            color = "lightgray"
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=color,
            linewidth=3.0 if chosen else 1.0,
            zorder=2 if chosen else 1,
        )

    for frag_id, (x, y) in pos.items():
        chosen = frag_id in selected_nodes
        ax.scatter(
            [x], [y],
            s=340,
            c=[frag_colors[frag_id]] if chosen else "white",
            edgecolors="black",
            linewidths=2.0 if chosen else 1.0,
            zorder=3,
        )
        ax.text(x, y, f"F{frag_id}", ha="center", va="center", fontsize=8, zorder=4)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def build_molecule_figure(record):
    """Figure pairing, per view and aggregate, the molecule subgraph + fragment graph."""
    smiles = record["smiles"]
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = mol_with_atom_indices(mol)

    num_views = record["num_views"]
    panels = [
        (
            f"View {v}",
            record["per_view_atoms"][v],
            record["per_view_edge_ids"][v],
            record["per_view_frag_ids"][v],
        )
        for v in range(num_views)
    ]
    panels.append(
        ("Aggregated", record["agg_atoms"], record["agg_edge_ids"], record["agg_frag_ids"])
    )

    num_fragments = record["num_fragments"]
    edges = record["fragment_edges"]
    frag_colors = _fragment_colors(num_fragments)
    atom_to_fragment = record["atom_to_fragment"]

    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 2, figsize=(9, 3.4 * n_rows), squeeze=False)
    for i, (label, atoms, edge_ids, frag_ids) in enumerate(panels):
        ax_mol, ax_graph = axes[i][0], axes[i][1]

        connected_fragments = _selected_fragment_pairs(edges, edge_ids)
        bond_ids = internal_bond_ids(mol, atoms, atom_to_fragment, connected_fragments)
        atom_colors, bond_colors = _highlight_colors(
            mol, atoms, bond_ids, atom_to_fragment, frag_colors
        )
        image = render_mol_highlight(mol, atoms, bond_ids, atom_colors, bond_colors)
        ax_mol.imshow(np.asarray(image))
        ax_mol.axis("off")
        ax_mol.set_title(f"{label} subgraph: {len(atoms)} atoms", fontsize=10)

        draw_fragment_graph(ax_graph, num_fragments, edges, edge_ids, frag_ids, frag_colors)
        ax_graph.set_title(f"{label} fragment graph: {len(edge_ids)} edges", fontsize=10)

    title = record["molecule_id"] or record["row"]["molecule_idx"]
    fig.suptitle(f"Molecule {title}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def visualize_molecules(records, output_dir, num_visualize):
    """Render composite figures for the first ``num_visualize`` molecules."""
    if num_visualize <= 0:
        return
    viz_dir = output_dir / "molecules"
    viz_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for r in records[:num_visualize]:
        fig = build_molecule_figure(r)
        if fig is None:
            print(f"  Skipped molecule {r['row']['molecule_idx']} (no/unparsable SMILES).")
            continue
        label = _safe_filename(str(r["molecule_id"] or r["row"]["molecule_idx"]))
        path = viz_dir / f"molecule_{r['row']['molecule_idx']}_{label}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved += 1
    if saved:
        print(f"Wrote {saved} molecule visualization(s) to {viz_dir}")


def _safe_filename(text):
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in text).strip("_")
    return cleaned[:40] or "mol"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Trained checkpoint path (default: model_path or save_model from --config)",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help="Preprocessed data directory (default: processed_dir from --config)",
    )
    parser.add_argument("--split", type=str, default=None, help="Split to analyze (default: test)")
    parser.add_argument("--num_samples", type=int, default=None, help="Max molecules (default: all)")
    parser.add_argument("--num_visualize", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results/subgraph_analysis")
    parser.add_argument(
        "--metric_min_available",
        type=int,
        default=10,
        help="Minimum available_count required for interpretation tables.",
    )
    parser.add_argument(
        "--metric_top_k",
        type=int,
        default=25,
        help="Rows to keep in each ranked interpretation table; <=0 keeps all rows.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_config(args)

    # CLI values are merged into the config by build_config, so config lookups
    # cover both --model_path/--processed_dir and YAML-provided defaults.
    model_path = config.get("model_path") or config.get("save_model")
    if not model_path:
        raise ValueError(
            "No checkpoint specified: pass --model_path or set model_path/save_model in the config."
        )
    if not config.get("processed_dir"):
        raise ValueError(
            "No data directory specified: pass --processed_dir or set processed_dir in the config."
        )

    device = resolve_device(config)
    print(f"Using device: {device}")

    ensemble, dataloaders = load_ensemble(config, model_path, device)

    split = args.split or config.get("test_split", "test")
    loader = dataloaders.get(split)
    if loader is None:
        available = ", ".join(sorted(dataloaders)) or "none"
        raise ValueError(f"Split {split!r} not found. Available splits: {available}")

    print(f"Extracting subgraphs from split {split!r}...")
    records = collect_subgraph_records(ensemble, loader, device, num_samples=args.num_samples)
    if not records:
        print("No molecules found in the selected split.")
        return

    num_views = records[0]["num_views"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_summary(records)
    fragment_rows, edge_rows = write_fragment_selection_metrics(records, output_dir)
    write_fragment_interpretation_tables(
        fragment_rows,
        edge_rows,
        output_dir,
        min_available=args.metric_min_available,
        top_k=args.metric_top_k,
    )
    plot_agreement_heatmap(records, output_dir, num_views)
    visualize_molecules(records, output_dir, args.num_visualize)
    print(f"\nDone. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
