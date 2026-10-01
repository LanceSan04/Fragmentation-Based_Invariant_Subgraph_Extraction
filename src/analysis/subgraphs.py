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

For mixed-view runs, fragment availability and selection counts are method-aware:
group graph, BRICS, and RECAP fragments are counted in separate rows via the
``fragmentation_method`` column. Single-method legacy runs use the original
unprefixed graph fields, so their method is reported as ``default``.

A checkpoint must be produced first via ``python main.py train``; both the
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
        Split to analyze. Defaults to subgraph_split from config, then
        test_split, then "test".
    --num_samples N
        Optional cap on molecules analyzed. Default: all molecules in the split.
    --num_visualize N
        Number of molecule visualization PNGs to write. Defaults to
        subgraph_num_visualize from config, then 3.
    --output_dir PATH
        Output directory for CSVs and plots. Defaults to subgraph_analysis_dir
        from config, then results/subgraph_analysis.
    --metric_min_available N
        Minimum available_count required for interpretation tables. Defaults to
        subgraph_metric_min_available from config, then 10.
    --metric_top_k N
        Rows to keep in each ranked interpretation table. Use <=0 to keep all.
        Defaults to subgraph_metric_top_k from config, then 25.
    --batch_size N
        Override analysis DataLoader batch size.
    --device {cpu,cuda}
        Override device. Defaults to config/device resolution.

Example with explicit paths:
    python -m src.analysis.subgraphs \
        --model_path checkpoints/smoke_sugar.pt \
        --processed_dir data/processed/bbbp/group_graph \
        --split test --num_samples 20 --num_visualize 3 --device cpu

Example reusing a training config for both paths:
    python -m src.analysis.subgraphs \
        --config configs/experiment.yaml \
        --split test --num_samples 20 --num_visualize 3 --device cpu
"""

import argparse
import csv
import io
import math
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


from src.data.dataloader import build_split_dataloaders
from src.data.pyg_batch import fragment_view_field
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
from src.training.pipeline import (
    align_num_views_to_checkpoint,
    apply_checkpoint_defaults,
    build_config,
    Config,
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

        # per_view[v] and aggregate(s) are batched; unbatch to align with full_list order.
        per_view_lists = [pv.to_data_list() for pv in result["per_view"]]
        agg_list = result["aggregated"].to_data_list()
        aggregated_by_method_lists = {
            method: subgraph_batch.to_data_list()
            for method, subgraph_batch in result.get("aggregated_by_method", {}).items()
        }
        num_views = len(per_view_lists)
        view_methods = [
            getattr(view, "fragmentation_method", None) or "default" for view in ensemble.views
        ]

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
            agg_by_method = {
                method: subgraphs[i] for method, subgraphs in aggregated_by_method_lists.items()
            }

            per_view_edge_ids = [
                _id_list(getattr(sub, "selected_fragment_edge_ids", None)) for sub in per_view_subs
            ]
            per_view_frag_ids = [
                _id_list(getattr(sub, "selected_fragment_ids", None)) for sub in per_view_subs
            ]
            agg_edge_ids = _id_list(getattr(agg_sub, "selected_fragment_edge_ids", None))
            agg_frag_ids = _id_list(getattr(agg_sub, "selected_fragment_ids", None))
            method_fragment_data = {}
            methods = sorted(set(view_methods) | set(agg_by_method))
            for method in methods:
                method_key = None if method == "default" else method
                method_agg = agg_by_method.get(method)
                if method_agg is None and method == "default":
                    method_agg = agg_sub
                method_fragment_data[method] = {
                    "num_fragments": _num_fragments(full, method_key),
                    "fragment_edges": _edge_pairs(
                        _fragment_value(full, "fragment_edge_index", method_key)
                    ),
                    "fragment_signatures": _fragment_signatures(full, method_key),
                    "atom_to_fragment": _int_list(
                        _fragment_value(full, "atom_to_fragment", method_key)
                    ),
                    "agg_atoms": (
                        _atom_indices(getattr(method_agg, "selected_original_atom_indices", None))
                        if method_agg is not None
                        else []
                    ),
                    "agg_edge_ids": (
                        _id_list(getattr(method_agg, "selected_fragment_edge_ids", None))
                        if method_agg is not None
                        else []
                    ),
                    "agg_frag_ids": (
                        _id_list(getattr(method_agg, "selected_fragment_ids", None))
                        if method_agg is not None
                        else []
                    ),
                }
            primary_method = view_methods[0] if view_methods else "default"
            primary_data = method_fragment_data[primary_method]
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
                    "view_methods": view_methods,
                    "smiles": row["smiles"],
                    "molecule_id": row["molecule_id"],
                    "per_view_atoms": per_view_atoms,
                    "agg_atoms": agg_atoms,
                    "agreement": agreement,
                    "num_fragments": primary_data["num_fragments"],
                    "fragment_signatures": primary_data["fragment_signatures"],
                    "fragment_edges": primary_data["fragment_edges"],
                    "per_view_edge_ids": per_view_edge_ids,
                    "per_view_frag_ids": per_view_frag_ids,
                    "agg_edge_ids": agg_edge_ids,
                    "agg_frag_ids": agg_frag_ids,
                    "atom_to_fragment": primary_data["atom_to_fragment"],
                    "method_fragment_data": method_fragment_data,
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


def _fragment_value(full_data, field, method=None):
    """Read a default or method-prefixed fragment field."""
    name = fragment_view_field(method, field)
    return getattr(full_data, name, None)


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


def _fragment_signatures(full_data, method=None):
    """Return stable fragment identity tuples aligned to fragment ids.

    The first implementation uses the metadata already preserved by preprocessing:
    ``(fragment_type, fragment_smiles)``. This is specific enough to distinguish
    Group Graph split fragments from ordinary fragments while still aggregating
    the same chemistry across molecules.
    """
    num_fragments = _num_fragments(full_data, method)
    fragment_types = list(_fragment_value(full_data, "fragment_types", method) or [])
    fragment_smiles = list(_fragment_value(full_data, "fragment_smiles", method) or [])
    signatures = []
    for fragment_id in range(num_fragments):
        fragment_type = (
            str(fragment_types[fragment_id]) if fragment_id < len(fragment_types) else "unknown"
        )
        fragment_smiles_value = (
            str(fragment_smiles[fragment_id])
            if fragment_id < len(fragment_smiles)
            else f"fragment_{fragment_id}"
        )
        signatures.append((fragment_type, fragment_smiles_value))
    return signatures


def _num_fragments(full_data, method=None):
    """Number of fragments in a molecule (rows of fragment_x_static)."""
    frag_x = _fragment_value(full_data, "fragment_x_static", method)
    if frag_x is not None:
        return int(frag_x.size(0))
    atom_to_fragment = _fragment_value(full_data, "atom_to_fragment", method)
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


def _source_names(view_indices):
    return [f"view_{view_idx}" for view_idx in view_indices] + [
        "any_view",
        "aggregated",
    ]


def _selected_ids_by_source(record, selected_key, method):
    """Return selected fragment/edge ids grouped by view, any-view, and aggregate."""
    per_view = record[selected_key]
    view_methods = record.get("view_methods") or ["default"] * len(per_view)
    by_source = {
        f"view_{view_idx}": set(ids)
        for view_idx, ids in enumerate(per_view)
        if view_idx < len(view_methods) and view_methods[view_idx] == method
    }
    any_view = set()
    for view_idx, ids in enumerate(per_view):
        if view_idx >= len(view_methods) or view_methods[view_idx] != method:
            continue
        any_view.update(ids)
    by_source["any_view"] = any_view
    method_data = _record_method_data(record, method)
    aggregate_key = "agg_frag_ids" if selected_key == "per_view_frag_ids" else "agg_edge_ids"
    by_source["aggregated"] = set(method_data.get(aggregate_key, []))
    return by_source


def _record_methods(record):
    if "method_fragment_data" in record:
        return list(record["method_fragment_data"])
    return ["default"]


def _record_method_data(record, method):
    if "method_fragment_data" in record:
        return record["method_fragment_data"][method]
    return {
        "num_fragments": record.get("num_fragments", 0),
        "fragment_signatures": record.get("fragment_signatures", []),
        "fragment_edges": record.get("fragment_edges", []),
        "atom_to_fragment": record.get("atom_to_fragment", []),
        "agg_atoms": record.get("agg_atoms", []),
        "agg_edge_ids": record.get("agg_edge_ids", []),
        "agg_frag_ids": record.get("agg_frag_ids", []),
    }


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

    fragment_available = Counter()
    fragment_selected = Counter()
    fragment_sources = {}
    edge_available = Counter()
    edge_selected = Counter()
    edge_sources = {}

    for record in records:
        label = str(record["row"].get("label", "unknown"))
        view_methods = record.get("view_methods") or ["default"] * record["num_views"]

        for method in _record_methods(record):
            method_data = _record_method_data(record, method)
            fragment_signatures = method_data["fragment_signatures"]
            view_indices = [
                idx for idx, view_method in enumerate(view_methods) if view_method == method
            ]
            sources = _source_names(view_indices)

            for fragment_id, signature in enumerate(fragment_signatures):
                fragment_available[(method, label, signature)] += 1
                fragment_sources.setdefault((method, label, signature), set()).update(sources)

            fragment_ids_by_source = _selected_ids_by_source(
                record,
                "per_view_frag_ids",
                method,
            )
            for source, selected_ids in fragment_ids_by_source.items():
                for fragment_id in selected_ids:
                    if 0 <= fragment_id < len(fragment_signatures):
                        fragment_selected[
                            (method, source, label, fragment_signatures[fragment_id])
                        ] += 1

            edge_signatures = []
            for edge_id, (src, dst) in enumerate(method_data["fragment_edges"]):
                if src >= len(fragment_signatures) or dst >= len(fragment_signatures):
                    continue
                edge_signature = _canonical_fragment_pair(
                    fragment_signatures[src],
                    fragment_signatures[dst],
                )
                edge_signatures.append((edge_id, edge_signature))
                edge_available[(method, label, edge_signature)] += 1
                edge_sources.setdefault((method, label, edge_signature), set()).update(sources)

            edge_ids_by_source = _selected_ids_by_source(record, "per_view_edge_ids", method)
            edge_signature_by_id = dict(edge_signatures)
            for source, selected_ids in edge_ids_by_source.items():
                for edge_id in selected_ids:
                    edge_signature = edge_signature_by_id.get(edge_id)
                    if edge_signature is not None:
                        edge_selected[(method, source, label, edge_signature)] += 1

    fragment_rows = []
    for method, label, signature in sorted(fragment_available):
        available = fragment_available[(method, label, signature)]
        fragment_type, fragment_smiles = signature
        sources = sorted(
            fragment_sources.get((method, label, signature), {"any_view", "aggregated"})
        )
        for source in sources:
            selected = fragment_selected[(method, source, label, signature)]
            fragment_rows.append(
                {
                    "fragmentation_method": method,
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
    for method, label, edge_signature in sorted(edge_available):
        available = edge_available[(method, label, edge_signature)]
        (type_a, smiles_a), (type_b, smiles_b) = edge_signature
        sources = sorted(
            edge_sources.get((method, label, edge_signature), {"any_view", "aggregated"})
        )
        for source in sources:
            selected = edge_selected[(method, source, label, edge_signature)]
            edge_rows.append(
                {
                    "fragmentation_method": method,
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
            "fragmentation_method",
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
            "fragmentation_method",
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
        key = tuple(row.get(field, "default") for field in group_fields)
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
    supported_fragments = [row for row in fragment_rows if _passes_support(row, min_available)]
    supported_edges = [row for row in edge_rows if _passes_support(row, min_available)]

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
            ["fragmentation_method", "label", "selection_source", "fragment_type"],
            "selection_rate",
            top_k,
        ),
        "top_functional_group_pairs.csv": _top_within_groups(
            supported_edges,
            [
                "fragmentation_method",
                "label",
                "selection_source",
                "fragment_a_type",
                "fragment_b_type",
            ],
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


# Source-metric identity columns that define a fragment / fragment-pair. In the
# consolidated contrast table both kinds are normalized to the fragment_a_*/fragment_b_*
# columns; a single fragment leaves the fragment_b_* columns blank.
_CONTRAST_IDENTITY = {
    "fragment": ["fragment_type", "fragment_smiles"],
    "pair": [
        "fragment_a_type",
        "fragment_a_smiles",
        "fragment_b_type",
        "fragment_b_smiles",
    ],
}

# Stable column order for the consolidated label-contrast table (so an empty table
# still gets headers).
_CONTRAST_FIELDNAMES = [
    "fragmentation_method",
    "kind",
    "selection_source",
    "fragment_a_type",
    "fragment_a_smiles",
    "fragment_b_type",
    "fragment_b_smiles",
    "neg_label",
    "pos_label",
    "available_neg",
    "available_pos",
    "selected_neg",
    "selected_pos",
    "selection_rate_neg",
    "selection_rate_pos",
    "rate_difference",
    "selection_log_odds",
    "min_support",
]


def resolve_binary_labels(records):
    """Return the two distinct (non-unknown) labels as (neg, pos), else None.

    The label contrast only makes sense for binary classification, so a split with one
    class or more than two is skipped by the caller.
    """
    labels = sorted(
        {
            str(record["row"].get("label", "unknown"))
            for record in records
            if str(record["row"].get("label", "unknown")) != "unknown"
        }
    )
    if len(labels) != 2:
        return None
    return labels[0], labels[1]


def _smoothed_logit(selected, available):
    """Laplace-smoothed logit of a selection rate; stable at rate 0 or 1."""
    rate = (float(selected) + 0.5) / (float(available) + 1.0)
    return math.log(rate / (1.0 - rate))


def _contrast_identity_columns(kind, identity_values):
    """Normalize a group key's identity into unified fragment_a_*/fragment_b_* columns."""
    if kind == "fragment":
        fragment_type, fragment_smiles = identity_values
        return {
            "fragment_a_type": fragment_type,
            "fragment_a_smiles": fragment_smiles,
            "fragment_b_type": "",
            "fragment_b_smiles": "",
        }
    return dict(zip(_CONTRAST_IDENTITY["pair"], identity_values))


def build_label_contrast_rows(metric_rows, kind, neg_label, pos_label, min_available):
    """Pivot per-label selection metrics into one contrast row per fragment/pair.

    Each row pairs the negative- and positive-label selection behavior for a single
    ``(fragmentation_method, selection_source, fragment signature)`` and reports the
    difference in selection rate. That rate is conditional on the fragment being present
    in the molecule, so the difference reflects the model *choosing* the fragment more
    for one class rather than the class simply containing it more often. Only signatures
    with ``available_count >= min_available`` in *both* labels are kept, and rows are
    returned sorted by descending absolute rate difference.
    """
    identity = _CONTRAST_IDENTITY[kind]
    grouped = {}
    for row in metric_rows:
        key = (
            row.get("fragmentation_method", "default"),
            row["selection_source"],
            *(row[field] for field in identity),
        )
        grouped.setdefault(key, {})[str(row["label"])] = row

    contrast_rows = []
    for key in sorted(grouped):
        by_label = grouped[key]
        neg_row = by_label.get(neg_label)
        pos_row = by_label.get(pos_label)
        if neg_row is None or pos_row is None:
            continue
        available_neg = _metric_int(neg_row, "available_count")
        available_pos = _metric_int(pos_row, "available_count")
        if available_neg < min_available or available_pos < min_available:
            continue

        selected_neg = _metric_int(neg_row, "selected_count")
        selected_pos = _metric_int(pos_row, "selected_count")
        rate_neg = _metric_value(neg_row, "selection_rate")
        rate_pos = _metric_value(pos_row, "selection_rate")

        contrast_rows.append(
            {
                "fragmentation_method": key[0],
                "kind": kind,
                "selection_source": key[1],
                **_contrast_identity_columns(kind, key[2:]),
                "neg_label": neg_label,
                "pos_label": pos_label,
                "available_neg": available_neg,
                "available_pos": available_pos,
                "selected_neg": selected_neg,
                "selected_pos": selected_pos,
                "selection_rate_neg": round(rate_neg, 6),
                "selection_rate_pos": round(rate_pos, 6),
                "rate_difference": round(rate_pos - rate_neg, 6),
                "selection_log_odds": round(
                    _smoothed_logit(selected_pos, available_pos)
                    - _smoothed_logit(selected_neg, available_neg),
                    6,
                ),
                "min_support": min(available_neg, available_pos),
            }
        )

    contrast_rows.sort(key=lambda row: -abs(row["rate_difference"]))
    return contrast_rows


def write_label_contrast_table(
    fragment_rows,
    edge_rows,
    output_dir,
    neg_label,
    pos_label,
    min_available=10,
):
    """Write one consolidated label-contrast CSV for fragments and fragment-pairs.

    Fragments and pairs share the fragment_a_*/fragment_b_* schema and are told apart by
    the ``kind`` column, grouped by fragmentation method then kind, sorted within each by
    descending absolute rate difference (strongest label-discriminative rows first).
    """
    contrast_frag = build_label_contrast_rows(
        fragment_rows, "fragment", neg_label, pos_label, min_available
    )
    contrast_pair = build_label_contrast_rows(
        edge_rows, "pair", neg_label, pos_label, min_available
    )
    combined = contrast_frag + contrast_pair
    combined.sort(
        key=lambda row: (
            row["fragmentation_method"],
            row["kind"],
            -abs(row["rate_difference"]),
        )
    )
    path = write_csv_rows(output_dir / "label_contrast.csv", combined, _CONTRAST_FIELDNAMES)
    print(f"Wrote label-contrast table: {path} (neg_label={neg_label}, pos_label={pos_label})")
    return contrast_frag, contrast_pair


def _contrast_methods(contrast_rows):
    """Fragmentation methods present in a set of contrast rows."""
    return sorted({row["fragmentation_method"] for row in contrast_rows})


def _rows_for_method(contrast_rows, method):
    return [row for row in contrast_rows if row["fragmentation_method"] == method]


def _aggregated_only(contrast_rows):
    """Contrast rows from the shared aggregated subgraph (the model's decision path)."""
    return [row for row in contrast_rows if row["selection_source"] == "aggregated"]


def _view_sources(contrast_rows):
    """View source names present in these rows, ordered by view index.

    Views are method-scoped: a fragmentation method only owns the views assigned to it,
    so the columns are read off the rows rather than assumed to be range(num_views).
    """
    sources = {
        row["selection_source"]
        for row in contrast_rows
        if row["selection_source"].startswith("view_")
        and row["selection_source"].split("_", 1)[1].isdigit()
    }
    return sorted(sources, key=lambda source: int(source.split("_", 1)[1]))


def _contrast_signature_key(row, kind):
    """Identity tuple for a contrast row, used to match a fragment across sources."""
    if kind == "fragment":
        return (row["fragment_a_type"], row["fragment_a_smiles"])
    return (
        row["fragment_a_type"],
        row["fragment_a_smiles"],
        row["fragment_b_type"],
        row["fragment_b_smiles"],
    )


def _contrast_row_label(row, kind):
    """Short display label for a contrast row (fragment SMILES or pair 'A—B')."""
    if kind == "fragment":
        return str(row.get("fragment_a_smiles", ""))
    return f"{row.get('fragment_a_smiles', '')}—{row.get('fragment_b_smiles', '')}"


def _top_aggregated_rows(contrast_rows, top_n):
    """Aggregated-source rows sorted by descending absolute rate difference (top-N)."""
    rows = sorted(_aggregated_only(contrast_rows), key=lambda row: -abs(row["rate_difference"]))
    return rows[:top_n]


def plot_label_dumbbell(
    contrast_rows, output_dir, filename, kind, neg_label, pos_label, method, top_n=20
):
    """Dumbbell chart: each fragment/pair as its two per-label selection rates joined.

    One row per top-discriminative fragment; a dot at each label's selection rate joined
    by a line, so the absolute selection levels and the gap between them both read at a
    glance. ``contrast_rows`` must already be scoped to a single fragmentation method.
    """
    rows = _top_aggregated_rows(contrast_rows, top_n)
    if not rows:
        print(f"  No aggregated contrast rows; skipping {filename}.")
        return
    rows.sort(key=lambda row: row["rate_difference"])  # largest positive gap ends up on top
    display_labels = [_contrast_row_label(row, kind) for row in rows]
    neg_vals = [row["selection_rate_neg"] for row in rows]
    pos_vals = [row["selection_rate_pos"] for row in rows]
    positions = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.4 * len(rows) + 1.0)))
    for position, neg_val, pos_val in zip(positions, neg_vals, pos_vals):
        ax.plot([neg_val, pos_val], [position, position], color="#c7cbd1", linewidth=2.0, zorder=1)
    ax.scatter(neg_vals, positions, s=60, color="#d62728", label=f"label {neg_label}", zorder=2)
    ax.scatter(pos_vals, positions, s=60, color="#1f77b4", label=f"label {pos_label}", zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels(display_labels, fontsize=7)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("selection rate (aggregated subgraph)")
    ax.set_title(f"Top label-discriminative {kind}s by selection rate\nmethod: {method}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", color="#e8eaed", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def _perview_panel_data(contrast_rows, kind, top_n):
    """Compute (view_sources, display_labels, matrix) for one kind's per-view heatmap panel."""
    top_rows = _top_aggregated_rows(contrast_rows, top_n)
    view_sources = _view_sources(contrast_rows)
    if not top_rows or not view_sources:
        return None

    per_view_diff = {}
    for row in contrast_rows:
        if row["selection_source"] in view_sources:
            key = _contrast_signature_key(row, kind)
            per_view_diff.setdefault(key, {})[row["selection_source"]] = row["rate_difference"]

    display_labels = [_contrast_row_label(row, kind) for row in top_rows]
    matrix = np.array(
        [
            [
                per_view_diff.get(_contrast_signature_key(row, kind), {}).get(source, np.nan)
                for source in view_sources
            ]
            for row in top_rows
        ],
        dtype=float,
    )
    return view_sources, display_labels, matrix


def plot_perview_consistency(
    contrast_frag_rows,
    contrast_pair_rows,
    output_dir,
    filename,
    neg_label,
    pos_label,
    method,
    top_n=18,
):
    """Combined fragment + pair heatmap of each view's own label rate-difference.

    Rows are the top-discriminative fragments/pairs (ranked by the aggregated rate
    difference); columns are the views belonging to this fragmentation method. When every
    column agrees in sign, each independently-trained view learned the same label
    association, which is evidence of a real pattern rather than a single-view artifact.
    The fragment and pair panels share one title and one colorbar (rate-difference axis).
    ``contrast_frag_rows``/``contrast_pair_rows`` must already be scoped to a single
    fragmentation method.
    """
    panels = []
    for kind, rows in (("fragment", contrast_frag_rows), ("pair", contrast_pair_rows)):
        data = _perview_panel_data(rows, kind, top_n)
        if data is not None:
            panels.append((kind, *data))
    if not panels:
        print(f"  No per-view contrast rows; skipping {filename}.")
        return

    finite = np.concatenate([m[np.isfinite(m)] for _, _, _, m in panels])
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = vmax or 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e8eaed")

    n_rows_max = max(len(labels) for _, _, labels, _ in panels)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(
            sum(max(4.5, 0.9 * len(vs) + 3.5) for _, vs, _, _ in panels),
            max(4.0, 0.55 * n_rows_max + 1.5),
        ),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(h_pad=0.12)
    axes = [axes] if len(panels) == 1 else list(axes)

    im = None
    for ax, (kind, view_sources, display_labels, matrix) in zip(axes, panels):
        im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(view_sources)))
        ax.set_xticklabels(
            [source.replace("view_", "V") for source in view_sources],
            fontsize=17,
            fontweight="bold",
        )
        ax.set_yticks(range(len(display_labels)))
        ax.set_yticklabels(display_labels, fontsize=15)
        ax.set_title(f"{kind.capitalize()}s", fontsize=24, fontweight="bold", pad=10)

    fig.suptitle(
        f"Per-view label selection-rate difference\n"
        f"Red favors label {pos_label}; Blue favors label {neg_label}",
        fontsize=32,
        fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label(
        f"rate difference (label {pos_label} − label {neg_label})",
        fontsize=18,
        fontweight="bold",
    )
    cbar.ax.tick_params(labelsize=14)
    path = output_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def run_label_contrast_analysis(records, fragment_rows, edge_rows, output_dir, min_available):
    """Build the consolidated label-contrast table and per-method plots when binary."""
    labels = resolve_binary_labels(records)
    if labels is None:
        print("Skipping label-contrast analysis (need exactly two label classes).")
        return None
    neg_label, pos_label = labels
    contrast_frag, contrast_pair = write_label_contrast_table(
        fragment_rows,
        edge_rows,
        output_dir,
        neg_label,
        pos_label,
        min_available=min_available,
    )

    for kind, contrast_rows in (("fragment", contrast_frag), ("pair", contrast_pair)):
        for method in _contrast_methods(contrast_rows):
            method_rows = _rows_for_method(contrast_rows, method)
            suffix = _safe_filename(method)
            plot_label_dumbbell(
                method_rows,
                output_dir,
                f"label_dumbbell_{kind}s_{suffix}.png",
                kind,
                neg_label,
                pos_label,
                method,
            )

    all_methods = sorted(
        set(_contrast_methods(contrast_frag)) | set(_contrast_methods(contrast_pair))
    )
    for method in all_methods:
        suffix = _safe_filename(method)
        plot_perview_consistency(
            _rows_for_method(contrast_frag, method),
            _rows_for_method(contrast_pair, method),
            output_dir,
            f"label_perview_{suffix}.png",
            neg_label,
            pos_label,
            method,
        )
    return contrast_frag, contrast_pair


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


def draw_fragment_graph(
    ax, num_fragments, edges, selected_edge_ids, selected_frag_ids, frag_colors
):
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
            [x],
            [y],
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
            f"View {v} ({record.get('view_methods', ['default'] * num_views)[v]})",
            record.get("view_methods", ["default"] * num_views)[v],
            record["per_view_atoms"][v],
            record["per_view_edge_ids"][v],
            record["per_view_frag_ids"][v],
        )
        for v in range(num_views)
    ]
    for method, method_data in record.get("method_fragment_data", {}).items():
        panels.append(
            (
                f"Aggregated ({method})",
                method,
                method_data["agg_atoms"],
                method_data["agg_edge_ids"],
                method_data["agg_frag_ids"],
            )
        )
    if "method_fragment_data" not in record:
        panels.append(
            (
                "Aggregated",
                "default",
                record["agg_atoms"],
                record["agg_edge_ids"],
                record["agg_frag_ids"],
            )
        )

    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 2, figsize=(9, 3.4 * n_rows), squeeze=False)
    for i, (label, method, atoms, edge_ids, frag_ids) in enumerate(panels):
        ax_mol, ax_graph = axes[i][0], axes[i][1]
        method_data = _record_method_data(record, method)
        num_fragments = method_data["num_fragments"]
        edges = method_data["fragment_edges"]
        frag_colors = _fragment_colors(num_fragments)
        atom_to_fragment = method_data["atom_to_fragment"]

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


def _config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _set_config_value(config, key, value):
    if value is None:
        return
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


def _int_option(config, key, default):
    value = _config_value(config, key, default)
    if value is None or value == "":
        return default
    return int(value)


def _optional_int_option(config, key):
    value = _config_value(config, key, None)
    if value is None or value == "":
        return None
    return int(value)


def resolve_subgraph_analysis_options(
    config,
    *,
    model_path=None,
    processed_dir=None,
    split=None,
    num_samples=None,
    num_visualize=None,
    output_dir=None,
    metric_min_available=None,
    metric_top_k=None,
    batch_size=None,
    device=None,
):
    """Resolve subgraph-analysis settings from explicit args, config, then defaults."""
    if not isinstance(config, Config):
        config = Config(config)

    _set_config_value(config, "model_path", model_path)
    _set_config_value(config, "processed_dir", processed_dir)
    _set_config_value(config, "batch_size", batch_size)
    _set_config_value(config, "device", device)

    resolved_model_path = _config_value(config, "model_path") or _config_value(config, "save_model")
    if not resolved_model_path:
        raise ValueError(
            "No checkpoint specified: pass --model_path/--model-path or set "
            "model_path/save_model in the config."
        )

    resolved_processed_dir = _config_value(config, "processed_dir")
    if not resolved_processed_dir:
        raise ValueError(
            "No data directory specified: pass --processed_dir/--processed-dir or "
            "set processed_dir in the config."
        )

    resolved_split = (
        split
        or _config_value(config, "subgraph_split", None)
        or _config_value(config, "test_split", "test")
    )
    resolved_output_dir = (
        output_dir
        or _config_value(config, "subgraph_analysis_dir", None)
        or "results/subgraph_analysis"
    )
    return {
        "config": config,
        "model_path": resolved_model_path,
        "processed_dir": resolved_processed_dir,
        "split": resolved_split,
        "num_samples": (
            int(num_samples)
            if num_samples is not None
            else _optional_int_option(config, "subgraph_num_samples")
        ),
        "num_visualize": (
            int(num_visualize)
            if num_visualize is not None
            else _int_option(config, "subgraph_num_visualize", 3)
        ),
        "output_dir": Path(resolved_output_dir),
        "metric_min_available": (
            int(metric_min_available)
            if metric_min_available is not None
            else _int_option(config, "subgraph_metric_min_available", 10)
        ),
        "metric_top_k": (
            int(metric_top_k)
            if metric_top_k is not None
            else _int_option(config, "subgraph_metric_top_k", 25)
        ),
    }


def run_subgraph_analysis(
    config,
    *,
    model_path=None,
    processed_dir=None,
    split=None,
    num_samples=None,
    num_visualize=None,
    output_dir=None,
    metric_min_available=None,
    metric_top_k=None,
    batch_size=None,
    device=None,
):
    """Run trained-checkpoint subgraph analysis using config-backed defaults."""
    options = resolve_subgraph_analysis_options(
        config,
        model_path=model_path,
        processed_dir=processed_dir,
        split=split,
        num_samples=num_samples,
        num_visualize=num_visualize,
        output_dir=output_dir,
        metric_min_available=metric_min_available,
        metric_top_k=metric_top_k,
        batch_size=batch_size,
        device=device,
    )
    config = options["config"]

    _set_config_value(config, "model_path", options["model_path"])
    _set_config_value(config, "processed_dir", options["processed_dir"])

    device = resolve_device(config)
    print(f"Using device: {device}")

    ensemble, dataloaders = load_ensemble(config, options["model_path"], device)

    split = options["split"]
    loader = dataloaders.get(split)
    if loader is None:
        available = ", ".join(sorted(dataloaders)) or "none"
        raise ValueError(f"Split {split!r} not found. Available splits: {available}")

    print(f"Extracting subgraphs from split {split!r}...")
    records = collect_subgraph_records(
        ensemble,
        loader,
        device,
        num_samples=options["num_samples"],
    )
    if not records:
        print("No molecules found in the selected split.")
        return {
            "records": records,
            "output_dir": options["output_dir"],
            "split": split,
        }

    num_views = records[0]["num_views"]
    output_dir = options["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print_summary(records)
    fragment_rows, edge_rows = write_fragment_selection_metrics(records, output_dir)
    interpretation_paths = write_fragment_interpretation_tables(
        fragment_rows,
        edge_rows,
        output_dir,
        min_available=options["metric_min_available"],
        top_k=options["metric_top_k"],
    )
    run_label_contrast_analysis(
        records,
        fragment_rows,
        edge_rows,
        output_dir,
        min_available=options["metric_min_available"],
    )
    plot_agreement_heatmap(records, output_dir, num_views)
    visualize_molecules(records, output_dir, options["num_visualize"])
    print(f"\nDone. Outputs in {output_dir}")
    return {
        "records": records,
        "fragment_rows": fragment_rows,
        "edge_rows": edge_rows,
        "interpretation_paths": interpretation_paths,
        "output_dir": output_dir,
        "split": split,
        "num_views": num_views,
    }


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
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Max molecules (default: all)"
    )
    parser.add_argument("--num_visualize", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--metric_min_available",
        type=int,
        default=None,
        help="Minimum available_count required for interpretation tables.",
    )
    parser.add_argument(
        "--metric_top_k",
        type=int,
        default=None,
        help="Rows to keep in each ranked interpretation table; <=0 keeps all rows.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_config(args)
    run_subgraph_analysis(
        config,
        model_path=args.model_path,
        processed_dir=args.processed_dir,
        split=args.split,
        num_samples=args.num_samples,
        num_visualize=args.num_visualize,
        output_dir=args.output_dir,
        metric_min_available=args.metric_min_available,
        metric_top_k=args.metric_top_k,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
