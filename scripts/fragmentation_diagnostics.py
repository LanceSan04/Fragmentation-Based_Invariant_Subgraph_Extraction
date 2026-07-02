#!/usr/bin/env python
"""Summarize processed split fragmentation statistics and make report plots.

Example:
    .\.venv\Scripts\python.exe scripts\fragmentation_diagnostics.py ^
        data\processed\drugood_lbap_core_ec50_assay\group_graph ^
        --splits train ood_val ood_test iid_val iid_test ^
        --output-dir reports\fragmentation_diagnostics
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def load_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dictionary payload: {path}")
    if "metadata" not in payload or "molecule_data" not in payload:
        raise ValueError(f"Processed split is missing metadata or molecule_data: {path}")
    return payload


def tensor_to_scalar(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        value = value.reshape(-1)[0].item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def safe_attr(data, name, default=None):
    return getattr(data, name, default)


def fragment_atom_counts(data):
    num_fragments = int(data.fragment_x_static.size(0))
    if num_fragments == 0:
        return []
    counts = torch.bincount(data.atom_to_fragment.long(), minlength=num_fragments)
    return [int(value) for value in counts.tolist()]


def fragment_internal_bond_counts(data):
    num_fragments = int(data.fragment_x_static.size(0))
    counts = [0] * num_fragments
    bond_index = safe_attr(data, "bond_index")
    atom_to_fragment = safe_attr(data, "atom_to_fragment")
    if (
        bond_index is None
        or atom_to_fragment is None
        or bond_index.numel() == 0
        or num_fragments == 0
    ):
        return counts

    src_frag = atom_to_fragment[bond_index[0]].long()
    dst_frag = atom_to_fragment[bond_index[1]].long()
    internal = src_frag == dst_frag
    if internal.any():
        internal_fragments = src_frag[internal]
        bincount = torch.bincount(internal_fragments, minlength=num_fragments)
        counts = [int(value) for value in bincount.tolist()]
    return counts


def density(num_edges, num_nodes):
    if num_nodes <= 1:
        return 0.0
    return float(num_edges) / float(num_nodes * (num_nodes - 1) / 2.0)


def molecule_row(split_name, split_index, data):
    num_atoms = int(data.x.size(0))
    num_bonds = int(data.bond_attr.size(0))
    num_fragments = int(data.fragment_x_static.size(0))
    num_fragment_edges = int(data.fragment_edge_index.size(1))
    atom_counts = fragment_atom_counts(data)
    internal_bond_counts = fragment_internal_bond_counts(data)
    zero_edge = num_fragment_edges == 0

    return {
        "split": split_name,
        "split_index": split_index,
        "molecule_id": safe_attr(data, "molecule_id", ""),
        "smiles": safe_attr(data, "smiles", ""),
        "label": tensor_to_scalar(safe_attr(data, "y", None)),
        "environment_or_group": safe_attr(data, "environment_or_group", ""),
        "num_atoms": num_atoms,
        "num_bonds": num_bonds,
        "num_fragments": num_fragments,
        "num_fragment_edges": num_fragment_edges,
        "fragment_edge_density": density(num_fragment_edges, num_fragments),
        "mean_atoms_per_fragment": mean(atom_counts),
        "median_atoms_per_fragment": median(atom_counts),
        "max_atoms_per_fragment": max(atom_counts) if atom_counts else 0,
        "mean_internal_bonds_per_fragment": mean(internal_bond_counts),
        "single_fragment": num_fragments == 1,
        "zero_fragment_edges": zero_edge,
    }


def fragment_rows(split_name, split_index, data):
    atom_counts = fragment_atom_counts(data)
    internal_bond_counts = fragment_internal_bond_counts(data)
    fragment_types = list(safe_attr(data, "fragment_types", []) or [])
    fragment_smiles = list(safe_attr(data, "fragment_smiles", []) or [])
    rows = []
    for fragment_id, atom_count in enumerate(atom_counts):
        rows.append(
            {
                "split": split_name,
                "split_index": split_index,
                "molecule_id": safe_attr(data, "molecule_id", ""),
                "fragment_id": fragment_id,
                "fragment_type": fragment_types[fragment_id] if fragment_id < len(fragment_types) else "",
                "fragment_smiles": fragment_smiles[fragment_id] if fragment_id < len(fragment_smiles) else "",
                "num_atoms": atom_count,
                "num_internal_bonds": internal_bond_counts[fragment_id],
            }
        )
    return rows


def mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def stdev(values):
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def summarize_numeric(values):
    values = list(values)
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
        }
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "std": stdev(values),
    }


def split_summary_rows(molecule_rows, fragment_rows_all, top_k):
    by_split = defaultdict(list)
    fragments_by_split = defaultdict(list)
    for row in molecule_rows:
        by_split[row["split"]].append(row)
    for row in fragment_rows_all:
        fragments_by_split[row["split"]].append(row)

    rows = []
    for split_name in sorted(by_split):
        rows_for_split = by_split[split_name]
        fragments = fragments_by_split[split_name]
        label_counts = Counter(str(row["label"]) for row in rows_for_split)
        domain_counts = Counter(str(row["environment_or_group"]) for row in rows_for_split)
        type_counts = Counter(row["fragment_type"] for row in fragments)
        smiles_counts = Counter(row["fragment_smiles"] for row in fragments if row["fragment_smiles"])

        summary = {
            "split": split_name,
            "num_molecules": len(rows_for_split),
            "num_fragments_total": len(fragments),
            "single_fragment_molecules": sum(bool(row["single_fragment"]) for row in rows_for_split),
            "zero_fragment_edge_molecules": sum(bool(row["zero_fragment_edges"]) for row in rows_for_split),
            "label_counts_json": json.dumps(dict(label_counts), sort_keys=True),
            "top_domains_json": json.dumps(dict(domain_counts.most_common(top_k)), sort_keys=True),
            "fragment_type_counts_json": json.dumps(dict(type_counts), sort_keys=True),
            "top_fragment_smiles_json": json.dumps(dict(smiles_counts.most_common(top_k)), sort_keys=True),
        }
        for column in (
            "num_atoms",
            "num_bonds",
            "num_fragments",
            "num_fragment_edges",
            "fragment_edge_density",
            "mean_atoms_per_fragment",
            "mean_internal_bonds_per_fragment",
        ):
            stats = summarize_numeric(row[column] for row in rows_for_split)
            for stat_name, value in stats.items():
                summary[f"{column}_{stat_name}"] = value
        rows.append(summary)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def collect_diagnostics(processed_dir, splits, max_molecules=None, top_k=20):
    molecule_rows = []
    fragments = []
    metadata = {}
    for split_name in splits:
        split_path = processed_dir / f"{split_name}.pt"
        payload = load_payload(split_path)
        metadata[split_name] = payload["metadata"]
        molecules = payload["molecule_data"]
        if max_molecules is not None:
            molecules = molecules[:max_molecules]
        for split_index, data in enumerate(molecules):
            molecule_rows.append(molecule_row(split_name, split_index, data))
            fragments.extend(fragment_rows(split_name, split_index, data))

    summary_rows = split_summary_rows(molecule_rows, fragments, top_k=top_k)
    return {
        "metadata": metadata,
        "molecules": molecule_rows,
        "fragments": fragments,
        "summary": summary_rows,
    }


def values_by_split(rows, column):
    ret = defaultdict(list)
    for row in rows:
        ret[row["split"]].append(row[column])
    return dict(ret)


def plot_bar(plt, labels, values, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.9), 4.5))
    ax.bar(labels, values, color="#386cb0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_box(plt, split_values, title, ylabel, path):
    labels = list(split_values.keys())
    values = [split_values[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.0), 4.8))
    try:
        ax.boxplot(values, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(values, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_grouped_counts(plt, rows, column, title, ylabel, path, top_k=None):
    split_names = sorted({row["split"] for row in rows})
    total_counts = Counter(str(row[column]) for row in rows)
    categories = [key for key, _ in total_counts.most_common(top_k)] if top_k else sorted(total_counts)
    x_positions = list(range(len(split_names)))
    width = 0.8 / max(1, len(categories))

    fig, ax = plt.subplots(figsize=(max(8, len(split_names) * 1.2), 4.8))
    for category_index, category in enumerate(categories):
        values = []
        for split_name in split_names:
            values.append(
                sum(1 for row in rows if row["split"] == split_name and str(row[column]) == category)
            )
        offsets = [x + (category_index - (len(categories) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=category)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(split_names, rotation=30)
    ax.legend(title=column, fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_plots(diagnostics, output_dir, top_k=20):
    plt = load_matplotlib()
    if plt is None:
        print("matplotlib is not installed; skipping PNG plots.")
        return []

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    molecule_rows = diagnostics["molecules"]
    fragment_rows_all = diagnostics["fragments"]
    summary_rows = diagnostics["summary"]
    written = []

    path = plot_dir / "molecule_counts_by_split.png"
    plot_bar(
        plt,
        [row["split"] for row in summary_rows],
        [row["num_molecules"] for row in summary_rows],
        "Molecule Count By Split",
        "molecules",
        path,
    )
    written.append(path)

    path = plot_dir / "label_distribution_by_split.png"
    plot_grouped_counts(
        plt,
        molecule_rows,
        "label",
        "Label Distribution By Split",
        "molecules",
        path,
    )
    written.append(path)

    for column, title, ylabel, filename in (
        ("num_atoms", "Atoms Per Molecule By Split", "atoms", "atoms_per_molecule_by_split.png"),
        ("num_fragments", "Fragments Per Molecule By Split", "fragments", "fragments_per_molecule_by_split.png"),
        ("num_fragment_edges", "Fragment Edges Per Molecule By Split", "fragment edges", "fragment_edges_per_molecule_by_split.png"),
        ("fragment_edge_density", "Fragment Edge Density By Split", "density", "fragment_edge_density_by_split.png"),
    ):
        path = plot_dir / filename
        plot_box(plt, values_by_split(molecule_rows, column), title, ylabel, path)
        written.append(path)

    path = plot_dir / "atoms_per_fragment_by_split.png"
    plot_box(
        plt,
        values_by_split(fragment_rows_all, "num_atoms"),
        "Atoms Per Fragment By Split",
        "atoms per fragment",
        path,
    )
    written.append(path)

    if any(row["fragment_type"] for row in fragment_rows_all):
        path = plot_dir / "fragment_type_counts_by_split.png"
        plot_grouped_counts(
            plt,
            fragment_rows_all,
            "fragment_type",
            "Fragment Type Counts By Split",
            "fragments",
            path,
        )
        written.append(path)

    smiles_counts = Counter(row["fragment_smiles"] for row in fragment_rows_all if row["fragment_smiles"])
    if smiles_counts:
        labels = [key for key, _ in smiles_counts.most_common(top_k)]
        values = [smiles_counts[key] for key in labels]
        path = plot_dir / "top_fragment_smiles.png"
        plot_bar(plt, labels, values, f"Top {min(top_k, len(labels))} Fragment SMILES", "fragments", path)
        written.append(path)

    domain_counts = Counter(str(row["environment_or_group"]) for row in molecule_rows if row["environment_or_group"] != "")
    if domain_counts:
        labels = [key for key, _ in domain_counts.most_common(top_k)]
        values = [domain_counts[key] for key in labels]
        path = plot_dir / "top_domains.png"
        plot_bar(plt, labels, values, f"Top {min(top_k, len(labels))} Domains", "molecules", path)
        written.append(path)

    return written


def default_splits(processed_dir):
    return [path.stem for path in sorted(processed_dir.glob("*.pt"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_dir", help="Directory containing processed split .pt files.")
    parser.add_argument("--splits", nargs="+", help="Split names to analyze. Defaults to all .pt files.")
    parser.add_argument("--output-dir", default="reports/fragmentation_diagnostics")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top domains/fragments to report.")
    parser.add_argument("--max-molecules", type=int, help="Optional quick-test cap per split.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV/JSON only.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    splits = args.splits or default_splits(processed_dir)
    if not splits:
        raise FileNotFoundError(f"No .pt split files found in {processed_dir}")

    diagnostics = collect_diagnostics(
        processed_dir,
        splits,
        max_molecules=args.max_molecules,
        top_k=args.top_k,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    molecule_path = output_dir / "molecule_fragmentation_stats.csv"
    fragment_path = output_dir / "fragment_stats.csv"
    summary_path = output_dir / "split_summary.csv"
    metadata_path = output_dir / "metadata.json"
    write_csv(molecule_path, diagnostics["molecules"])
    write_csv(fragment_path, diagnostics["fragments"])
    write_csv(summary_path, diagnostics["summary"])
    write_json(metadata_path, diagnostics["metadata"])

    plot_paths = [] if args.no_plots else make_plots(diagnostics, output_dir, top_k=args.top_k)

    print(f"Analyzed splits: {', '.join(splits)}")
    print(f"Wrote molecule stats: {molecule_path}")
    print(f"Wrote fragment stats: {fragment_path}")
    print(f"Wrote split summary: {summary_path}")
    print(f"Wrote metadata: {metadata_path}")
    if plot_paths:
        print(f"Wrote {len(plot_paths)} plot(s) under {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
