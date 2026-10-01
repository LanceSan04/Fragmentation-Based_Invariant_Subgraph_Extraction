#!/usr/bin/env python
"""Inspect processed molecular .pt splits created by src/data/preprocess.py.

CLI arguments:
    path
        Path to a processed .pt split file or a directory of split .pt files.

CLI options:
    --splits NAME [NAME ...]
        Split names to inspect when path is a directory.
    --index N
        Molecule index to inspect. Default: 0.
    --max-items N
        Maximum list/dict items to print. Default: 8.
    --max-chars N
        Maximum characters for scalar/list previews. Default: 220.
    --metadata-json
        Print full metadata as JSON.
    --full-lists
        Print complete list/tensor values like the old snippet.
    --all-fields
        Also print a generic summary of every stored field.

Examples:
    .\.venv\Scripts\python.exe scripts\inspect_processed_pt.py data\processed\drugood_lbap_core_ec50_assay\group_graph\ood_val.pt --index 0
    .\.venv\Scripts\python.exe scripts\inspect_processed_pt.py data\processed\drugood_lbap_core_ec50_assay\group_graph --splits train ood_val ood_test
    .\.venv\Scripts\python.exe scripts\inspect_processed_pt.py data\processed\drugood_lbap_core_ec50_assay\group_graph\ood_val.pt --metadata-json --all-fields
"""

import argparse
import json
from pathlib import Path

import torch


def summarize_tensor(value):
    summary = f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}"
    if value.numel() > 0 and value.is_floating_point():
        summary += f", min={float(value.min()):.4g}, max={float(value.max()):.4g}"
    elif value.numel() > 0 and value.dtype != torch.bool:
        summary += f", min={int(value.min())}, max={int(value.max())}"
    return summary + ")"


def shorten(value, max_chars):
    text = repr(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def summarize_value(value, max_items, max_chars):
    if torch.is_tensor(value):
        return summarize_tensor(value)
    if isinstance(value, dict):
        keys = list(value.keys())
        shown = {key: value[key] for key in keys[:max_items]}
        suffix = "" if len(keys) <= max_items else f", ... {len(keys) - max_items} more keys"
        return f"dict(len={len(value)}, sample={shorten(shown, max_chars)}{suffix})"
    if isinstance(value, (list, tuple)):
        sample = list(value[:max_items])
        suffix = "" if len(value) <= max_items else f", ... {len(value) - max_items} more"
        return (
            f"{type(value).__name__}(len={len(value)}, sample={shorten(sample, max_chars)}{suffix})"
        )
    return shorten(value, max_chars)


def preview_sequence(value, max_items):
    if not isinstance(value, (list, tuple)):
        return value
    shown = []
    for item in list(value[:max_items]):
        shown.append(preview_sequence(item, max_items))
    if len(value) > max_items:
        shown.append(f"... {len(value) - max_items} more")
    return shown


def preview_list(value, max_items, full=False):
    if torch.is_tensor(value):
        value = value.tolist()
    if full or not isinstance(value, (list, tuple)):
        return value
    return preview_sequence(value, max_items)


def load_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Expected a dict payload")
    if "metadata" not in payload or "molecule_data" not in payload:
        raise ValueError("Expected metadata and molecule_data keys")
    return payload


def data_keys(data):
    keys = data.keys() if callable(getattr(data, "keys", None)) else data.keys
    return sorted(str(key) for key in keys)


def optional_attr(data, key, default=None):
    return getattr(data, key, default)


def shape_or_none(value):
    if value is None:
        return None
    return tuple(value.shape) if torch.is_tensor(value) else None


def tensor_list_or_none(value, max_items, full=False):
    if value is None:
        return None
    if torch.is_tensor(value):
        return preview_list(value, max_items, full=full)
    return preview_list(value, max_items, full=full)


def print_sectioned_record(path, payload, index, args):
    metadata = payload["metadata"]
    molecules = payload["molecule_data"]

    print(f"\n=== {path.name} ===")
    print("path:", path)
    print("payload keys:", list(payload.keys()))
    print("metadata keys:", sorted(metadata.keys()))
    if args.metadata_json:
        print("metadata:")
        print(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    else:
        print("metadata:", summarize_value(metadata, args.max_items, args.max_chars))
    print("num molecules:", len(molecules))

    if not molecules:
        return
    if index < 0 or index >= len(molecules):
        raise IndexError(f"--index must be between 0 and {len(molecules) - 1} for {path}")

    example = molecules[index]

    print(f"\nMolecule {index}")
    print(
        "smiles:", summarize_value(optional_attr(example, "smiles"), args.max_items, args.max_chars)
    )
    print("molecule_id:", optional_attr(example, "molecule_id"))
    print(
        "label:",
        tensor_list_or_none(optional_attr(example, "y"), args.max_items, full=args.full_lists),
    )
    print("environment_or_group:", optional_attr(example, "environment_or_group"))
    print("metadata:", optional_attr(example, "metadata"))

    print("\nAtom graph")
    print("x:", shape_or_none(optional_attr(example, "x")))
    print("edge_index:", shape_or_none(optional_attr(example, "edge_index")))
    print("edge_attr:", shape_or_none(optional_attr(example, "edge_attr")))
    print("bond_index:", shape_or_none(optional_attr(example, "bond_index")))
    print("bond_attr:", shape_or_none(optional_attr(example, "bond_attr")))
    print(
        "atom_ids:",
        tensor_list_or_none(
            optional_attr(example, "atom_ids"), args.max_items, full=args.full_lists
        ),
    )
    print(
        "bond_ids:",
        tensor_list_or_none(
            optional_attr(example, "bond_ids"), args.max_items, full=args.full_lists
        ),
    )

    print("\nFragment graph")
    print("fragment_x_static:", shape_or_none(optional_attr(example, "fragment_x_static")))
    print("fragment_edge_index:", shape_or_none(optional_attr(example, "fragment_edge_index")))
    print("fragment_edge_attr:", shape_or_none(optional_attr(example, "fragment_edge_attr")))
    print(
        "fragment_types:",
        preview_list(
            optional_attr(example, "fragment_types"), args.max_items, full=args.full_lists
        ),
    )
    print(
        "fragment_smiles:",
        preview_list(
            optional_attr(example, "fragment_smiles"), args.max_items, full=args.full_lists
        ),
    )

    print("\nMappings / provenance")
    print(
        "atom_to_fragment:",
        tensor_list_or_none(
            optional_attr(example, "atom_to_fragment"), args.max_items, full=args.full_lists
        ),
    )
    print(
        "fragment_to_atom_index:",
        tensor_list_or_none(
            optional_attr(example, "fragment_to_atom_index"), args.max_items, full=args.full_lists
        ),
    )
    print(
        "fragment_edge_to_bond:",
        tensor_list_or_none(
            optional_attr(example, "fragment_edge_to_bond"), args.max_items, full=args.full_lists
        ),
    )
    print(
        "fragment_edge_to_bond_index:",
        tensor_list_or_none(
            optional_attr(example, "fragment_edge_to_bond_index"),
            args.max_items,
            full=args.full_lists,
        ),
    )
    print(
        "fragment_edge_to_attachment_atoms:",
        tensor_list_or_none(
            optional_attr(example, "fragment_edge_to_attachment_atoms"),
            args.max_items,
            full=args.full_lists,
        ),
    )

    if args.all_fields:
        print("\nAll fields")
        print(f"record type: {type(example).__name__}")
        for key in data_keys(example):
            value = getattr(example, key)
            print(f"{key}: {summarize_value(value, args.max_items, args.max_chars)}")


def paths_to_inspect(path, split_names):
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    if split_names:
        return [path / f"{split}.pt" for split in split_names]
    return sorted(path.glob("*.pt"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path", help="Path to a processed .pt split file or a directory of split .pt files."
    )
    parser.add_argument(
        "--splits", nargs="+", help="Split names to inspect when path is a directory."
    )
    parser.add_argument("--index", type=int, default=0, help="Molecule index to inspect.")
    parser.add_argument(
        "--max-items", type=int, default=8, help="Maximum list/dict items to print."
    )
    parser.add_argument(
        "--max-chars", type=int, default=220, help="Maximum characters for scalar/list previews."
    )
    parser.add_argument("--metadata-json", action="store_true", help="Print full metadata as JSON.")
    parser.add_argument(
        "--full-lists",
        action="store_true",
        help="Print complete list/tensor values like the old snippet.",
    )
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help="Also print a generic summary of every stored field.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    split_paths = paths_to_inspect(path, args.splits)
    if not split_paths:
        raise FileNotFoundError(f"No .pt files found under {path}")
    for split_path in split_paths:
        payload = load_payload(split_path)
        print_sectioned_record(split_path, payload, args.index, args)


if __name__ == "__main__":
    main()
