"""Dataset preprocessing coordinator.

The load-convert-save pattern is based on Group Graph's data_loader.py.
Dual-view graph construction, DrugOOD split preservation, metadata, and cache
validation are project-specific.

Run the preprocessing pipeline from the repository root with:

    .\.venv\Scripts\python.exe src\data\preprocess.py --config configs\experiment.yaml

For the local DrugOOD assay data, switch ``configs/experiment.yaml`` to the
DrugOOD fields documented there, then run:

    .\.venv\Scripts\python.exe src\data\preprocess.py --config configs\experiment.yaml --validate-loaders

To also verify the saved .pt files can be loaded and batched, add:

    --validate-loaders

Preprocessing writes one .pt file per selected split under
config["processed_dir"]. For a simple ratio split this is usually:

    train.pt
    valid.pt
    test.pt

For native DrugOOD JSON, split names are preserved from the source file. The
current LBAP config writes:

    train.pt
    iid_val.pt
    ood_val.pt
    iid_test.pt
    ood_test.pt

Each .pt file contains:

    {
        "metadata": {
            "dataset": "...",
            "fragmentation": "group_graph", "brics", "recap",
                or a mixed recipe label,
            "view_fragmentations": {"group_graph": 5, "brics": 0, "recap": 0},
            "view_fragmentation_methods": ["group_graph"],
            "pooling": "mean_atom_and_internal_bond",
            "atom_feature_dim": 133,
            "bond_feature_dim": 14,
            "fragment_feature_dim": 149,
            "config_hash": "...",
            "source": {...},  # present for native split sources such as DrugOOD
        },
        "molecule_data": [MoleculeData, MoleculeData, ...],
    }

Each MoleculeData record stores aligned atom-level and fragment-level views,
including x, edge_index, edge_attr, bond_index, bond_attr, atom_ids,
bond_ids, directed_edge_to_bond, fragment_x_static, fragment_edge_index,
fragment_edge_attr, atom_to_fragment, fragment_to_atom_index,
fragment_edge_to_bond, fragment_edge_to_bond_index,
fragment_edge_to_attachment_atoms, fragment_types, fragment_smiles, smiles,
molecule_id, metadata, environment_or_group, and y. Here,
x/edge_index/edge_attr are the atom-level PyG message-passing tensors;
bond_index/bond_attr are the unique original bond table; fragment_x_static
stores pooled no-GNN fragment features; environment_or_group stores the
configured domain/environment value; metadata stores non-model provenance such
as DrugOOD assay IDs and source split names; and fragment_to_atom_index is a
sparse [2, num_atoms] mapping from fragment IDs to original atom IDs.

Example smoke molecule shape summary:

    smiles: CCOC(=O)C
    x: [6, 133]
    edge_index: [2, 10]
    bond_index: [2, 5]
    fragment_x_static: [4, 149]
    fragment_edge_index: [2, 3]
    fragment_types: ["frag", "split_frag", "split_frag", "frag"]
    fragment_smiles: ["CC", "O", "C=O", "C"]
    atom_to_fragment: [0, 0, 1, 2, 2, 3]
    fragment_edge_to_bond: [1, 2, 4]
    fragment_edge_to_attachment_atoms: [[1, 2, 3], [2, 3, 5]]
"""

import argparse
import csv
import json
import os
import os.path as osp
import random
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.atom_graph import ATOM_FDIM, BOND_FDIM
from src.data.config_hash import preprocessing_config_hash
from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data
from src.data.dataloader import build_split_dataloaders, validate_dataloader_batch
from src.fragmentation import build_fragmenter
from src.fragmentation.multi_view import (
    fragmentation_recipe_label,
    normalized_view_fragmentations,
    primary_fragmentation_method,
    unique_fragmentation_methods,
    uses_mixed_fragmentations,
)
from src.models.fragment_pooler import FRAGMENT_FEATURE_DIM


def load_raw_records(config):
    """Load flat raw records from inline config data, CSV/TSV, JSON, or JSONL.

    This helper is for unsplit or split-column datasets. Native split payloads,
    such as DrugOOD JSON files with a top-level ``split`` mapping, should go
    through ``load_raw_dataset`` so named splits are preserved.
    """
    _debug_print(config, "Loading raw records")
    records = _config_get(config, "records")
    if records is not None:
        records = list(records)
        _debug_print(config, f"Loaded {len(records)} inline records")
        return records

    raw_path = _config_get(config, "raw_path")
    if raw_path is None:
        raise ValueError("config must provide either records or raw_path")

    _debug_print(config, f"Reading raw file: {raw_path}")
    suffix = osp.splitext(raw_path)[1].lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(raw_path, newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle, delimiter=delimiter))
        _debug_print(config, f"Loaded {len(records)} records from {raw_path}")
        return records

    if suffix == ".json":
        with open(raw_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            if "split" in payload:
                raise ValueError(
                    "JSON contains native split data. Use load_raw_dataset() so split names "
                    "such as train, iid_val, ood_val, iid_test, and ood_test are preserved."
                )
            records = payload.get("records", [])
        else:
            records = payload
        _debug_print(config, f"Loaded {len(records)} records from {raw_path}")
        return records

    if suffix == ".jsonl":
        with open(raw_path, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        _debug_print(config, f"Loaded {len(records)} records from {raw_path}")
        return records

    raise ValueError(f"Unsupported raw data format: {raw_path}")


def load_raw_dataset(config):
    """Load raw data as a mapping of split name to records.

    The preferred path is an explicit ``splits`` config or a native DrugOOD JSON
    file with top-level ``cfg``, ``split``, and ``statistics`` keys. Flat files
    fall back to either a configured split column or deterministic ratio splits.
    """
    split_records = _config_get(config, "splits")
    if split_records is not None:
        return dict(split_records), {}

    raw_path = _config_get(config, "raw_path")
    raw_format = str(_config_get(config, "raw_format", "") or "").lower()
    if raw_format in {"good_hiv_scaffold_covariate_zip", "good_hiv_size_covariate_zip"}:
        from src.data.good_hiv import read_good_hiv_archive
        domain = "scaffold" if "scaffold" in raw_format else "size"
        return read_good_hiv_archive(raw_path, domain,
                                    _config_get(config, "split_names", ["train", "ood_val", "ood_test"]))
    if raw_path is not None and osp.splitext(raw_path)[1].lower() == ".json":
        with open(raw_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "split" in payload:
            if raw_format not in {"", "drugood_json"}:
                raise ValueError(f"Unsupported split JSON raw_format: {raw_format!r}")
            split_records = select_split_records(payload["split"], config)
            return split_records, {
                "raw_format": raw_format or "drugood_json",
                "source_cfg": payload.get("cfg"),
                "source_statistics": payload.get("statistics"),
                "source_split_names": list(payload["split"].keys()),
            }

    records = load_raw_records(config)
    split_key = _config_get(config, "split_key", "split")
    if records and isinstance(records[0], dict) and split_key in records[0]:
        split_records = {}
        for record in records:
            split_name = str(record[split_key])
            split_records.setdefault(split_name, []).append(record)
    else:
        split_records = split_records_by_ratio(records, config)
    return split_records, {}


def select_split_records(raw_splits, config):
    """Return configured native splits while preserving source split names."""
    if not isinstance(raw_splits, dict):
        raise ValueError("Native split payload must be a mapping of split name to records")

    requested = _config_get(config, "split_names", None)
    if requested is None:
        return {str(name): list(records) for name, records in raw_splits.items()}

    selected = {}
    for split_name in requested:
        split_name = str(split_name)
        if split_name not in raw_splits:
            available = ", ".join(sorted(str(name) for name in raw_splits))
            raise ValueError(f"Requested split {split_name!r} not found. Available: {available}")
        selected[split_name] = list(raw_splits[split_name])
    return selected


def process_record(record, config, split_name=None, split_index=None):
    """Convert one raw record into a validated dual-view MoleculeExample.

    Field names are config-driven. For DrugOOD, ``smiles_key`` normally points
    to ``smiles``, ``label_key`` to ``cls_label``, and ``environment_key`` to
    ``domain_id``. Source-only values such as ``assay_id`` and ``reg_label`` can
    be retained through ``metadata_keys`` without being fed to the model.
    """
    smiles_key = _config_get(config, "smiles_key", "smiles")
    label_key = _config_get(config, "label_key", "label")
    environment_key = _config_get(config, "environment_key", "environment")
    molecule_id_key = _config_get(config, "molecule_id_key", "molecule_id")

    smiles = _record_get(record, smiles_key)
    if smiles is None:
        raise ValueError(f"Record is missing SMILES key {smiles_key!r}: {record!r}")

    if _debug_molecule_details(config):
        _debug_print(config, f"Processing molecule: {smiles}")
    label = _record_get(record, label_key, None)
    label = _coerce_label(label)
    environment_or_group = _record_get(record, environment_key, None)
    molecule_id = _record_get(record, molecule_id_key, None)
    if molecule_id is None and bool(_config_get(config, "generate_molecule_id", True)):
        molecule_id = generated_molecule_id(config, split_name, split_index)
    metadata = record_metadata(record, config, split_name=split_name, split_index=split_index)
    fragmenter = build_fragmenter(primary_fragmentation_method(config))
    extra_fragmenters = None
    if uses_mixed_fragmentations(config):
        extra_fragmenters = {
            method: build_fragmenter(method)
            for method in unique_fragmentation_methods(config)
        }

    example = build_molecule_example(
        smiles=smiles,
        label=label,
        environment_or_group=environment_or_group,
        molecule_id=molecule_id,
        metadata=metadata,
        fragmenter=fragmenter,
        extra_fragmenters=extra_fragmenters,
        pooling=_config_get(config, "pooling", "mean_atom_and_internal_bond"),
    )
    if _debug_molecule_details(config):
        _debug_print(
            config,
            "Built molecule "
            f"{smiles}: atoms={example.atom_x.size(0)}, "
            f"bonds={example.bond_attr.size(0)}, "
            f"fragments={example.fragment_x_static.size(0)}, "
            f"fragment_edges={example.fragment_edge_index.size(1)}",
        )
    return example


def generated_molecule_id(config, split_name, split_index):
    """Create a stable fallback ID for sources without molecule identifiers."""
    dataset = _config_get(config, "dataset", "dataset")
    if split_name is None or split_index is None:
        return None
    return f"{dataset}:{split_name}:{split_index}"


def record_metadata(record, config, split_name=None, split_index=None):
    """Collect non-model provenance that should survive preprocessing."""
    metadata = {}
    if split_name is not None:
        metadata["split"] = split_name
    if split_index is not None:
        metadata["split_index"] = split_index

    for key in _config_get(config, "metadata_keys", []) or []:
        value = _record_get(record, key, None)
        if value is not None:
            metadata[key] = value
    return metadata or None


def process_split(records, config, split_name=None):
    """Process every record in one named split."""
    examples = []
    failures = []
    strict = bool(_config_get(config, "strict", True))

    for index, record in enumerate(records):
        try:
            if _should_debug_record(config, index, len(records)):
                _debug_print(config, f"Record {index + 1}/{len(records)}")
            examples.append(process_record(record, config, split_name=split_name, split_index=index))
        except Exception as exc:
            if strict:
                raise
            failures.append({"index": index, "record": record, "error": str(exc)})
            _debug_print(config, f"Skipped invalid record {index}: {exc}")

    return examples, failures


def build_metadata(config):
    """Build metadata stored alongside every processed split."""
    return {
        "preprocess_version": _config_get(config, "preprocess_version", "0.1.0"),
        "dataset": _config_get(config, "dataset", "unknown"),
        "fragmentation": fragmentation_recipe_label(config),
        "view_fragmentations": dict(normalized_view_fragmentations(config)),
        "view_fragmentation_methods": unique_fragmentation_methods(config),
        "pooling": _config_get(config, "pooling", "mean_atom_and_internal_bond"),
        "atom_feature_dim": ATOM_FDIM,
        "bond_feature_dim": BOND_FDIM,
        "fragment_feature_dim": FRAGMENT_FEATURE_DIM,
        "config_hash": preprocessing_config_hash(config),
    }


def save_processed_split(examples, metadata, output_path, config=None):
    """Serialize processed examples and metadata to a .pt file."""
    if config is not None:
        _debug_print(config, f"Saving {len(examples)} molecules to {output_path}")
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    molecule_data = [
        molecule_example_to_data(example)
        for example in examples
    ]
    torch.save(
        {
            "metadata": metadata,
            "molecule_data": molecule_data,
        },
        output_path,
    )
    return output_path


def run_preprocessing(config):
    """Run preprocessing for configured splits and save one .pt file per split."""
    from src.utils.run_outputs import resolve_processed_dir

    resolve_processed_dir(config)
    _debug_print(config, "Starting preprocessing")
    metadata = build_metadata(config)
    _debug_print(
        config,
        "Metadata: "
        f"dataset={metadata['dataset']}, "
        f"fragmentation={metadata['fragmentation']}, "
        f"pooling={metadata['pooling']}",
    )
    processed_dir = _config_get(config, "processed_dir")
    if processed_dir is None:
        raise ValueError("config must provide processed_dir")

    split_records, source_metadata = load_raw_dataset(config)
    if source_metadata:
        metadata["source"] = source_metadata

    outputs = {}
    failures = {}
    for split_name, records in split_records.items():
        _debug_print(config, f"Processing split '{split_name}' with {len(records)} records")
        examples, split_failures = process_split(records, config, split_name=split_name)
        output_path = osp.join(processed_dir, f"{split_name}.pt")
        save_processed_split(examples, metadata, output_path, config=config)
        outputs[split_name] = output_path
        failures[split_name] = split_failures
        _debug_print(
            config,
            f"Finished split '{split_name}': "
            f"saved={len(examples)}, failures={len(split_failures)}, path={output_path}",
        )

    failure_report_path = save_failure_report(failures, config, processed_dir)
    if failure_report_path is not None:
        _debug_print(config, f"Saved failure report: {failure_report_path}")

    _debug_print(config, "Finished preprocessing")
    return {
        "metadata": metadata,
        "outputs": outputs,
        "failures": failures,
        "failure_report": failure_report_path,
    }


def save_failure_report(failures, config, processed_dir):
    """Write skipped records to CSV so dataset issues can be inspected later."""
    from src.utils.run_outputs import output_enabled

    if not output_enabled(config, "preprocess_failures"):
        return None

    rows = []
    smiles_key = _config_get(config, "smiles_key", "smiles")
    molecule_id_key = _config_get(config, "molecule_id_key", "molecule_id")
    label_key = _config_get(config, "label_key", "label")
    for split_name, split_failures in failures.items():
        for failure in split_failures:
            record = failure.get("record", {})
            rows.append(
                {
                    "split": split_name,
                    "split_index": failure.get("index"),
                    "molecule_id": _record_get(record, molecule_id_key, ""),
                    "smiles": _record_get(record, smiles_key, ""),
                    "label": _record_get(record, label_key, ""),
                    "error": failure.get("error", ""),
                    "record_json": json.dumps(record, sort_keys=True, default=str),
                }
            )

    if not rows:
        return None

    report_path = _config_get(config, "failure_report_path")
    if report_path is None:
        report_path = osp.join(processed_dir, "preprocess_failures.csv")

    os.makedirs(osp.dirname(report_path), exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "split_index",
                "molecule_id",
                "smiles",
                "label",
                "error",
                "record_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return report_path


def split_records_by_ratio(records, config):
    """Create deterministic ratio splits when raw data has no native split."""
    ratios = _config_get(
        config,
        "split_ratios",
        {"train": 0.8, "valid": 0.1, "test": 0.1},
    )
    if not isinstance(ratios, dict):
        raise ValueError("split_ratios must be a mapping such as {train: 0.8, valid: 0.1, test: 0.1}")

    split_names = list(ratios.keys())
    if not split_names:
        raise ValueError("split_ratios must define at least one split")

    ratio_values = [float(ratios[name]) for name in split_names]
    if any(value < 0 for value in ratio_values):
        raise ValueError("split_ratios cannot contain negative values")
    ratio_total = sum(ratio_values)
    if ratio_total <= 0:
        raise ValueError("split_ratios must sum to a positive value")

    indexed_records = list(enumerate(records))
    seed = int(_config_get(config, "seed", 0))
    rng = random.Random(seed)
    rng.shuffle(indexed_records)
    shuffled_records = [record for _, record in indexed_records]

    total_records = len(shuffled_records)
    split_records = {}
    cursor = 0
    for index, split_name in enumerate(split_names):
        if index == len(split_names) - 1:
            end = total_records
        else:
            end = cursor + int(total_records * (ratio_values[index] / ratio_total))
        split_records[split_name] = shuffled_records[cursor:end]
        cursor = end

    _debug_print(
        config,
        "Created deterministic ratio splits: "
        + ", ".join(f"{name}={len(items)}" for name, items in split_records.items()),
    )
    return split_records


def load_config(path):
    """Load a YAML config file."""
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def _config_get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _record_get(record, key, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _coerce_label(label):
    if label is None or label == "":
        return None
    if isinstance(label, (list, tuple)):
        return [float(value) for value in label]
    try:
        return float(label)
    except (TypeError, ValueError):
        return label


def _to_plain_dict(config):
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return {"config": str(config)}


def _debug_enabled(config):
    return bool(_config_get(config, "debug", False))


def _debug_print(config, message):
    if _debug_enabled(config):
        print(f"[preprocess] {message}")


def _debug_molecule_details(config):
    return bool(_config_get(config, "debug_molecule_details", True))


def _should_debug_record(config, index, total):
    interval = int(_config_get(config, "debug_record_interval", 1))
    if interval <= 1:
        return True
    return index == 0 or index == total - 1 or (index + 1) % interval == 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run molecular preprocessing.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--validate-loaders",
        action="store_true",
        help="After preprocessing, build dataloaders and validate the first batch of each split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    result = run_preprocessing(config)

    if args.validate_loaders:
        _debug_print(config, "Building dataloaders for validation")
        loaders = build_split_dataloaders(config["processed_dir"], config)
        for split_name, loader in loaders.items():
            batch = next(iter(loader))
            validate_dataloader_batch(batch)
            _debug_print(
                config,
                f"Validated loader '{split_name}': "
                f"atoms={tuple(batch.x.shape)}, "
                f"fragments={tuple(batch.fragment_x_static.shape)}",
            )

    if not _debug_enabled(config):
        outputs = ", ".join(f"{split}={path}" for split, path in result["outputs"].items())
        print(f"Preprocessing complete: {outputs}")


if __name__ == "__main__":
    main()
