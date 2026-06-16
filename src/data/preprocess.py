"""Dataset preprocessing coordinator.

The load-convert-save pattern is based on Group Graph's data_loader.py.
Dual-view graph construction, metadata, splitting, and cache validation are
project-specific.
"""

import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data
from src.data.dataloader import build_split_dataloaders, validate_dataloader_batch
from src.fragmentation import build_fragmenter


def load_raw_records(config):
    """Load raw records from config-provided records or a simple data file."""
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


def process_record(record, config):
    """Convert one raw record into a validated MoleculeExample."""
    smiles_key = _config_get(config, "smiles_key", "smiles")
    label_key = _config_get(config, "label_key", "label")
    environment_key = _config_get(config, "environment_key", "environment")
    molecule_id_key = _config_get(config, "molecule_id_key", "molecule_id")

    smiles = _record_get(record, smiles_key)
    if smiles is None:
        raise ValueError(f"Record is missing SMILES key {smiles_key!r}: {record!r}")

    _debug_print(config, f"Processing molecule: {smiles}")
    label = _record_get(record, label_key, None)
    label = _coerce_label(label)
    environment_or_group = _record_get(record, environment_key, None)
    molecule_id = _record_get(record, molecule_id_key, None)
    fragmenter = build_fragmenter(_config_get(config, "fragmentation", "group_graph"))

    example = build_molecule_example(
        smiles=smiles,
        label=label,
        environment_or_group=environment_or_group,
        molecule_id=molecule_id,
        fragmenter=fragmenter,
    )
    _debug_print(
        config,
        "Built molecule "
        f"{smiles}: atoms={example.atom_x.size(0)}, "
        f"bonds={example.bond_attr.size(0)}, "
        f"fragments={example.fragment_x_static.size(0)}, "
        f"fragment_edges={example.fragment_edge_index.size(1)}",
    )
    return example


def process_split(records, config):
    """Process every record in one split."""
    examples = []
    failures = []
    strict = bool(_config_get(config, "strict", True))

    for index, record in enumerate(records):
        try:
            _debug_print(config, f"Record {index + 1}/{len(records)}")
            examples.append(process_record(record, config))
        except Exception as exc:
            if strict:
                raise
            failures.append({"index": index, "record": record, "error": str(exc)})
            _debug_print(config, f"Skipped invalid record {index}: {exc}")

    return examples, failures


def build_metadata(config):
    """Build metadata stored alongside every processed split."""
    config_payload = _to_plain_dict(config)
    config_json = json.dumps(config_payload, sort_keys=True, default=str)
    return {
        "preprocess_version": _config_get(config, "preprocess_version", "0.1.0"),
        "dataset": _config_get(config, "dataset", "unknown"),
        "fragmentation": _config_get(config, "fragmentation", "group_graph"),
        "pooling": _config_get(config, "pooling", "mean_atom_and_internal_bond"),
        "atom_feature_dim": 133,
        "bond_feature_dim": 14,
        "config_hash": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
    }


def save_processed_split(examples, metadata, output_path):
    """Serialize processed examples and metadata to a .pt file."""
    _debug_print(config_from_metadata(metadata), f"Saving {len(examples)} molecules to {output_path}")
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


def validate_processed_cache(path, config):
    """Load a processed split if it is compatible with the current config."""
    _debug_print(config, f"Validating processed cache: {path}")
    if not osp.exists(path):
        raise FileNotFoundError(path)
    if osp.getsize(path) <= 1:
        raise ValueError(f"Processed cache is a placeholder, not valid data: {path}")

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Processed cache must contain a dictionary payload")
    if "metadata" not in payload or "molecule_data" not in payload:
        raise ValueError("Processed cache is missing metadata or molecule_data")

    expected = build_metadata(config)
    actual = payload["metadata"]
    for key in ["preprocess_version", "fragmentation", "pooling", "atom_feature_dim", "bond_feature_dim"]:
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"Processed cache metadata mismatch for {key}: "
                f"{actual.get(key)!r} != {expected.get(key)!r}"
            )

    return payload


def run_preprocessing(config):
    """Run preprocessing for configured splits and save processed .pt files."""
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

    split_records = _config_get(config, "splits")
    if split_records is None:
        split_key = _config_get(config, "split_key", "split")
        records = load_raw_records(config)
        split_records = {"all": records}
        if records and isinstance(records[0], dict) and split_key in records[0]:
            split_records = {}
            for record in records:
                split_name = str(record[split_key])
                split_records.setdefault(split_name, []).append(record)

    outputs = {}
    failures = {}
    for split_name, records in split_records.items():
        _debug_print(config, f"Processing split '{split_name}' with {len(records)} records")
        examples, split_failures = process_split(records, config)
        output_path = osp.join(processed_dir, f"{split_name}.pt")
        save_processed_split(examples, metadata, output_path)
        outputs[split_name] = output_path
        failures[split_name] = split_failures
        _debug_print(
            config,
            f"Finished split '{split_name}': "
            f"saved={len(examples)}, failures={len(split_failures)}, path={output_path}",
        )

    _debug_print(config, "Finished preprocessing")
    return {"metadata": metadata, "outputs": outputs, "failures": failures}


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


def config_from_metadata(metadata):
    return {
        "debug": False,
        "dataset": metadata.get("dataset"),
        "fragmentation": metadata.get("fragmentation"),
        "pooling": metadata.get("pooling"),
    }


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
