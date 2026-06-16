"""Dataset preprocessing coordinator.

The load-convert-save pattern is based on Group Graph's data_loader.py.
Dual-view graph construction, metadata, splitting, and cache validation are
project-specific.
"""

import csv
import hashlib
import json
import os
import os.path as osp

import torch

from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data


def load_raw_records(config):
    """Load raw records from config-provided records or a simple data file."""
    records = _config_get(config, "records")
    if records is not None:
        return list(records)

    raw_path = _config_get(config, "raw_path")
    if raw_path is None:
        raise ValueError("config must provide either records or raw_path")

    suffix = osp.splitext(raw_path)[1].lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(raw_path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))

    if suffix == ".json":
        with open(raw_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload.get("records", [])
        return payload

    if suffix == ".jsonl":
        with open(raw_path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

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

    label = _record_get(record, label_key, None)
    label = _coerce_label(label)
    environment_or_group = _record_get(record, environment_key, None)
    molecule_id = _record_get(record, molecule_id_key, None)

    return build_molecule_example(
        smiles=smiles,
        label=label,
        environment_or_group=environment_or_group,
        molecule_id=molecule_id,
    )


def process_split(records, config):
    """Process every record in one split."""
    examples = []
    failures = []
    strict = bool(_config_get(config, "strict", True))

    for index, record in enumerate(records):
        try:
            examples.append(process_record(record, config))
        except Exception as exc:
            if strict:
                raise
            failures.append({"index": index, "record": record, "error": str(exc)})

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
    metadata = build_metadata(config)
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
        examples, split_failures = process_split(records, config)
        output_path = osp.join(processed_dir, f"{split_name}.pt")
        save_processed_split(examples, metadata, output_path)
        outputs[split_name] = output_path
        failures[split_name] = split_failures

    return {"metadata": metadata, "outputs": outputs, "failures": failures}


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
