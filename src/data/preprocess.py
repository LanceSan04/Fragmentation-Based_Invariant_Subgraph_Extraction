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
import random

import torch
from rdkit import Chem

from src.data.atom_graph import mol_to_atom_graph
from src.data.pyg_batch import MoleculeData
from src.fragmentation.group_graph import (
    GroupGraphFragmenter,
    build_fragment_edge_features,
)
from src.models.fragment_pooler import FragmentPooler


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
    """Convert one raw record into a validated MoleculeData object."""
    smiles_key = _config_get(config, "smiles_key", "smiles")
    label_key = _config_get(config, "label_key", "label")
    environment_key = _config_get(config, "environment_key", "environment")
    molecule_id_key = _config_get(config, "molecule_id_key", "molecule_id")
    fragmentation = _config_get(config, "fragmentation", "group_graph")

    smiles = _record_get(record, smiles_key)
    if smiles is None:
        raise ValueError(f"Record is missing SMILES key {smiles_key!r}: {record!r}")

    label = _record_get(record, label_key, None)
    label = _coerce_label(label)
    environment_or_group = _record_get(record, environment_key, None)
    molecule_id = _record_get(record, molecule_id_key, None)

    return build_molecule_data(
        smiles=smiles,
        label=label,
        environment_or_group=environment_or_group,
        molecule_id=molecule_id,
        fragmentation=fragmentation,
    )


def build_molecule_data(
    smiles,
    label=None,
    environment_or_group=None,
    molecule_id=None,
    fragmentation="group_graph",
):
    """Convert SMILES into a PyG MoleculeData record with fragment provenance."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")

    atom_graph = mol_to_atom_graph(mol)
    fragmenter = _build_fragmenter(fragmentation)
    fragmentation_result = fragmenter.fragment(mol)
    fragment_edge_attr = build_fragment_edge_features(mol, fragmentation_result)

    pooler = FragmentPooler()
    fragment_x_static = pooler(
        atom_graph.atom_x,
        atom_graph.bond_index,
        atom_graph.bond_attr,
        fragmentation_result.atom_to_fragment,
        num_fragments=fragmentation_result.num_fragments,
    )

    y = None
    if label is not None:
        y = torch.as_tensor(label, dtype=torch.float32)
        if y.dim() == 0:
            y = y.view(1)

    return MoleculeData(
        x=atom_graph.atom_x,
        edge_index=atom_graph.atom_edge_index,
        edge_attr=atom_graph.atom_edge_attr,
        bond_index=atom_graph.bond_index,
        bond_attr=atom_graph.bond_attr,
        atom_ids=atom_graph.atom_ids,
        bond_ids=atom_graph.bond_ids,
        directed_edge_to_bond=atom_graph.directed_edge_to_bond,
        fragment_x_static=fragment_x_static,
        fragment_edge_index=fragmentation_result.fragment_edge_index,
        fragment_edge_attr=fragment_edge_attr,
        atom_to_fragment=fragmentation_result.atom_to_fragment,
        fragment_to_atoms=fragmentation_result.fragment_to_atoms,
        fragment_to_atom_index=build_fragment_to_atom_index(fragmentation_result.fragment_to_atoms),
        fragment_edge_to_bond=fragmentation_result.fragment_edge_to_bond,
        fragment_edge_to_bond_index=fragmentation_result.fragment_edge_to_bond_index,
        fragment_edge_to_attachment_atoms=fragmentation_result.fragment_edge_to_attachment_atoms,
        y=y,
        environment_or_group=environment_or_group,
        smiles=smiles,
        molecule_id=molecule_id,
    )


def _build_fragmenter(fragmentation_name):
    if fragmentation_name == "group_graph":
        return GroupGraphFragmenter()
    raise ValueError(f"Unsupported fragmentation method: {fragmentation_name}")


def build_fragment_to_atom_index(fragment_to_atoms):
    """Build a [fragment_id, atom_id] membership index for PyG batching."""
    pairs = []
    for fragment_id, atoms in enumerate(fragment_to_atoms):
        for atom_id in torch.as_tensor(atoms, dtype=torch.long).tolist():
            pairs.append([fragment_id, atom_id])

    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def process_split(records, config):
    """Process every record in one split."""
    examples = []
    failures = []
    strict = False

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
    """Serialize processed MoleculeData records and metadata to a .pt file."""
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "molecule_data": examples,
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


def generate_random_splits(records, train_ratio=0.7, valid_ratio=0.15, seed=None):
    """Generate train/valid/test splits by random sampling.
    
    Args:
        records: List of records to split
        train_ratio: Fraction for training (default 0.7)
        valid_ratio: Fraction for validation (default 0.15, remaining goes to test)
        seed: Random seed for reproducibility
        
    Returns:
        dict: {"train": [...], "valid": [...], "test": [...]}
    """
    if seed is not None:
        random.seed(seed)
    
    # Shuffle records
    shuffled = records.copy()
    random.shuffle(shuffled)
    
    # Calculate split indices
    n = len(records)
    train_size = int(n * train_ratio)
    valid_size = int(n * valid_ratio)
    
    splits = {
        "train": shuffled[:train_size],
        "valid": shuffled[train_size : train_size + valid_size],
        "test": shuffled[train_size + valid_size :],
    }
    
    return splits


def generate_scaffold_splits(records, smiles_key="smiles", train_ratio=0.7, valid_ratio=0.15, seed=None):
    """Generate train/valid/test splits by scaffold (requires RDKit).
    
    Molecules with similar scaffolds are kept together to avoid scaffold leakage.
    
    Args:
        records: List of records (must have smiles_key field)
        smiles_key: Field name containing SMILES
        train_ratio: Fraction for training
        valid_ratio: Fraction for validation
        seed: Random seed
        
    Returns:
        dict: {"train": [...], "valid": [...], "test": [...]}
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    if seed is not None:
        random.seed(seed)
    
    # Group records by scaffold
    scaffold_dict = {}
    for record in records:
        try:
            smiles = _record_get(record, smiles_key)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)
            scaffold_dict.setdefault(scaffold, []).append(record)
        except Exception:
            # If scaffold extraction fails, add to "unknown" scaffold
            scaffold_dict.setdefault("unknown", []).append(record)
    
    # Shuffle scaffolds
    scaffolds = list(scaffold_dict.keys())
    random.shuffle(scaffolds)
    
    # Distribute scaffolds across splits
    train_records = []
    valid_records = []
    test_records = []
    
    train_size = int(len(records) * train_ratio)
    valid_size = int(len(records) * valid_ratio)
    
    for scaffold in scaffolds:
        scaffold_records = scaffold_dict[scaffold]
        if len(train_records) < train_size:
            train_records.extend(scaffold_records)
        elif len(valid_records) < valid_size:
            valid_records.extend(scaffold_records)
        else:
            test_records.extend(scaffold_records)
    
    # Handle overflow
    while len(train_records) > train_size and len(test_records) < len(records) - train_size - valid_size:
        test_records.append(train_records.pop())
    
    return {
        "train": train_records[:train_size],
        "valid": valid_records[:valid_size],
        "test": test_records,
    }


def run_preprocessing(config):
    """Run preprocessing for configured splits and save processed .pt files."""
    metadata = build_metadata(config)
    processed_dir = _config_get(config, "processed_dir")
    if processed_dir is None:
        raise ValueError("config must provide processed_dir")

    # Step 1: Try to get pre-configured splits
    split_records = _config_get(config, "splits")
    
    if split_records is None:
        # Step 2: Try to load records and check for split_key field
        records = load_raw_records(config)
        split_key = _config_get(config, "split_key", None)
        
        if split_key and records and isinstance(records[0], dict) and split_key in records[0]:
            # Records have split_key field - use it
            split_records = {}
            for record in records:
                split_name = str(record[split_key])
                split_records.setdefault(split_name, []).append(record)
        else:
            # Step 3: Auto-generate splits if enabled
            auto_split = _config_get(config, "auto_split", False)
            if auto_split:
                split_method = _config_get(config, "split_method", "random")
                train_ratio = _config_get(config, "train_ratio", 0.7)
                valid_ratio = _config_get(config, "valid_ratio", 0.15)
                split_seed = _config_get(config, "split_seed", None)
                
                print(f"Auto-generating splits using {split_method} method...")
                print(f"  Train: {train_ratio*100:.0f}% | Valid: {valid_ratio*100:.0f}% | Test: {(1-train_ratio-valid_ratio)*100:.0f}%")
                
                if split_method == "scaffold":
                    split_records = generate_scaffold_splits(
                        records,
                        smiles_key=_config_get(config, "smiles_key", "smiles"),
                        train_ratio=train_ratio,
                        valid_ratio=valid_ratio,
                        seed=split_seed,
                    )
                else:  # default: random
                    split_records = generate_random_splits(
                        records,
                        train_ratio=train_ratio,
                        valid_ratio=valid_ratio,
                        seed=split_seed,
                    )
                
                # Print split statistics
                for split_name, split_data in split_records.items():
                    print(f"  {split_name}: {len(split_data)} molecules")
            else:
                # No splits specified - process all as one
                split_records = {"all": records}

    outputs = {}
    failures = {}
    for split_name, records in split_records.items():
        examples, split_failures = process_split(records, config)
        output_path = osp.join(processed_dir, f"{split_name}.pt")
        save_processed_split(examples, metadata, output_path)
        outputs[split_name] = output_path
        failures[split_name] = split_failures
        print(f"Saved {split_name} split: {len(examples)} molecules → {output_path}")

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


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Preprocess molecular data into PyG format with optional train/valid/test splitting")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    result = run_preprocessing(config)
    print("\nPreprocessing complete!")
    print(f"Outputs: {result['outputs']}")
    if result['failures']:
        print(f"Failures: {result['failures']}")

