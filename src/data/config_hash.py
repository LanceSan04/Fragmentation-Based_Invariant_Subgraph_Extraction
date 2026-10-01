"""Stable hashing for preprocessing-relevant configuration fields.

Preprocessing saves tensor-backed molecule records to .pt files. Those records
depend on choices like the raw data source, split rules, SMILES/label columns,
view fragmentation recipe, pooling method, feature dimensions, strict parsing mode,
and preprocessing version. If any of those choices change, an old .pt file may
no longer match the active experiment and should be rejected by the dataloader.

The config hash is the lightweight cache key used for that check:

1. ``src/data/preprocess.py`` stores ``config_hash`` in every saved split's
   metadata.
2. ``src/data/dataloader.py`` recomputes the hash from the active config.
3. Loading fails if the saved hash and active hash differ.

This file intentionally hashes only fields that can change the actual
preprocessed .pt contents. It excludes training/loader/model settings such as
batch size, hidden dimensions, learning rate, number of SuGAr base models,
selection ratio, or the DataLoader shuffle ``seed``.
"""

import hashlib
import json

PREPROCESS_CONFIG_KEYS = [
    "preprocess_version",
    "dataset",
    "raw_format",
    "raw_path",
    "records",
    "splits",
    "split_names",
    "smiles_key",
    "label_key",
    "split_key",
    "environment_key",
    "molecule_id_key",
    "generate_molecule_id",
    "metadata_keys",
    "split_ratios",
    "strict",
    "view_fragmentations",
    "pooling",
    "atom_feature_dim",
    "bond_feature_dim",
    "fragment_feature_dim",
]


def preprocessing_config_hash(config):
    """Hash only fields that can affect preprocessed .pt contents."""
    payload = preprocessing_config_payload(config)
    config_json = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def preprocessing_config_payload(config):
    """Return a plain dict with preprocessing-relevant config fields."""
    plain_config = _to_plain_dict(config)
    return {key: plain_config[key] for key in PREPROCESS_CONFIG_KEYS if key in plain_config}


def _to_plain_dict(config):
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    return {"config": str(config)}
