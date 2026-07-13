"""Load processed molecular splits and construct PyG data loaders.

The sequence-style dataset methods and ``torch_geometric.loader.DataLoader``
usage are adapted from Group Graph's ``data_loader.py`` and task scripts.
Loading this project's dual-view records, validating metadata, and configuring
batching are project-specific. Split names are config-driven, so loaders can be
built for either simple ``train/valid/test`` datasets or native DrugOOD splits
such as ``train/iid_val/ood_val/iid_test/ood_test``.
"""

import os.path as osp
import random
from functools import partial

import torch
from torch_geometric.loader import DataLoader

from src.data.config_hash import preprocessing_config_hash
from src.data.pyg_batch import fragment_view_field
from src.fragmentation.multi_view import (
    fragmentation_recipe_label,
    unique_fragmentation_methods,
    uses_mixed_fragmentations,
)


class ProcessedMoleculeDataset:
    """Sequence wrapper for one processed molecular split.

    ``__getitem__``, ``__len__``, and ``__repr__`` retain the behavior of
    Group Graph's ``SmilesDataset``. The expected project payload contains
    PyG-ready molecule records under ``molecule_data``.
    """

    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, id):
        return self.datas["molecule_data"][id]

    def __len__(self):
        return len(self.datas["molecule_data"])

    def __repr__(self):  # pragma: no cover
        return "{}({})".format(self.__class__.__name__, len(self))


def seed_worker(worker_id, base_seed):
    """Seed a DataLoader worker using a picklable module-level function."""
    worker_seed = int(base_seed) + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def load_processed_split(path, expected_metadata=None):
    """Load and validate one processed split created by preprocess.py."""
    if not osp.exists(path):
        raise FileNotFoundError(path)
    if osp.getsize(path) <= 1:
        raise ValueError(f"Processed split is a placeholder, not valid data: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Processed split must contain a dictionary payload")
    if "metadata" not in payload or "molecule_data" not in payload:
        raise ValueError("Processed split is missing metadata or molecule_data")

    if expected_metadata is not None:
        for key, expected_value in expected_metadata.items():
            if expected_value is None:
                continue
            actual_value = payload["metadata"].get(key)
            if actual_value != expected_value:
                raise ValueError(
                    f"Metadata mismatch for {key}: {actual_value!r} != {expected_value!r}"
                )

    return payload


def build_dataset(path, expected_metadata=None):
    """Create a ProcessedMoleculeDataset from one validated split file."""
    payload = load_processed_split(path, expected_metadata=expected_metadata)
    return ProcessedMoleculeDataset(payload)


def split_dataset_for_fold(dataset, cv_folds, cv_fold, seed=0):
    """Partition one dataset into (train, val) subsets for k-fold CV.

    Fold membership is a deterministic function of (len(dataset), cv_folds,
    seed): indices are shuffled with a local RNG seeded by ``seed`` and dealt
    round-robin into ``cv_folds`` folds, so every run sharing those values
    sees identical folds regardless of training seed. Fold ``cv_fold`` becomes
    the validation subset; the remaining folds become the training subset.
    """
    cv_folds = int(cv_folds)
    cv_fold = int(cv_fold)
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")
    if not 0 <= cv_fold < cv_folds:
        raise ValueError(f"cv_fold must be in [0, {cv_folds - 1}], got {cv_fold}")
    total = len(dataset)
    if total < cv_folds:
        raise ValueError(f"Dataset has {total} example(s); cannot form {cv_folds} folds")

    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[cv_fold::cv_folds])

    records = dataset.datas["molecule_data"]
    metadata = dataset.datas.get("metadata")
    train_subset = ProcessedMoleculeDataset({
        "molecule_data": [records[i] for i in range(total) if i not in val_indices],
        "metadata": metadata,
    })
    val_subset = ProcessedMoleculeDataset({
        "molecule_data": [records[i] for i in range(total) if i in val_indices],
        "metadata": metadata,
    })
    return train_subset, val_subset


def build_dataloader(
    dataset,
    batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=False,
    pin_memory=False,
    seed=None,
    follow_batch=None,
):
    """Construct a deterministic PyG DataLoader for dual-view MoleculeData."""
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        worker_init_fn = partial(seed_worker, base_seed=int(seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        follow_batch=follow_batch or ["fragment_x_static"],
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def build_split_dataloaders(processed_dir, config):
    """Build loaders for every configured processed split file that exists."""
    expected_metadata = _expected_metadata(config)
    batch_size = _config_get(config, "batch_size", 32)
    num_workers = _config_get(config, "num_workers", 0)
    pin_memory = _config_get(config, "pin_memory", False)
    seed = _config_get(config, "seed", None)
    follow_batch = configured_follow_batch(config)

    train_split = _config_get(config, "train_split", "train")

    # Optional k-fold cross-validation: when cv_folds >= 2 and cv_fold is set,
    # the train split file is partitioned in memory and one fold is held out as
    # the validation loader (registered under cv_val_split) instead of loading
    # a separate validation file.
    cv_folds = int(_config_get(config, "cv_folds", 0) or 0)
    cv_fold = _config_get(config, "cv_fold", None)
    cv_active = cv_folds >= 2 and cv_fold is not None
    if cv_active:
        cv_fold = int(cv_fold)
    cv_val_split = str(_config_get(config, "cv_val_split", "cv_val"))

    loaders = {}
    for split_name in configured_split_names(config):
        if cv_active and split_name == cv_val_split:
            continue  # built below from the train split, not from a file
        split_path = osp.join(processed_dir, f"{split_name}.pt")
        if not osp.exists(split_path):
            continue
        dataset = build_dataset(split_path, expected_metadata=expected_metadata)
        if cv_active and split_name == train_split:
            train_subset, val_subset = split_dataset_for_fold(
                dataset, cv_folds, cv_fold,
                seed=_config_get(config, "cv_seed", 0),
            )
            loaders[train_split] = build_dataloader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=bool(_config_get(config, "drop_last", False)),
                pin_memory=pin_memory,
                seed=seed,
                follow_batch=follow_batch,
            )
            loaders[cv_val_split] = build_dataloader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=pin_memory,
                seed=seed,
                follow_batch=follow_batch,
            )
            print(f"k-fold CV: train split {split_name!r} -> fold {cv_fold + 1}/{cv_folds} "
                  f"held out as {cv_val_split!r} "
                  f"({len(train_subset)} train / {len(val_subset)} val examples)")
            continue
        loaders[split_name] = build_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == train_split),
            num_workers=num_workers,
            drop_last=bool(_config_get(config, "drop_last", False) and split_name == train_split),
            pin_memory=pin_memory,
            seed=seed,
            follow_batch=follow_batch,
        )

    if not loaders:
        raise FileNotFoundError(f"No processed split files found in {processed_dir}")

    return loaders


def configured_split_names(config):
    """Return requested split names, preserving order and removing duplicates.

    ``loader_split_names`` can override loading completely. Otherwise the
    function uses ``split_names`` from preprocessing, or falls back to
    train/validation/test plus any ``report_splits`` configured for evaluation.
    """
    configured = _config_get(config, "loader_split_names", None)
    if configured is None:
        configured = _config_get(config, "split_names", None)
    if configured is None:
        configured = [
            _config_get(config, "train_split", "train"),
            _config_get(config, "validation_split", _config_get(config, "valid_split", "valid")),
            _config_get(config, "test_split", "test"),
        ]
        configured.extend(_config_get(config, "report_splits", []) or [])

    names = []
    for split_name in configured:
        if split_name is None:
            continue
        split_name = str(split_name)
        if split_name not in names:
            names.append(split_name)
    return names


def configured_follow_batch(config):
    """Return follow_batch fields, including method-specific fragment features."""
    fields = list(_config_get(config, "follow_batch", ["fragment_x_static"]) or [])
    if uses_mixed_fragmentations(config):
        for method in unique_fragmentation_methods(config):
            field = fragment_view_field(method, "fragment_x_static")
            if field not in fields:
                fields.append(field)
    return fields


def validate_dataloader_batch(batch):
    """Check atom, bond, fragment, and provenance mappings after collation."""
    required = [
        "x",
        "edge_index",
        "bond_index",
        "bond_attr",
        "fragment_x_static",
        "fragment_edge_index",
        "atom_to_fragment",
        "fragment_to_atom_index",
    ]
    for key in required:
        if not hasattr(batch, key):
            raise ValueError(f"Batch is missing {key}")

    if batch.x.size(0) != batch.atom_to_fragment.numel():
        raise ValueError("atom_to_fragment must contain one value per batched atom")

    if batch.bond_index.numel() > 0:
        if int(batch.bond_index.max()) >= batch.x.size(0):
            raise ValueError("bond_index references an atom outside the batch")

    if batch.fragment_edge_index.numel() > 0:
        if int(batch.fragment_edge_index.max()) >= batch.fragment_x_static.size(0):
            raise ValueError("fragment_edge_index references a fragment outside the batch")

    if batch.fragment_to_atom_index.numel() > 0:
        if int(batch.fragment_to_atom_index[0].max()) >= batch.fragment_x_static.size(0):
            raise ValueError("fragment_to_atom_index references a fragment outside the batch")
        if int(batch.fragment_to_atom_index[1].max()) >= batch.x.size(0):
            raise ValueError("fragment_to_atom_index references an atom outside the batch")

    return batch


def _config_get(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _expected_metadata(config):
    keys = [
        "preprocess_version",
        "dataset",
        "pooling",
        "atom_feature_dim",
        "bond_feature_dim",
        "fragment_feature_dim",
    ]
    metadata = {}
    for key in keys:
        value = _config_get(config, key, None)
        if value is not None:
            metadata[key] = value
    metadata["fragmentation"] = fragmentation_recipe_label(config)
    metadata["config_hash"] = preprocessing_config_hash(config)
    return metadata
