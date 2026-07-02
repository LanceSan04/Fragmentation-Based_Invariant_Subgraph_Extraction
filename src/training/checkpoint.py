"""Checkpoint helpers for FragmentCIGA/SuGAr-style training runs."""

from copy import deepcopy
from pathlib import Path

import torch


CHECKPOINT_VERSION = "0.1"


def config_to_dict(config):
    """Return a plain dictionary from a Config-like object or mapping."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    return {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_")
    }


def feature_dims_dict(atom_feat_dim, fragment_feat_dim, atom_edge_dim, fragment_edge_dim):
    """Build the feature-dimension block needed to reconstruct views."""
    return {
        "atom_feature_dim": int(atom_feat_dim),
        "fragment_feature_dim": int(fragment_feat_dim),
        "atom_edge_dim": int(atom_edge_dim),
        "fragment_edge_dim": int(fragment_edge_dim),
    }


def save_training_checkpoint(
    path,
    trainer,
    config,
    feature_dims,
    history=None,
    selected_view_indices=None,
    metrics=None,
    epoch=None,
    include_optimizers=True,
):
    """Save a full training checkpoint.

    The checkpoint is intentionally richer than a bare state_dict. It stores
    enough configuration and shape metadata to rebuild FragmentCIGA views for
    evaluation or further training.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "config": config_to_dict(config),
        "feature_dims": dict(feature_dims or {}),
        "view_state_dicts": {
            int(index): deepcopy(view.state_dict())
            for index, view in enumerate(trainer.views)
        },
        "selected_view_indices": (
            list(selected_view_indices)
            if selected_view_indices is not None
            else None
        ),
        "history": history,
        "metrics": metrics or {},
        "epoch": epoch,
    }
    if include_optimizers:
        payload["optimizer_state_dicts"] = {
            int(index): deepcopy(optimizer.state_dict())
            for index, optimizer in enumerate(trainer.optimizers)
        }

    torch.save(payload, path)
    return path


def load_checkpoint(path, map_location="cpu"):
    """Load a full or legacy checkpoint from disk."""
    return torch.load(path, map_location=map_location, weights_only=False)


def is_full_checkpoint(payload):
    """Return True when payload uses the project checkpoint schema."""
    return isinstance(payload, dict) and "view_state_dicts" in payload


def view_state_dicts_from_checkpoint(payload):
    """Return view state dicts from either full or legacy checkpoint payloads."""
    if is_full_checkpoint(payload):
        view_state_dicts = payload["view_state_dicts"]
    elif isinstance(payload, dict):
        view_state_dicts = payload
    else:
        raise ValueError("Checkpoint payload must be a dictionary")

    return {
        int(index): state_dict
        for index, state_dict in view_state_dicts.items()
    }


def load_view_state_dicts(views, payload, strict=True):
    """Load checkpoint view weights into an existing view list."""
    view_state_dicts = view_state_dicts_from_checkpoint(payload)
    missing = sorted(set(range(len(views))) - set(view_state_dicts))
    if missing:
        raise ValueError(f"Checkpoint is missing state_dicts for view(s): {missing}")

    for index, view in enumerate(views):
        view.load_state_dict(view_state_dicts[index], strict=strict)


def load_optimizer_state_dicts(trainer, payload):
    """Restore optimizer state when the checkpoint contains it."""
    if not is_full_checkpoint(payload):
        return 0
    optimizer_state_dicts = payload.get("optimizer_state_dicts") or {}
    restored = 0
    for index, optimizer in enumerate(trainer.optimizers):
        state = optimizer_state_dicts.get(index)
        if state is None:
            state = optimizer_state_dicts.get(str(index))
        if state is not None:
            optimizer.load_state_dict(state)
            restored += 1
    return restored


def checkpoint_config(payload):
    """Return the saved config dictionary, or empty dict for legacy payloads."""
    if is_full_checkpoint(payload):
        return dict(payload.get("config") or {})
    return {}


def checkpoint_feature_dims(payload):
    """Return saved feature dimensions, or empty dict for legacy payloads."""
    if is_full_checkpoint(payload):
        return dict(payload.get("feature_dims") or {})
    return {}
