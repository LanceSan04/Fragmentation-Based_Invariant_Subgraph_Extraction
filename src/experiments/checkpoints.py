"""Shared helpers for restoring frozen checkpoints and checking validation parity."""

import hashlib
import math
from pathlib import Path

from src.data.dataloader import build_split_dataloaders
from src.experiments.common import set_all_seeds
from src.models.sugar_ensemble import SuGAREnsemble
from src.training.checkpoint import load_checkpoint, load_view_state_dicts
from src.training.pipeline import (
    Config,
    _evaluate_ensemble,
    build_views_from_feature_dims,
    strip_non_tensor_batch_fields,
)

PARITY_TOLERANCE = 1e-4


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_parity(observed, expected, tolerance=1e-4):
    if (
        not all(math.isfinite(v) for v in [observed, expected])
        or abs(observed - expected) > tolerance
    ):
        raise ValueError(
            f"Validation parity failed: {observed} versus {expected}; no test evaluation allowed"
        )


def restore(path, cache, device):
    payload = load_checkpoint(path, map_location="cpu")
    config = Config(
        dict(
            payload["config"],
            processed_dir=str(cache),
            device=device,
            num_workers=0,
            pin_memory=False,
            test_split=None,
            report_splits=[],
        )
    )
    views = build_views_from_feature_dims(config, payload["feature_dims"])
    load_view_state_dicts(views, payload)
    for view in views:
        view.to(device).eval()
    return config, views


def evaluate_saved(checkpoint, split, cache, device, policy, seed, prediction_path=None):
    """Restore one frozen checkpoint and load only the requested split."""
    payload = load_checkpoint(checkpoint, map_location="cpu")
    values = dict(payload["config"])
    values.update(
        processed_dir=str(cache),
        loader_split_names=[split],
        num_workers=0,
        pin_memory=False,
        test_split=None,
        report_splits=[],
    )
    config = Config(values)
    set_all_seeds(seed)
    loaders = build_split_dataloaders(cache, config)
    if split not in loaders:
        raise FileNotFoundError(cache / f"{split}.pt")
    assays = [str((getattr(d, "metadata", None) or {}).get("assay_id", ""))
              for d in loaders[split].dataset] if prediction_path is not None else None
    strip_non_tensor_batch_fields(loaders)
    views = build_views_from_feature_dims(config, payload["feature_dims"])
    load_view_state_dicts(views, payload)
    for view in views:
        view.to(device)
    ensemble = SuGAREnsemble(
        views, aggregation_mode=config.aggregation_mode, inference_policy=policy
    )
    labels, probabilities = [], []
    if prediction_path is not None:
        # Capture the very same forward passes used to compute the reported AUC.
        class Collector:
            def eval(self):
                ensemble.eval()

            def __call__(self, batch, **kwargs):
                output = ensemble(batch, **kwargs)
                labels.append(batch.y.detach().cpu().reshape(-1).numpy())
                probabilities.append(output.detach().exp().cpu().numpy()[:, 1])
                return output
        evaluator = Collector()
    else:
        evaluator = ensemble
    import torch
    with torch.no_grad():
        loss, accuracy, auc = _evaluate_ensemble(evaluator, loaders[split], device, True)
    if auc is None or not math.isfinite(auc):
        raise ValueError(f"Finite {split} AUC required for seed {seed}")
    if prediction_path is not None:
        import numpy as np
        prediction_path = Path(prediction_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        y, p = np.concatenate(labels), np.concatenate(probabilities)
        if len(y) != len(assays):
            raise ValueError("Prediction membership differs from dataset")
        np.savez_compressed(prediction_path, labels=y, probabilities=p,
                            assays=np.asarray(assays), row_index=np.arange(len(y)))
    return {"loss": loss, "acc": accuracy, "auc": auc}
