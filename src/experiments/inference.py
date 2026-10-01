"""Read-only checkpoint comparison: shared graph versus own-graph soft voting."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.experiments.metrics import metrics
from src.training.pipeline import (
    Config,
    build_views_from_feature_dims,
    strip_non_tensor_batch_fields,
)
from src.training.checkpoint import load_checkpoint, load_view_state_dicts
from src.data.dataloader import build_split_dataloaders
from src.models.sugar_ensemble import SuGAREnsemble


def checkpoint_selection_metadata(payload):
    """Describe the validation graph policy recorded by our checkpoint writers.

    Both training pipelines save their selection graph policy in the checkpoint
    config's inference_policy. Missing metadata is not evidence of shared
    selection, even though shared is the model's legacy inference default.
    """
    policy = (payload.get("config") or {}).get("inference_policy")
    if policy not in ("shared", "independent"):
        return {
            "checkpoint_selection": "unknown validation selection policy",
            "checkpoint_selection_policy": "unknown",
            "checkpoint_selection_source": None,
        }
    return {
        "checkpoint_selection": f"original {policy}-graph validation selection",
        "checkpoint_selection_policy": policy,
        "checkpoint_selection_source": "checkpoint.config.inference_policy",
    }


def compare(checkpoint, output, device="cpu", processed_dir=None, expected=None):
    payload = load_checkpoint(checkpoint)
    values = dict(payload["config"])
    if values.get("greedy_model_selection") or values.get("selection_weight_avg"):
        raise ValueError(
            "This comparison requires the fixed all-view, non-weight-averaged protocol"
        )
    splits = [values["validation_split"], values.get("test_split") or "ood_test"]
    values.update(loader_split_names=splits, num_workers=0)
    if processed_dir:
        values["processed_dir"] = str(processed_dir)
    cfg = Config(values)
    views = build_views_from_feature_dims(cfg, payload["feature_dims"])
    load_view_state_dicts(views, payload)
    if len(views) < 2 or any(v.erm_baseline for v in views):
        raise ValueError("Requires a multi-view fragment checkpoint")
    for view in views:
        view.to(device).eval()
    ensemble = SuGAREnsemble(views, aggregation_mode=cfg.get("aggregation_mode", "mean"))
    loaders = build_split_dataloaders(cfg.processed_dir, cfg)
    strip_non_tensor_batch_fields(loaders)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        seed=values["seed"],
        splits={},
        note="Exploratory fixed-checkpoint comparison. Both policies retain top-k. No retraining.",
        **checkpoint_selection_metadata(payload),
    )
    with torch.no_grad():
        for split in splits:
            labels, shared, independent = [], [], []
            for batch in loaders[split]:
                batch = batch.to(device)
                labels.append(batch.y.reshape(-1).cpu().numpy())
                shared.append(ensemble(batch).exp().cpu().numpy())
                independent.append(
                    torch.stack([v(batch).exp() for v in views]).mean(0).cpu().numpy()
                )
            y, a, b = map(np.concatenate, (labels, shared, independent))
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Nonfinite predictions")
            np.testing.assert_allclose(a.sum(1), 1, atol=1e-5)
            np.testing.assert_allclose(b.sum(1), 1, atol=1e-5)
            ma, mb = metrics(y, a), metrics(y, b)
            if ma["auc"] is None or mb["auc"] is None:
                raise ValueError("Both classes required for AUC")
            prior = (expected or {}).get(split, {}).get("auc")
            entry = dict(
                n=len(y),
                shared=ma,
                independent=mb,
                auc_delta=mb["auc"] - ma["auc"],
                saved_shared_auc=prior,
                parity_pass=abs(ma["auc"] - prior) <= 1e-4 if prior is not None else None,
            )
            report["splits"][split] = entry
            np.savez_compressed(
                output / f"{split}_predictions.npz",
                row_index=np.arange(len(y)),
                labels=y,
                shared=a,
                independent=b,
            )
            print(split, entry, flush=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if any(r["parity_pass"] is False for r in report["splits"].values()):
        raise RuntimeError(
            f"Shared AUC differs from saved result; inspect {output} before interpreting changes"
        )
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "results", help="Original evaluation output root containing eval/eval_results.json"
    )
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--processed-dir")
    p.add_argument("--output-dir")
    args = p.parse_args(argv)
    torch.set_num_threads(4)
    root = Path(args.results)
    prior = json.loads((root / "eval/eval_results.json").read_text())
    records = [r for r in prior["runs"] if args.seeds is None or r["seed"] in args.seeds]
    if not records or (args.seeds and set(args.seeds) != {r["seed"] for r in records}):
        p.error("Requested seeds missing from result file")
    out = Path(args.output_dir or f"outputs/analysis/inference/{datetime.now():%Y%m%d_%H%M%S_%f}")
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    for r in records:
        checkpoints = list(
            (root / "eval/runs/artifacts").glob(f'**/{r["run_name"]}/checkpoints/best_model.pt')
        )
        if len(checkpoints) != 1:
            raise ValueError(
                f'Expected one checkpoint for {r["run_name"]}, found {len(checkpoints)}'
            )
        reports.append(
            compare(checkpoints[0], out / r["run_name"], args.device, args.processed_dir, r)
        )
    (out / "comparison_all.json").write_text(json.dumps(reports, indent=2, allow_nan=False))
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
