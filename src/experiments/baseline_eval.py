"""Evaluate frozen Assay checkpoints once, after validation parity succeeds."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch

from src.data.dataloader import build_split_dataloaders
from src.experiments.common import set_all_seeds
from src.experiments.selection import verify_study_cache
from src.models.sugar_ensemble import SuGAREnsemble
from src.training.checkpoint import load_checkpoint, load_view_state_dicts
from train_sugar import (
    Config,
    _evaluate_ensemble,
    build_views_from_feature_dims,
    strip_non_tensor_batch_fields,
)


POLICIES = {"learned": "independent", "random": "independent", "erm": "shared"}
PARITY_TOLERANCE = 1e-4


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def stats(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def source_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sources(roots, tuning_root, cache, raw, seeds):
    """Freeze inputs and verify the study before touching the test loader."""
    verify_study_cache(cache, raw)
    for split in ("ood_val", "ood_test"):
        if not (cache / f"{split}.pt").is_file():
            raise FileNotFoundError(cache / f"{split}.pt")
    leaderboard = json.loads((tuning_root / "leaderboard.json").read_text())
    if leaderboard.get("complete") is not True:
        raise ValueError("ERM tuning leaderboard is incomplete")
    winner = leaderboard["winner"]["parameters"]
    records = {}
    manifests = {}
    for arm, root in roots.items():
        summary_path = root / "summary.json"
        summary = json.loads(summary_path.read_text())
        runs = {int(row["seed"]): row for row in summary["runs"]}
        if len(runs) != len(summary["runs"]) or set(runs) != set(seeds):
            raise ValueError(f"{arm} must have exactly the requested unique seeds")
        launch_root = root.parent
        launch = json.loads((launch_root / "launch.json").read_text())
        if launch.get("status") != "succeeded" or launch.get("exit_code") != 0:
            raise ValueError(f"{arm} source run did not succeed")
        manifest = json.loads((launch_root / "source-manifest.json").read_text())
        manifests[arm] = manifest["files"]
        records[arm] = {}
        policy = POLICIES[arm]
        for seed in seeds:
            row = runs[seed]
            if "ood_test" in json.dumps(row):
                raise ValueError(f"{arm} seed {seed} already has test results")
            config = json.loads((root / f"seed{seed}" / "config.json").read_text())
            if config["seed"] != seed or config["sampler_seed"] != seed:
                raise ValueError(f"{arm} seed mismatch in saved config")
            if Path(config["processed_dir"]).resolve() != cache or Path(config["raw_path"]).resolve() != raw:
                raise ValueError(f"{arm} seed {seed} uses another data source")
            if arm == "erm":
                if not config["erm_baseline"] or any(config[k] != v for k, v in winner.items()):
                    raise ValueError(f"ERM seed {seed} differs from the validation-selected setting")
            elif config["erm_baseline"] or config["selection_mode"] != arm:
                raise ValueError(f"{arm} seed {seed} has the wrong model mode")
            checkpoint = root / f"seed{seed}" / f"best_{policy}.pt"
            if not checkpoint.is_file() or checkpoint.is_symlink():
                raise FileNotFoundError(checkpoint)
            payload = load_checkpoint(checkpoint, map_location="cpu")
            if payload["epoch"] != row["best"][policy]["epoch"]:
                raise ValueError(f"{arm} seed {seed} checkpoint epoch differs from selection record")
            if payload["config"]["inference_policy"] != policy:
                raise ValueError(f"{arm} seed {seed} checkpoint policy differs from frozen policy")
            expected = float(row["results"][policy][policy]["ood_val"]["auc"])
            if not math.isfinite(expected):
                raise ValueError(f"{arm} seed {seed} lacks finite saved validation AUC")
            records[arm][seed] = {
                "checkpoint": checkpoint,
                "checkpoint_sha256": source_digest(checkpoint),
                "expected_ood_val_auc": expected,
                "selected_epoch": payload["epoch"],
                "config": config,
            }
    for seed in seeds:
        learned = dict(records["learned"][seed]["config"])
        random = dict(records["random"][seed]["config"])
        learned.pop("selection_mode")
        random.pop("selection_mode")
        if learned != random:
            raise ValueError(f"Learned/random settings differ beyond selection_mode for seed {seed}")
    scientific = lambda name: name.startswith(("src/", "configs/")) or name in {"main.py", "train_sugar.py"}
    reference = {name: digest for name, digest in manifests["learned"].items() if scientific(name)}
    for arm, files in manifests.items():
        if {name: digest for name, digest in files.items() if scientific(name)} != reference:
            raise ValueError(f"{arm} was trained with different scientific source files")
    return records, winner


def evaluate_saved(checkpoint, split, cache, device, policy, seed):
    """Restore one frozen checkpoint and load only the requested split."""
    payload = load_checkpoint(checkpoint, map_location="cpu")
    values = dict(payload["config"])
    values.update(processed_dir=str(cache), loader_split_names=[split], num_workers=0,
                  pin_memory=False, test_split=None, report_splits=[])
    config = Config(values)
    set_all_seeds(seed)
    loaders = build_split_dataloaders(cache, config)
    if split not in loaders:
        raise FileNotFoundError(cache / f"{split}.pt")
    strip_non_tensor_batch_fields(loaders)
    views = build_views_from_feature_dims(config, payload["feature_dims"])
    load_view_state_dicts(views, payload)
    for view in views:
        view.to(device)
    ensemble = SuGAREnsemble(views, aggregation_mode=config.aggregation_mode,
                             inference_policy=policy)
    loss, accuracy, auc = _evaluate_ensemble(ensemble, loaders[split], device, True)
    if auc is None or not math.isfinite(auc):
        raise ValueError(f"Finite {split} AUC required for seed {seed}")
    return {"loss": loss, "acc": accuracy, "auc": auc}


def compare_results(rows):
    report = {
        "seeds": [row["seed"] for row in rows],
        "primary_policy": {arm: f"{policy}/{policy}" for arm, policy in POLICIES.items()},
        "per_seed": rows,
        "summaries": {
            arm: {split: stats([row[arm][split]["auc"] for row in rows])
                  for split in ("ood_val", "ood_test")}
            for arm in POLICIES
        },
        "paired_differences": {},
        "caveat": "Assay test results were previously inspected; this comparison is exploratory.",
    }
    for name, first, second in (("learned_minus_random", "learned", "random"),
                                ("learned_minus_erm", "learned", "erm"),
                                ("random_minus_erm", "random", "erm")):
        report["paired_differences"][name] = stats(
            [row[first]["ood_test"]["auc"] - row[second]["ood_test"]["auc"] for row in rows]
        )
    return report


def run(roots, tuning_root, cache, raw, seeds, output, device="cpu", dry_run=False):
    roots = {arm: Path(roots[arm]).resolve() for arm in POLICIES}
    tuning_root, cache, raw, output = map(lambda path: Path(path).resolve(),
                                           (tuning_root, cache, raw, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    records, winner = load_sources(roots, tuning_root, cache, raw, seeds)
    plan = {
        "arms": {arm: str(root) for arm, root in roots.items()},
        "tuning_root": str(tuning_root), "erm_winner": winner,
        "seeds": seeds, "cache": str(cache), "raw_sha256": source_digest(raw),
        "policies": POLICIES, "parity_tolerance": PARITY_TOLERANCE,
        "checkpoints": {arm: {str(seed): {
            "path": str(records[arm][seed]["checkpoint"]),
            "sha256": records[arm][seed]["checkpoint_sha256"],
            "selected_epoch": records[arm][seed]["selected_epoch"],
        } for seed in seeds} for arm in POLICIES},
    }
    if dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    validation = {}
    for arm, policy in POLICIES.items():
        validation[arm] = {}
        for seed in seeds:
            record = records[arm][seed]
            observed = evaluate_saved(record["checkpoint"], "ood_val", cache, device, policy, seed)
            delta = observed["auc"] - record["expected_ood_val_auc"]
            validation[arm][str(seed)] = {
                "expected_auc": record["expected_ood_val_auc"],
                "observed": observed, "delta": delta,
                "pass": abs(delta) <= PARITY_TOLERANCE,
            }
            write(output / "validation_parity.json", validation)
            if abs(delta) > PARITY_TOLERANCE:
                raise RuntimeError(f"Validation parity failed for {arm} seed {seed}; test was not loaded")
            print(f"Validation parity passed: {arm} seed {seed}, AUC={observed['auc']:.6f}", flush=True)
    rows = [{"seed": seed} for seed in seeds]
    for arm, policy in POLICIES.items():
        for row in rows:
            seed = row["seed"]
            test = evaluate_saved(records[arm][seed]["checkpoint"], "ood_test", cache,
                                  device, policy, seed)
            row[arm] = {"ood_val": validation[arm][str(seed)]["observed"], "ood_test": test}
            write(output / "test_progress.json", rows)
            print(f"Test evaluated: {arm} seed {seed}, AUC={test['auc']:.6f}", flush=True)
    report = compare_results(rows)
    write(output / "comparison.json", report)
    print(f"COMPLETE: {output / 'comparison.json'}", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in POLICIES:
        parser.add_argument(f"--{arm}-root", required=True)
    parser.add_argument("--tuning-root", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[8101, 8102, 8103, 8104, 8105])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        parser.error("Unique nonempty seeds required")
    run({arm: getattr(args, f"{arm}_root") for arm in POLICIES}, args.tuning_root,
        args.processed_dir, args.raw_path, args.seeds, args.output_dir,
        args.device, args.dry_run)


if __name__ == "__main__":
    main()
