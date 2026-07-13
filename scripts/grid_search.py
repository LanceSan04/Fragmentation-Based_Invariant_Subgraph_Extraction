#!/usr/bin/env python
r"""Optuna hyperparameter search + repeat evaluation for the SuGAr pipeline.

Two modes, both driven by ``configs/grid_search.yaml`` and both wrapping the
existing ``run_training`` from ``train_sugar``:

  search
      Run an Optuna study that samples hyperparameters from the config ``grid``
      (each list is a categorical choice). For each trial, record its
      VALIDATION-split performance to a leaderboard CSV, then write
      ``best_params.json`` for the best trial.

  eval
      Load ``best_params.json`` (or an explicit ``--params`` file), run training
      ``eval_runs`` times with different seeds, and write each run's TEST-split
      result plus the final mean/std to ``eval_results.json``.

Every training run gets a run name, and its full effective config is saved to
``{output_dir}/{mode}/runs/{run_name}.json``. Heavy per-run training artifacts
(checkpoints, run manifests, loss CSVs, diagnostics) are suppressed so a sweep
stays light.

During ``search``, pressing Ctrl+C once stops gracefully: the training run in
flight finishes, the trial is scored from the repeats completed so far, the
leaderboard/best-params snapshots are written, and the study halts (resume by
re-running the same command). Pressing Ctrl+C a second time aborts immediately.

Examples:
    .\.venv\Scripts\python.exe scripts\grid_search.py search --config configs\grid_search.yaml
    .\.venv\Scripts\python.exe scripts\grid_search.py search --config configs\grid_search.yaml --trials 20
    .\.venv\Scripts\python.exe scripts\grid_search.py eval   --config configs\grid_search.yaml --runs 5
    .\.venv\Scripts\python.exe scripts\grid_search.py eval   --config configs\grid_search.yaml --params final_testing_outputs\search\best_params.json
"""

import argparse
import csv
import json
import random
import signal
import statistics
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from train_sugar import Config, load_config, run_training


# --------------------------------------------------------------------------- #
# Spec loading + run-config assembly
# --------------------------------------------------------------------------- #
def load_spec(path):
    """Load the search spec YAML and its referenced base training config."""
    spec = load_config(path)
    base_config_path = spec.get("base_config", "configs/experiment.yaml")
    # Resolve base_config relative to the repo root so either CWD works.
    base_path = Path(base_config_path)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = load_config(str(base_path))
    spec["_base_config_path"] = str(base_path)
    return spec, base


def set_all_seeds(seed):
    """Seed every RNG the training path draws from (model init is not seeded)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_run_dict(base, overrides, combo, seed, run_name):
    """Merge base config + fixed overrides + one combo, forcing quiet outputs."""
    run_dict = dict(base)
    run_dict.update(overrides or {})
    run_dict.update(combo or {})
    # Suppress all consolidated run artifacts: empty output_root makes
    # resolve_run_directory() a no-op, and loss_log_path=None disables the CSV.
    # We still record the effective config ourselves (see save_run_config).
    run_dict["output_root"] = ""
    run_dict["run_name"] = run_name
    run_dict["save_model"] = None
    run_dict["final_metrics_path"] = None
    run_dict["loss_log_path"] = None
    run_dict["save_outputs"] = {key: False for key in (
        "checkpoint",
        "training_epoch_losses",
        "final_evaluation_metrics",
        "training_curves",
        "preprocessing_fragmentation_diagnostics",
        "subgraph_analysis",
        "effective_config",
        "run_manifest",
        "preprocess_failures",
    )}
    # Vary randomness reproducibly per run.
    run_dict["seed"] = seed
    run_dict["sampler_seed"] = seed
    return run_dict


def save_run_config(runs_dir, run_name, run_dict, combo, seed):
    """Persist the exact config options a run used, for provenance."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_name": run_name,
        "seed": seed,
        "params": combo,
        "config": run_dict,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = runs_dir / f"{run_name}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def run_once(base, overrides, combo, seed, run_name, runs_dir):
    """Name the run, save its config, train it, and return per-split metrics."""
    run_dict = build_run_dict(base, overrides, combo, seed, run_name)
    save_run_config(runs_dir, run_name, run_dict, combo, seed)
    set_all_seeds(seed)
    result = run_training(Config(run_dict))
    return result["metrics"]


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _summarize(values):
    """mean/std/n over a list that may contain None (e.g. single-class AUC)."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.mean(clean),
        "std": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "n": len(clean),
    }


def resolve_overrides(spec, args):
    """Fixed per-run overrides from the spec, with CLI epochs/device on top."""
    overrides = dict(spec.get("overrides", {}) or {})
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.device is not None:
        overrides["device"] = args.device
    return overrides


# --------------------------------------------------------------------------- #
# search mode (Optuna study)
# --------------------------------------------------------------------------- #
def run_search(spec, base, args):
    try:
        import optuna
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "optuna is required for `search`. Install it with "
            "`.\\.venv\\Scripts\\python.exe -m pip install optuna`."
        ) from exc

    grid = spec.get("grid", {}) or {}
    grid_keys = sorted(grid)
    if not grid_keys:
        raise SystemExit("`grid` is empty; nothing to search.")

    overrides = resolve_overrides(spec, args)
    repeats = args.repeats if args.repeats is not None else int(spec.get("search_repeats", 1))
    n_trials = args.trials if args.trials is not None else int(spec.get("n_trials", 40))
    base_seed = int(spec.get("base_seed", 7))
    sampler_seed = int(spec.get("sampler_seed", base_seed))
    val_split = spec.get("validation_split", "ood_val")
    metric = spec.get("target_metric", "auc")
    maximize = bool(spec.get("maximize", True))

    out_dir = Path(spec.get("output_dir", "final_testing_outputs")) / "search"
    runs_dir = out_dir / "runs"

    print(f"search: Optuna study of {n_trials} trial(s) x {repeats} repeat(s)")
    print(f"  objective: mean {val_split}[{metric}] ({'maximize' if maximize else 'minimize'})")
    print(f"  search space: { {key: grid[key] for key in grid_keys} }")
    if args.dry_run:
        print("dry-run: no training performed.")
        return

    # Graceful Ctrl+C: the first press lets the in-flight training run finish,
    # scores the trial from the repeats completed so far, and stops the study
    # after the snapshot is written -- the trial lands as COMPLETE rather than
    # FAIL, so it keeps its leaderboard row and its slot in the resume budget.
    # A second press aborts immediately (previous behavior).
    stop_requested = False

    def handle_sigint(_signum, _frame):
        nonlocal stop_requested
        if stop_requested:
            raise KeyboardInterrupt
        stop_requested = True
        print("\n[Ctrl+C] finishing the current training run, then saving and "
              "stopping. Press Ctrl+C again to abort immediately.")

    def objective(trial):
        combo = {key: trial.suggest_categorical(key, list(grid[key])) for key in grid_keys}
        scores = []
        for r in range(repeats):
            if stop_requested and scores:
                print(f"--- trial {trial.number}: stop requested; scoring with "
                      f"{len(scores)} of {repeats} repeat(s) ---")
                break
            seed = base_seed + trial.number * repeats + r
            run_name = f"search_trial{trial.number:04d}_rep{r}_seed{seed}"
            print(f"\n=== {run_name} : {combo} ===")
            metrics = run_once(base, overrides, combo, seed, run_name, runs_dir)
            scores.append((metrics.get(val_split, {}) or {}).get(metric))
        summary = _summarize(scores)
        trial.set_user_attr(f"val_{metric}_std", summary["std"])
        trial.set_user_attr(f"val_{metric}_n", summary["n"])
        if summary["mean"] is None:
            raise optuna.TrialPruned()  # no valid metric this trial
        print(f"--- trial {trial.number}: {val_split}[{metric}] mean={summary['mean']} "
              f"std={summary['std']} n={summary['n']} ---")
        return summary["mean"]

    # Persist the study to SQLite so a multi-day sweep survives crashes/reboots
    # (an in-memory study would lose every completed trial). Re-running the same
    # command resumes: load_if_exists reopens the study and we only run the
    # remaining trials. The same storage also lets extra worker processes share
    # one study for parallelism. Note: after an interruption the TPE suggestion
    # sequence is not bit-identical to one uninterrupted run, but per-trial seeds
    # (keyed off trial.number) stay stable.
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{(out_dir / 'study.db').resolve().as_posix()}"
    study_name = str(spec.get("study_name") or "sugar_search")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize" if maximize else "minimize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
    )
    existing = len(study.trials)
    remaining = max(0, n_trials - existing)
    if existing:
        print(f"  resuming study '{study_name}' from {storage_url}: "
              f"{existing} existing trial(s), running {remaining} more to reach {n_trials}")

    csv_path = out_dir / "leaderboard.csv"
    best_json_path = out_dir / "best_params.json"

    def write_snapshot(snapshot_study, _frozen_trial=None):
        """Rewrite leaderboard.csv + best_params.json from the study's current
        state. Registered as an optimize() callback so both files stay current
        after every finished trial; an interrupted sweep keeps its latest
        standings instead of only the raw study.db."""
        # Leaderboard: one row per trial with its params + objective + std/n/state.
        fieldnames = ["trial", "state"] + grid_keys + [
            f"val_{metric}_mean", f"val_{metric}_std", f"val_{metric}_n",
        ]
        completed = [t for t in snapshot_study.trials if t.value is not None]
        completed.sort(key=lambda t: t.value, reverse=maximize)
        ordered = completed + [t for t in snapshot_study.trials if t.value is None]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for trial in ordered:
                row = {"trial": trial.number, "state": trial.state.name}
                for key in grid_keys:
                    row[key] = trial.params.get(key)
                row[f"val_{metric}_mean"] = trial.value
                row[f"val_{metric}_std"] = trial.user_attrs.get(f"val_{metric}_std")
                row[f"val_{metric}_n"] = trial.user_attrs.get(f"val_{metric}_n")
                writer.writerow(row)

        if not completed:
            return  # best_trial would raise; nothing meaningful to record yet
        best = snapshot_study.best_trial
        best_params = {
            "params": dict(best.params),
            "overrides": overrides,
            "validation_split": val_split,
            "target_metric": metric,
            f"val_{metric}_mean": best.value,
            f"val_{metric}_std": best.user_attrs.get(f"val_{metric}_std"),
            "trial_number": best.number,
            "n_trials": n_trials,
            "base_config": spec.get("_base_config_path"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        best_json_path.write_text(json.dumps(best_params, indent=2, sort_keys=True),
                                  encoding="utf-8")

    def stop_when_requested(cb_study, _frozen_trial=None):
        if stop_requested:
            cb_study.stop()

    previous_sigint = signal.signal(signal.SIGINT, handle_sigint)
    try:
        study.optimize(objective, n_trials=remaining,
                       callbacks=[write_snapshot, stop_when_requested])
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
    # Covers remaining == 0 (pure resume/report run, where no callback fires).
    write_snapshot(study)
    if stop_requested:
        print(f"\nStopped early at {len(study.trials)} of {n_trials} trial(s); "
              "re-run the same command to resume.")

    print(f"\nWrote leaderboard: {csv_path}")
    print(f"Saved per-run configs under: {runs_dir}")
    if not any(t.value is not None for t in study.trials):
        print("No completed trials yet; best_params.json not written.")
        return
    best = study.best_trial
    print(f"Wrote best params: {best_json_path}")
    print(f"Best trial {best.number}: {val_split}[{metric}]={best.value} params={best.params}")


# --------------------------------------------------------------------------- #
# eval mode
# --------------------------------------------------------------------------- #
def run_eval(spec, base, args):
    out_root = Path(spec.get("output_dir", "final_testing_outputs"))
    # Precedence: CLI --params > config params_path > search output default.
    if args.params:
        params_path = Path(args.params)
    elif spec.get("params_path"):
        params_path = Path(spec["params_path"])
    else:
        params_path = out_root / "search" / "best_params.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"No params file at {params_path}. Run `search` first or pass --params PATH."
        )
    best = json.loads(params_path.read_text(encoding="utf-8"))
    combo = best.get("params", {}) or {}

    # Prefer the overrides captured at search time so eval matches selection;
    # CLI --epochs/--device win over both the spec and the captured overrides.
    overrides = dict(spec.get("overrides", {}) or {})
    overrides.update(best.get("overrides", {}) or {})
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.device is not None:
        overrides["device"] = args.device

    runs = args.runs if args.runs is not None else int(spec.get("eval_runs", 5))
    base_seed = int(spec.get("base_seed", 7))
    test_split = spec.get("test_split", "ood_test")
    report_splits = [
        split for split in (base.get("report_splits", []) or [])
        if split != test_split
    ]
    eval_splits = [test_split] + report_splits

    out_dir = out_root / "eval"
    runs_dir = out_dir / "runs"

    print(f"eval: {runs} run(s) of chosen config on splits {eval_splits}")
    print(f"  params: {combo}")
    if args.dry_run:
        print("dry-run: no training performed.")
        return

    per_run = []
    collected = {split: {"loss": [], "acc": [], "auc": []} for split in eval_splits}
    for r in range(runs):
        seed = base_seed + r
        run_name = f"eval_run{r:02d}_seed{seed}"
        print(f"\n=== {run_name} ===")
        metrics = run_once(base, overrides, combo, seed, run_name, runs_dir)
        run_record = {"run_name": run_name, "seed": seed}
        for split in eval_splits:
            split_metrics = metrics.get(split, {}) or {}
            run_record[split] = {
                "loss": split_metrics.get("loss"),
                "acc": split_metrics.get("acc"),
                "auc": split_metrics.get("auc"),
            }
            for key in ("loss", "acc", "auc"):
                collected[split][key].append(split_metrics.get(key))
        per_run.append(run_record)

    summary = {
        split: {key: _summarize(values) for key, values in metric_values.items()}
        for split, metric_values in collected.items()
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "params": combo,
        "overrides": overrides,
        "params_source": str(params_path),
        "test_split": test_split,
        "eval_splits": eval_splits,
        "runs": per_run,
        "summary": summary,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    results_path = out_dir / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== per-run test results ===")
    for record in per_run:
        test_record = record[test_split]
        print(f"  {record['run_name']}: acc={test_record['acc']} auc={test_record['auc']} "
              f"loss={test_record['loss']}")
    print(f"\n=== {test_split} summary over {runs} run(s) ===")
    for key in ("auc", "acc", "loss"):
        stats = summary[test_split][key]
        print(f"  {key}: mean={stats['mean']} std={stats['std']} n={stats['n']}")
    print(f"\nSaved per-run configs under: {runs_dir}")
    print(f"Wrote eval results: {results_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/grid_search.yaml")
    common.add_argument("--epochs", type=int, help="Override epochs for every run.")
    common.add_argument("--device", help="Override device (cuda|cpu) for every run.")
    common.add_argument("--dry-run", action="store_true",
                        help="Describe the work without training.")

    search = subparsers.add_parser("search", parents=[common],
                                   help="Optuna study; record validation metric + best params.")
    search.add_argument("--trials", type=int, help="Number of Optuna trials (overrides spec).")
    search.add_argument("--repeats", type=int, help="Runs per trial (overrides spec).")
    search.set_defaults(func=run_search)

    evaluate = subparsers.add_parser("eval", parents=[common],
                                     help="Run the chosen config N times; record test mean/std.")
    evaluate.add_argument("--runs", type=int, help="Number of eval runs (overrides spec).")
    evaluate.add_argument("--params", help="Path to a params JSON (default: search/best_params.json).")
    evaluate.set_defaults(func=run_eval)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    spec, base = load_spec(str(config_path))
    args.func(spec, base, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
