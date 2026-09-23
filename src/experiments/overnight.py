#!/usr/bin/env python
"""Focused validation-only Optuna grid, followed by new-seed OOD test evaluation.

Run from the repository: python scripts/run_overnight.py --device cuda
Use --smoke for a tiny synthetic end-to-end run, or --dry-run to print the budget.
Fresh output directories are required. No automatic resume or overwriting.
"""
import argparse
from collections import Counter
from datetime import datetime
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from src.experiments import search
from src.data.preprocess import run_preprocessing
from src.training.sugar_trainer import SuGARTrainer
from train_sugar import Config, build_split_dataloaders


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


from src.experiments.plans import make_plan


def require_complete_search(spec):
    """Never run test evaluation following a partial/interrupted search."""
    import optuna
    storage = optuna.storages.RDBStorage(f"sqlite:///{Path(spec['output_dir']).as_posix()}/search/study.db")
    try:
        study = optuna.load_study(study_name=spec["study_name"], storage=storage)
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    finally:
        storage.remove_session()
        storage.engine.dispose()
    keys = sorted(spec["grid"])
    expected = set(product(*(spec["grid"][key] for key in keys)))
    if (len(complete) != len(expected) or
            {tuple(t.params.get(key) for key in keys) for t in complete} != expected or
            any(t.user_attrs.get("val_auc_best_ma_n") != spec["search_repeats"] for t in complete)):
        raise RuntimeError("Search incomplete: test evaluation was NOT started. Keep these outputs for diagnosis.")


def write_irm_comparisons(spec):
    """Paired validation changes at fixed other-loss weights and training seed."""
    runs = Path(spec["output_dir"]) / "search" / "runs"
    scores = {}
    for path in runs.glob("*.json"):
        if path.name.endswith(".result.json"):
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        result = json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))
        params = config["params"]
        key = (params["lambda_contrast"], params["lambda_diversity"],
               params["lambda_irm"], config["seed"])
        scores[key] = search._best_moving_average(result["history"]["val_auc"], 1)
    rows = []
    for contrast, diversity, irm in product(spec["grid"]["lambda_contrast"],
                                          spec["grid"]["lambda_diversity"],
                                          spec["grid"]["lambda_irm"]):
        if irm == 0:
            continue
        pairs = []
        for seed in range(spec["base_seed"], spec["base_seed"] + spec["search_repeats"]):
            baseline = scores[(contrast, diversity, 0.0, seed)]
            active = scores[(contrast, diversity, irm, seed)]
            pairs.append(dict(seed=seed, without_irm=baseline, with_irm=active,
                              difference=active - baseline))
        rows.append(dict(lambda_contrast=contrast, lambda_diversity=diversity,
                         lambda_irm=irm, paired_seeds=pairs,
                         difference_summary=search._summarize([p["difference"] for p in pairs])))
    write_json(Path(spec["output_dir"]) / "search" / "irm_comparisons.json", rows)


def check_environments(base, output):
    """Check actual processed training metadata using the trainer's bucket logic."""
    loaders = build_split_dataloaders(base["processed_dir"], Config(base))
    dataset = loaders[base["train_split"]].dataset
    values = [SuGARTrainer._extract_env_value(data, base["env_split_key"]) for data in dataset]
    if not values or any(value is None for value in values):
        raise RuntimeError("Missing training environment metadata; refusing random IRM fallback")
    buckets = SuGARTrainer._assign_env_buckets(values, base["num_envs"])
    counts = Counter(buckets[value] for value in values)
    if len(counts) != base["num_envs"]:
        raise RuntimeError("Training domains do not populate all IRM buckets")
    write_json(output / "environment_summary.json", dict(
        split=base["train_split"], strategy=base["env_bucket_strategy"],
        unique_domains=len(set(values)), molecules_per_bucket=dict(counts)))


def execute(output, device="cuda", smoke=False, dry_run=False):
    output = Path(output).resolve()
    spec, base = make_plan(output, device, smoke)
    count = spec["n_trials"] * spec["search_repeats"] + spec["eval_runs"]
    print(f"Output: {output}\nBudget: {count} training runs x {spec['overrides']['epochs']} epochs")
    print("Loss grid:", spec["grid"])
    print("Five views; full factorial loss search; best individual validation epoch.")
    print("Search seeds 101, 102. Final evaluation starts at seed 1001; all test scores reported.")
    if dry_run:
        return
    import optuna
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not smoke and not Path(base["raw_path"]).is_file():
        raise FileNotFoundError(base["raw_path"])
    output.mkdir(parents=True, exist_ok=False)
    if smoke:
        torch.set_num_threads(1)
    source_paths = list(ROOT.glob("*.py")) + list((ROOT / "src").rglob("*.py"))
    source_paths += list((ROOT / "scripts").glob("*.py")) + list((ROOT / "configs").glob("*.yaml"))
    source_paths.append(ROOT / "requirements.txt")
    write_json(output / "provenance.json", dict(
        python=sys.version, torch=torch.__version__, optuna=optuna.__version__,
        cuda=torch.version.cuda, synthetic=smoke,
        source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(source_paths)}))
    (output / "base_config.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    spec["_base_config_path"] = str(output / "base_config.yaml")
    write_json(output / "search_spec.json", spec)
    print("Preprocessing into a fresh cache, including the held-out test split...")
    preprocessed = run_preprocessing(Config(base))
    write_json(output / "preprocessing_failures.json", preprocessed.get("failures", {}))
    check_environments(base, output)
    args = SimpleNamespace(epochs=None, device=device, dry_run=False, trials=None,
                           repeats=None, runs=None, params=None)
    search.run_search(spec, base, args)
    require_complete_search(spec)
    write_irm_comparisons(spec)
    print("Search complete. Locking winner and starting independent-seed test evaluation.")
    search.run_eval(spec, base, args)
    write_json(output / "COMPLETE.json", {"synthetic": smoke, "training_runs": count,
                                         "results": "eval/eval_results.json"})
    print(f"SUCCESS: final ROC-AUC is in {output / 'eval/eval_results.json'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", help="Must not already exist")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    prefix = "outputs/smoke/overnight" if args.smoke else "outputs/assay/overnight"
    destination = args.output_dir or f"{prefix}/{datetime.now():%Y%m%d_%H%M%S}"
    execute(destination, args.device, args.smoke, args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
