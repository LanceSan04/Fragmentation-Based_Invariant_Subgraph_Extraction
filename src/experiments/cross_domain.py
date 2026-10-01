"""Frozen three-arm transfer benchmark, exposed through main.py experiment.

All training and validation replays finish before any test loader is opened.
No search, retries, automatic seed expansion, or dataset-specific overrides.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
import yaml

from src.data.config_hash import preprocessing_config_hash
from src.data.dataloader import load_processed_split, _expected_metadata
from src.data.good_hiv import file_sha256
from src.data.preprocess import load_raw_dataset, run_preprocessing
from src.experiments.baseline_eval import evaluate_saved, PARITY_TOLERANCE
from src.experiments.plans import ROOT, base_config, make_plan
from src.experiments.selection import run_seed, write
from src.training.checkpoint import load_checkpoint
from train_sugar import Config


DATASETS = ("ec50_scaffold", "ec50_size", "good_hiv_scaffold", "good_hiv_size")
ARMS = ("learned", "random", "erm")
SEEDS = (8701, 8702, 8703, 8704, 8705)
POLICY = {"learned": "independent", "random": "independent", "erm": "shared"}
SPLITS = ("train", "ood_val", "ood_test")


def dataset_config(dataset, raw_root, cache_root, device):
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    cache = Path(cache_root).resolve() / dataset
    base = base_config("assay", cache, device)
    base.update(yaml.safe_load((ROOT / "configs/cross_domain_reference.yaml").read_text()))
    base.update(processed_dir=str(cache), split_names=list(SPLITS),
                loader_split_names=["train", "ood_val"], test_split=None, report_splits=[])
    if dataset.startswith("ec50_"):
        base.update(dataset=f"drugood_lbap_core_{dataset}", raw_format="drugood_json",
                    raw_path=str(Path(raw_root).resolve() / "drugood" / f"lbap_core_{dataset}.json"))
    else:
        domain = dataset.removeprefix("good_hiv_")
        base.update(dataset=f"good_hiv_{domain}_covariate",
                    raw_format=f"good_hiv_{domain}_covariate_zip",
                    raw_path=str(Path(raw_root).resolve() / "good" / "GOODHIV.zip"),
                    molecule_id_key="molecule_id", environment_key="domain_id",
                    metadata_keys=["good_source_id", "good_domain", "good_official_split",
                                   "good_split_position", "good_env_id", "domain_id"])
    return base


def arm_config(base, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    values = dict(base, erm_baseline=arm == "erm",
                  selection_mode="random" if arm == "random" else "learned",
                  inference_policy=POLICY[arm])
    if arm == "erm":
        values.update(lambda_contrast=0., lambda_diversity=0., lambda_irm=0.)
    return values


def cache_manifest(base):
    cache = Path(base["processed_dir"])
    return dict(raw_path=base["raw_path"],
                raw_sha256=file_sha256(base["raw_path"]) if base["raw_path"] else None,
                preprocessing_config_hash=preprocessing_config_hash(Config(base)),
                files={name: file_sha256(cache / f"{name}.pt") for name in SPLITS})


def prepare_cache(base):
    cache = Path(base["processed_dir"])
    if cache.exists() or cache.is_symlink():
        raise FileExistsError(f"Fresh cache required: {cache}")
    records, source = load_raw_dataset(Config(base))
    counts = {}
    for split in SPLITS:
        labels = Counter(row[base["label_key"]] for row in records[split])
        if set(labels) != {0, 1}:
            raise ValueError(f"{split}: finite binary labels and both classes required")
        counts[split] = dict(molecules=len(records[split]), negative=labels[0], positive=labels[1])
    del records
    result = run_preprocessing(Config(base))
    if any(result["failures"].values()):
        raise ValueError("Preprocessing lost molecules; study cannot continue")
    for split in SPLITS:
        payload = load_processed_split(cache / f"{split}.pt", _expected_metadata(Config(base)))
        if len(payload["molecule_data"]) != counts[split]["molecules"]:
            raise ValueError(f"Preprocessing changed {split} membership count")
    manifest = dict(cache_manifest(base), counts=counts, source=source)
    write(cache / "cross_domain_cache.json", manifest)
    return manifest


def verify_cache(base):
    cache = Path(base["processed_dir"])
    saved = json.loads((cache / "cross_domain_cache.json").read_text())
    current = cache_manifest(base)
    if any(saved.get(key) != value for key, value in current.items()):
        raise ValueError(f"Cache source/config/content changed: {cache}")
    return saved


def summary(rows, split):
    def stats(values):
        values = [float(value) for value in values]
        return dict(n=len(values), mean=statistics.mean(values),
                    sample_std=statistics.stdev(values) if len(values) > 1 else None)
    result = dict(arms={arm: stats([row[arm][split]["auc"] for row in rows]) for arm in ARMS},
                  paired_differences={})
    for other in ("random", "erm"):
        delta = [float(row["learned"][split]["auc"] - row[other][split]["auc"]) for row in rows]
        result["paired_differences"][f"learned_minus_{other}"] = dict(
            stats(delta), values=delta, positive_seeds=sum(value > 0 for value in delta))
    return result


def evaluate_frozen(entries, output, device, evaluate_test):
    """Replay every selected checkpoint before opening any test split."""
    frozen, validation = [], []
    for entry in entries:
        checkpoint = Path(entry["checkpoint"])
        payload = load_checkpoint(checkpoint, map_location="cpu")
        if payload["epoch"] != entry["selected_epoch"] or payload["config"]["seed"] != entry["seed"]:
            raise ValueError("Checkpoint identity differs from selection record")
        if payload["config"]["inference_policy"] != entry["policy"]:
            raise ValueError("Checkpoint policy differs from the frozen policy")
        frozen.append(dict(entry, checkpoint_sha256=file_sha256(checkpoint)))
    write(output / "frozen_checkpoints.json", frozen)
    for entry in frozen:
        val = evaluate_saved(entry["checkpoint"], "ood_val", entry["cache"], device,
                             entry["policy"], entry["seed"])
        delta = float(val["auc"] - entry["selected_auc"])
        passed = math.isfinite(delta) and abs(delta) <= PARITY_TOLERANCE
        validation.append(dict(dataset=entry["dataset"], arm=entry["arm"], seed=entry["seed"],
                               delta=delta, passed=passed))
        write(output / "validation_parity.json", validation)
        if not passed:
            raise RuntimeError("Validation replay failed; no test split was loaded")
        entry["metrics"] = {"ood_val": val}
    # Any identity/content failure across the whole batch prevents all test loading.
    for entry in frozen:
        if file_sha256(entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint changed during validation replay")
    if evaluate_test:
        for entry in frozen:
            entry["metrics"]["ood_test"] = evaluate_saved(
                entry["checkpoint"], "ood_test", entry["cache"], device,
                entry["policy"], entry["seed"])
            write(output / "evaluation_progress.json", frozen)
    comparisons = {}
    for dataset in dict.fromkeys(entry["dataset"] for entry in frozen):
        group = [entry for entry in frozen if entry["dataset"] == dataset]
        seeds = list(dict.fromkeys(entry["seed"] for entry in group))
        rows = [dict(seed=seed, **{e["arm"]: e["metrics"] for e in group if e["seed"] == seed})
                for seed in seeds]
        comparisons[dataset] = dict(per_seed=rows,
            summary={split: summary(rows, split) for split in (["ood_val", "ood_test"] if evaluate_test else ["ood_val"])})
    report = dict(datasets=comparisons, policies=POLICY, test_evaluated=evaluate_test,
                  note="Fixed Assay settings transferred without tuning; matched ERM is not a tuned best-possible baseline. Dataset test scores from earlier research may already have been inspected.")
    write(output / "comparison.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=["ec50_scaffold", "ec50_size"])
    parser.add_argument("--raw-root", default=str(ROOT / "data/raw"))
    parser.add_argument("--cache-root", default=str(ROOT / "data/processed/cross_domain_v1"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--stage", choices=("prepare", "pilot", "study"), default="study")
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Duplicate datasets are not allowed")
    if args.evaluate_test and args.stage != "study":
        parser.error("Test evaluation is allowed only in the fixed study")
    out = Path(args.output_dir).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    seeds = (8700,) if args.smoke or args.stage == "pilot" else SEEDS
    epochs = 2 if args.smoke else (1 if args.stage == "pilot" else 100)
    bases = {name: dataset_config(name, args.raw_root, args.cache_root, args.device) for name in args.datasets}
    for name, base in bases.items():
        base["epochs"] = epochs
        if args.smoke:
            _, fixture = make_plan(out / name, args.device, True)
            for key in ("dataset", "raw_path", "raw_format", "splits", "label_key", "processed_dir",
                        "strict", "batch_size", "num_workers", "pin_memory", "hidden_dim", "gnn_layers"):
                base[key] = fixture[key]
            base.update(molecule_id_key="molecule_id", metadata_keys=[])
    plan = dict(datasets=args.datasets, arms=ARMS, seeds=seeds, epochs=epochs, stage=args.stage,
                training_runs=0 if args.stage == "prepare" else len(bases) * len(ARMS) * len(seeds),
                synthetic=args.smoke, policies=POLICY, evaluate_test=args.evaluate_test,
                early_stopping=False, configs=bases,
                arm_rules="Random: ranking random and selected gates unity; ERM: full graph with prediction loss only, five classifiers; learned/random contrastive=0.1.",
                selection="Highest per-epoch OOD validation AUC; earliest exact tie; independent voting for fragments; full-graph voting for ERM.")
    from rdkit import rdBase
    import torch_geometric
    plan["software"] = dict(python=sys.version, torch=torch.__version__,
                            torch_geometric=torch_geometric.__version__, rdkit=rdBase.rdkitVersion)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(1 if args.smoke else 4)
    creating = args.prepare_cache or args.stage == "prepare" or args.smoke
    for base in bases.values():
        if base["raw_path"] and not Path(base["raw_path"]).is_file():
            raise FileNotFoundError(base["raw_path"])
        cache = Path(base["processed_dir"])
        if creating and cache.exists():
            raise FileExistsError(f"Fresh cache required: {cache}")
        if not creating and not (cache / "cross_domain_cache.json").is_file():
            raise FileNotFoundError(f"Prepare this study's cache first: {cache}")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    source_files = [*ROOT.glob("*.py"), *(ROOT / "src").rglob("*.py"), *(ROOT / "configs").glob("*.yaml")]
    write(out / "source_hashes.json", {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in source_files})
    caches = {}
    for name, base in bases.items():
        print(f"Cache validation/preparation: {name}", flush=True)
        caches[name] = prepare_cache(base) if creating else verify_cache(base)
        write(out / "data_manifest.json", caches)
    if args.stage == "prepare":
        write(out / "complete.json", dict(complete=True, stage="prepare"))
        return
    entries, timings = [], []
    for name, base in bases.items():
        for seed in seeds:
            for arm in ARMS:
                start = time.monotonic()
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                run_dir = out / name / arm / f"seed{seed}"
                result = run_seed(arm_config(base, arm), run_dir, seed, False, (POLICY[arm],))
                selected = result["best"][POLICY[arm]]
                replay = result["results"][POLICY[arm]][POLICY[arm]]["ood_val"]["auc"]
                if abs(replay - selected["auc"]) > PARITY_TOLERANCE:
                    raise RuntimeError("Restored validation AUC differs from selected AUC")
                entries.append(dict(dataset=name, arm=arm, seed=seed, policy=POLICY[arm],
                                    cache=base["processed_dir"], checkpoint=str(run_dir / f"best_{POLICY[arm]}.pt"),
                                    selected_epoch=selected["epoch"], selected_auc=selected["auc"]))
                timings.append(dict(dataset=name, arm=arm, seed=seed, epochs=epochs,
                                    elapsed_seconds=time.monotonic() - start,
                                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None))
                write(out / "training_progress.json", entries)
                write(out / "timings.json", timings)
    # Recheck data provenance before the validation/test evaluation phase.
    for base in bases.values():
        verify_cache(base)
    evaluate_frozen(entries, out, args.device, args.evaluate_test)
    write(out / "complete.json", dict(complete=True, training_runs=len(entries),
                                      test_evaluated=args.evaluate_test, synthetic=args.smoke))
    print(f"COMPLETE: {out / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
