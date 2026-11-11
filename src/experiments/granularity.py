"""Fixed four-arm atom/fragment control, three checkpoint rules, no automatic tuning."""

import argparse
import json
from copy import deepcopy
from pathlib import Path
import statistics
import time
from src.experiments.cross_domain import dataset_config, prepare_cache, verify_cache, cache_manifest
from src.experiments.selection import run_seed, write
from src.experiments.plans import make_plan
from src.experiments.checkpoints import evaluate_saved, require_parity, digest
from src.training.checkpoint import load_checkpoint

ARMS = ("fragment_learned", "fragment_random", "atom_learned", "atom_random")
RULES = ("best_raw", "final", "best_ma5")


def evaluate_frozen(source, out, device):
    """Evaluate the complete frozen study without training or changing selections."""
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    plan_path, frozen_path = source / "plan.json", source / "frozen_checkpoints.json"
    plan = json.loads(plan_path.read_text())
    entries = json.loads(frozen_path.read_text())
    if plan["primary_rule"] != "best_raw" or plan["inference_policy"] != "independent" or plan["test_selects"]:
        raise ValueError("Requires frozen best_raw primary rule and independent inference")
    arms = plan.get("arms", ARMS)
    if plan.get("study") in ("scorer_paths", "weighting_transfer", "atom_ranking", "good_weighting", "gating_mechanism"):
        from src.experiments.scorer_paths import PROFILES
        scorer_arms = PROFILES[plan["study"]][0]
        if set(arms) != set(scorer_arms):
            raise ValueError("Unexpected scorer-path arms")
    elif set(arms) != set(ARMS):
        raise ValueError("Unexpected granularity arms")
    expected = {(d, s, a, r) for d in plan["datasets"] for s in plan["seeds"] for a in arms for r in RULES}
    keys = [(e["dataset"], e["seed"], e["arm"], e["rule"]) for e in entries]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Frozen checkpoint list is incomplete or contains duplicates")
    for entry in entries:
        path = Path(entry["checkpoint"]).resolve()
        path.relative_to(source)
        if digest(path) != entry["checkpoint_sha256"]:
            raise ValueError("Checkpoint identity changed")
        payload = load_checkpoint(path)
        config = payload["config"]
        if plan.get("study") in ("scorer_paths", "weighting_transfer", "atom_ranking", "good_weighting", "gating_mechanism"):
            for key, value in plan["arm_settings"][entry["arm"]].items():
                if config.get(key) != value:
                    raise ValueError("Scorer-path checkpoint arm settings differ")
        if (payload["epoch"] != entry["epoch"] or config["seed"] != entry["seed"]
                or config["inference_policy"] != "independent"
                or config["checkpoint_selection_rule"] != entry["rule"]):
            raise ValueError("Checkpoint metadata differs from frozen selection")
    out.mkdir(parents=True)
    write(out / "plan.json", dict(source=str(source), source_plan_sha256=digest(plan_path),
        frozen_list_sha256=digest(frozen_path), training_runs=0, evaluated_checkpoints=len(entries),
        primary_rule="best_raw", inference_policy="independent", test_selects=False, device=device))
    # Every validation replay must pass before any test split is loaded.
    for entry in entries:
        export = {}
        if plan.get("study") in ("weighting_transfer", "atom_ranking", "good_weighting", "gating_mechanism"):
            prefix = out / "predictions" / entry["dataset"] / entry["arm"] / f"seed{entry['seed']}" / entry["rule"]
            entry["prediction_files"] = {s: str(prefix / f"{s}.npz") for s in ("ood_val", "ood_test")}
            export = dict(prediction_path=entry["prediction_files"]["ood_val"])
        val = evaluate_saved(entry["checkpoint"], "ood_val", Path(entry["cache"]), device,
                             "independent", entry["seed"], **export)
        require_parity(val["auc"], entry["selected_auc"])
        entry["metrics"] = {"ood_val": val}
    write(out / "validation_parity.json", entries)
    for index, entry in enumerate(entries, 1):
        if digest(entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise ValueError("Checkpoint identity changed before test evaluation")
        entry["metrics"]["ood_test"] = evaluate_saved(entry["checkpoint"], "ood_test",
            Path(entry["cache"]), device, "independent", entry["seed"],
            **(dict(prediction_path=entry["prediction_files"]["ood_test"]) if "prediction_files" in entry else {}))
        write(out / "evaluation_progress.json", entries)
        print(f"Evaluated {index}/{len(entries)}: {entry['dataset']} {entry['arm']} "
              f"seed{entry['seed']} {entry['rule']}", flush=True)
    report = {}
    for dataset in plan["datasets"]:
        report[dataset] = {}
        for rule in RULES:
            report[dataset][rule] = {}
            for split in ("ood_val", "ood_test"):
                rows = [dict(seed=seed, **{e["arm"]: e["metrics"][split]["auc"] for e in entries
                    if e["dataset"] == dataset and e["rule"] == rule and e["seed"] == seed})
                    for seed in plan["seeds"]]
                if plan.get("study") in ("scorer_paths", "weighting_transfer", "atom_ranking", "good_weighting", "gating_mechanism"):
                    from src.experiments.scorer_paths import PROFILES
                    pairs = PROFILES[plan["study"]][1]
                    comparisons = {}
                    for name, (first, second) in pairs.items():
                        values = [float(row[first] - row[second]) for row in rows]
                        comparisons[name] = dict(values=values, mean=statistics.mean(values),
                            positive_seeds=sum(x > 0 for x in values), n=len(values),
                            sample_std=statistics.stdev(values) if len(values) > 1 else None)
                    report[dataset][rule][split] = dict(per_seed=rows, paired_comparisons=comparisons)
                else:
                    report[dataset][rule][split] = dict(per_seed=rows, interaction=interaction(rows))
    if plan.get("study") in ("weighting_transfer", "atom_ranking", "good_weighting", "gating_mechanism"):
        from src.evaluation.paired_uncertainty import cluster_difference
        uncertainty = {}
        passes = []
        primary = plan["primary_comparison"]
        first, second = plan["paired_comparisons"][primary]
        for dataset in plan["datasets"]:
            paths = {arm: [next(e for e in entries if e["dataset"] == dataset and e["arm"] == arm
                and e["seed"] == s and e["rule"] == "best_raw")["prediction_files"]["ood_test"]
                for s in plan["seeds"]] for arm in (first, second)}
            uncertainty[dataset] = cluster_difference(paths[first], paths[second])
            contrast = report[dataset]["best_raw"]["ood_test"]["paired_comparisons"][primary]
            passes.append(contrast["mean"] >= .01 and contrast["positive_seeds"] >= 4 and contrast["n"] == 5)
        write(out / "primary_uncertainty.json", uncertainty)
        write(out / "decision.json", dict(outcome="synthetic/pilot only" if plan["synthetic"] or len(plan["seeds"]) != 5 else
            ("supported" if all(passes) else "mixed" if any(passes) else "not established"),
            per_split_pass=dict(zip(plan["datasets"], passes)), criterion=plan["screening_rule"]))
    write(out / "comparison.json", dict(datasets=report, checkpoints=entries,
        synthetic=plan["synthetic"], primary_rule="best_raw", training_runs=0, test_selects=False,
        outcome="Report all rules; test results do not select a policy"))
    return report


def interaction(rows):
    differences = [float((r["atom_learned"] - r["atom_random"]) -
                   (r["fragment_learned"] - r["fragment_random"])) for r in rows]
    return dict(values=differences, mean=statistics.mean(differences),
                sample_std=statistics.stdev(differences) if len(differences) > 1 else None,
                positive_seeds=sum(x > 0 for x in differences), n=len(differences))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=["ec50_scaffold", "ec50_size", "good_hiv_scaffold"],
                   default=["ec50_scaffold", "ec50_size"])
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--cache-root", default="data/processed/granularity_v1")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(8701, 8706)))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--stage", choices=["prepare", "train", "evaluate"], default="train")
    p.add_argument("--source-results", help="Completed granularity results for evaluation-only stage")
    p.add_argument("--selection-estimator", choices=["hard", "straight_through"], default="hard")
    p.add_argument("--retention-samples", type=int, default=100)
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.stage == "evaluate":
        if not a.source_results or a.smoke or a.dry_run:
            p.error("Evaluation requires --source-results and disallows smoke/dry-run")
        return evaluate_frozen(a.source_results, a.output_dir, a.device)
    if a.source_results:
        p.error("--source-results is only valid for evaluation")
    if a.epochs < 5 or len(set(a.seeds)) != len(a.seeds) or not a.seeds or a.retention_samples < 1:
        p.error("At least five epochs and unique seeds required")
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    seeds, epochs = ([17], 5) if a.smoke else (a.seeds, a.epochs)
    if a.smoke:
        import torch
        torch.set_num_threads(1)
    configs = {}
    for dataset in a.datasets:
        for method in ("group_graph", "atom"):
            cache = Path(a.cache_root).resolve() / dataset / method
            base = dataset_config(dataset, a.raw_root, cache, a.device)
            base.update(processed_dir=str(cache), epochs=epochs, view_fragmentations={method: 5},
                        compare_checkpoint_rules=True, report_within_assay=dataset.startswith("ec50"))
            if a.smoke:
                _, fixture = make_plan(out / dataset / method, a.device, True)
                for key in ("dataset", "raw_path", "raw_format", "splits", "label_key", "processed_dir",
                            "strict", "batch_size", "num_workers", "pin_memory", "hidden_dim", "gnn_layers"):
                    base[key] = fixture[key]
                base.update(molecule_id_key="molecule_id", metadata_keys=[], num_views=2,
                            view_fragmentations={method: 2})
            configs[dataset, method] = base
    plan = dict(datasets=a.datasets, arms=ARMS, seeds=seeds, epochs=epochs,
        stage=a.stage, synthetic=a.smoke, primary_metric="OOD validation paired granularity interaction",
        selection_estimator=a.selection_estimator, checkpoint_rules=RULES,
        retention_samples=a.retention_samples,
        primary_rule="best_raw", inference_policy="independent",
        test_selects=False, evaluate_test=a.evaluate_test,
        training_runs=0 if a.stage == "prepare" else len(a.datasets) * 4 * len(seeds),
        caveats="Same connection ratio is not the same atom budget. Atom arm has no internal-bond gates.",
        configs={f"{d}/{m}": c for (d, m), c in configs.items()})
    if a.dry_run:
        print(plan)
        return plan
    out.mkdir(parents=True)
    write(out / "plan.json", plan)
    timings = {}
    for (dataset, method), base in configs.items():
        started = time.perf_counter()
        if a.stage == "prepare" or a.smoke:
            prepare_cache(base)
        else:
            verify_cache(base)
        write(out / f"cache_{dataset}_{method}.json", cache_manifest(base))
        timings[f"{dataset}/{method}"] = time.perf_counter() - started
    write(out / "cache_timings.json", timings)
    if a.stage == "prepare":
        return plan
    frozen, resources = [], []
    for dataset in a.datasets:
        for seed in seeds:
            for arm in ARMS:
                method = "atom" if arm.startswith("atom") else "group_graph"
                base = deepcopy(configs[dataset, method])
                random = arm.endswith("random")
                base.update(selection_mode="random" if random else "learned",
                            selection_estimator="hard" if random else a.selection_estimator)
                record = run_seed(base, out / dataset / arm / f"seed{seed}", seed,
                                  policies=("independent",))
                resources.append(dict(dataset=dataset, seed=seed, arm=arm, **record["resources"]))
                write(out / "training_resources.json", resources)
                for rule in RULES:
                    selected = record["checkpoint_rules"]["records"]["independent"][rule]
                    frozen.append(dict(dataset=dataset, seed=seed, arm=arm, rule=rule,
                        checkpoint=selected["checkpoint"], checkpoint_sha256=digest(selected["checkpoint"]),
                        epoch=selected["epoch"], cache=base["processed_dir"],
                        selected_auc=selected["validation_auc_at_selection"]))
                write(out / "training_progress.json", frozen)
    # Freeze every decision and replay validation before loading ANY test split.
    write(out / "frozen_checkpoints.json", frozen)
    for entry in frozen:
        payload = load_checkpoint(entry["checkpoint"])
        if payload["epoch"] != entry["epoch"] or digest(entry["checkpoint"]) != entry["checkpoint_sha256"]:
            raise ValueError("Checkpoint identity changed")
        val = evaluate_saved(entry["checkpoint"], "ood_val", Path(entry["cache"]), a.device,
                             "independent", entry["seed"])
        require_parity(val["auc"], entry["selected_auc"])
        entry["metrics"] = {"ood_val": val}
        from src.experiments.checkpoints import restore
        from src.data.dataloader import load_processed_split, _expected_metadata
        from src.analysis.retention import measure
        config, views = restore(entry["checkpoint"], entry["cache"], "cpu")
        data = load_processed_split(Path(entry["cache"]) / "ood_val.pt",
                                    _expected_metadata(config))["molecule_data"]
        retention = measure(views, config, data, edge_ratio=config.edge_ratio,
                            count=a.retention_samples)
        path = Path(entry["checkpoint"]).with_suffix(".retention.json")
        write(path, retention)
        entry["validation_retention"] = dict(path=str(path), summary=retention["summary"],
            sample_rule=retention["sample_rule"], samples=len(retention["indices"]))
    write(out / "validation_parity.json", frozen)
    if a.evaluate_test:
        for entry in frozen:
            entry["metrics"]["ood_test"] = evaluate_saved(entry["checkpoint"], "ood_test",
                Path(entry["cache"]), a.device, "independent", entry["seed"])
    report = {}
    for dataset in a.datasets:
        report[dataset] = {}
        for rule in RULES:
            report[dataset][rule] = {}
            for split in (["ood_val", "ood_test"] if a.evaluate_test else ["ood_val"]):
                rows = [dict(seed=seed, **{e["arm"]: e["metrics"][split]["auc"] for e in frozen
                    if e["dataset"] == dataset and e["rule"] == rule and e["seed"] == seed}) for seed in seeds]
                report[dataset][rule][split] = dict(per_seed=rows, interaction=interaction(rows))
    write(out / "comparison.json", dict(datasets=report, checkpoints=frozen, synthetic=a.smoke,
        outcome="not assessed automatically; inspect predeclared criteria", test_selects=False))
    print(f"Complete: {out / 'comparison.json'}")
    return report
