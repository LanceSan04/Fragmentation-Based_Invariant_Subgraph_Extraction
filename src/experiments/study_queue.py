"""Bounded sequential native-pipeline study with validation-only decisions.

Resume is between successful jobs, never an automatic restart of interrupted
training. Upstream SuGAr is a separate pending prerequisite, not impersonated
by our atom-only or fragment pipeline. All subprocesses run without a shell.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]


class ResourceFailure(RuntimeError):
    """Stop the whole queue rather than moving on under resource/source failure."""


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity():
    return fingerprint({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for folder in ("src", "configs") for p in sorted((ROOT / folder).rglob("*"))
                        if p.is_file() and p.suffix in (".py", ".yaml", ".json")})


def validate(spec):
    if spec.get("schema") != 1 or spec.get("primary_rule") != "best_raw" or spec.get("inference_policy") != "independent":
        raise ValueError("Queue requires schema 1, best_raw and independent inference")
    if spec["epochs"] < 5 or spec["pilot_epochs"] < 1:
        raise ValueError("Invalid epoch budgets")
    if spec["feature_modes"] != ["static_pool", "internal_gnn"] or "off" not in spec["diversity"]:
        raise ValueError("Both feature references and diversity off are required")
    seeds = []
    for key in ("screen_seeds", "tune_seeds", "confirmation_seeds"):
        values = spec[key]
        if not values or any(not isinstance(s, int) for s in values):
            raise ValueError("Nonempty integer seeds required")
        seeds += values
    if len(set(seeds)) != len(seeds):
        raise ValueError("Screen/tune/confirmation seeds must be unique and disjoint")
    if spec["controls"] != ["random_learned_gates", "erm"]:
        raise ValueError("Expected predefined controls")
    from src.experiments.cross_domain import DATASETS
    if not spec["confirmation_datasets"] or not set(spec["confirmation_datasets"]) <= set(DATASETS):
        raise ValueError("Invalid confirmation datasets")
    if len(set(spec["confirmation_datasets"])) != len(spec["confirmation_datasets"]):
        raise ValueError("Duplicate datasets")
    for coefficient in spec["contrast_coefficients"]:
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("Contrast coefficients must be finite and nonnegative")
    if len(set(spec["contrast_coefficients"])) != len(spec["contrast_coefficients"]) or not spec["contrast_coefficients"]:
        raise ValueError("Unique contrast coefficients required")
    for name, values in spec["diversity"].items():
        if set(values) != {"lambda_diversity", "diversity_delay_epochs", "diversity_ramp_epochs"}:
            raise ValueError("Diversity presets cannot override unrelated model settings")
        if not math.isfinite(values["lambda_diversity"]) or values["lambda_diversity"] < 0:
            raise ValueError("Invalid diversity coefficient")
        if any(not isinstance(values[k], int) or values[k] < 0 for k in ("diversity_delay_epochs", "diversity_ramp_epochs")):
            raise ValueError("Invalid diversity schedule")
    if spec["diversity"]["off"]["lambda_diversity"] != 0:
        raise ValueError("Off must disable diversity")
    counts = dict(pilot=2,
        screen=2 * len(spec["diversity"]) * len(spec["screen_seeds"]),
        tune=2 * len(spec["contrast_coefficients"]) * len(spec["tune_seeds"]),
        confirm=4 * len(spec["confirmation_datasets"]) * len(spec["confirmation_seeds"]))
    if sum(counts.values()) > spec["max_training_runs"]:
        raise ValueError("Manifest exceeds its training-run budget")
    return counts


def choose_diversity(spec, records):
    """All candidates required; no selecting from a favorable incomplete batch."""
    choices = {}
    for feature in spec["feature_modes"]:
        off = [records[feature, "off", s] for s in spec["screen_seeds"]]
        candidates = []
        for index, preset in enumerate(spec["diversity"]):
            values = [records[feature, preset, s] for s in spec["screen_seeds"]]
            delta = [a - b for a, b in zip(values, off)]
            if preset == "off" or (statistics.mean(delta) >= spec["diversity_min_mean_gain"]
                    and sum(x > 0 for x in delta) >= spec["diversity_min_seed_wins"]):
                candidates.append((statistics.mean(values), -index, preset))
        choices[feature] = max(candidates)[2]
    return choices


def choose_contrast(spec, records):
    choices = {}
    for feature in spec["feature_modes"]:
        means = {c: statistics.mean(records[feature, c, s] for s in spec["tune_seeds"])
                 for c in spec["contrast_coefficients"]}
        best = max(means.values())
        choices[feature] = min(c for c, value in means.items()
                               if best - value <= spec["contrast_tie_tolerance"])
    return choices


@contextmanager
def queue_lock(path):
    # Advisory only; the remote launcher additionally owns the account-wide lock.
    with Path(path).open("a+b") as stream:
        stream.seek(0); stream.write(b"0"); stream.flush(); stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def task_worker(task_path):
    task = read(task_path)
    import torch
    torch.set_num_threads(1)
    from src.experiments.selection import run_seed
    if task["action"] == "train":
        run_seed(task["config"], Path(task["directory"]), task["seed"], False, ("independent",))
        if task["config"].get("diagnostic_molecules", 0) and not task["config"].get("erm_baseline", False):
            view_diagnostics(task)
    elif task["action"] == "evaluate":
        from src.experiments.checkpoints import evaluate_saved, require_parity, digest
        # Replay every rule before opening any test loader for this trajectory.
        train = Path(task["training_directory"])
        record = read(train / "results.json")["checkpoint_rules"]["records"]["independent"]
        result = {}
        for rule, item in record.items():
            path = Path(item["checkpoint"])
            observed = evaluate_saved(path, "ood_val", task["cache"], task["device"], "independent", task["seed"])
            require_parity(observed["auc"], item["validation_auc_at_selection"])
            result[rule] = dict(checkpoint=str(path), sha256=digest(path), ood_val=observed)
        for rule, item in result.items():
            if digest(item["checkpoint"]) != item["sha256"]:
                raise ValueError("Checkpoint changed during replay")
            item["ood_test"] = evaluate_saved(item["checkpoint"], "ood_test", task["cache"],
                task["device"], "independent", task["seed"],
                Path(task["directory"]) / (rule + "_predictions.npz"))
        write(Path(task["directory"]) / "results.json", result)
    else:
        raise ValueError("Unknown task action")


def view_diagnostics(task):
    """Describe the same frozen best checkpoint on validation molecules only."""
    from itertools import combinations
    import numpy as np
    import torch
    from torch_geometric.data import Batch
    from src.experiments.checkpoints import restore
    from src.data.dataloader import load_processed_split
    from src.analysis.view_overlap import overlap
    directory = Path(task["directory"])
    config, views = restore(directory / "best_independent.pt", task["config"]["processed_dir"], task["config"]["device"])
    data = load_processed_split(Path(config.processed_dir) / "ood_val.pt")["molecule_data"]
    order = sorted(range(len(data)), key=lambda i: (data[i].fragment_edge_index.size(1), i))
    indices = [order[i] for i in np.linspace(0, len(order)-1, min(config.diagnostic_molecules, len(data)), dtype=int)]
    rows, similarities, probabilities = [], [], []
    with torch.no_grad():
        for index in indices:
            record = data[index].clone()
            for key in ("fragment_to_atoms", "smiles", "molecule_id", "metadata"):
                if key in record:
                    del record[key]
            batch = Batch.from_data_list([record], follow_batch=config.follow_batch).to(config.device)
            atoms, bonds, preds = [], [], []
            for view_id, view in enumerate(views):
                prediction, info = view(batch, return_intermediates=True)
                a, b = info["selected_atom_ids"].tolist(), info["selected_bond_ids"].tolist()
                atoms.append(a); bonds.append(b); preds.append(float(prediction.exp()[0,1]))
                rows.append(dict(index=index, view=view_id, atoms=a, bonds=b,
                                 atom_fraction=len(a)/record.x.size(0),
                                 bond_fraction=len(b)/record.bond_index.size(1) if record.bond_index.size(1) else None))
            probabilities.append(preds)
            for a,b in combinations(range(len(views)),2):
                similarities.append(dict(index=index, views=[a,b], atom_jaccard=overlap(atoms[a],atoms[b]),
                                         bond_jaccard=overlap(bonds[a],bonds[b])))
    correlation = []
    p = np.asarray(probabilities)
    for a,b in combinations(range(len(views)),2):
        valid = len(p)>1 and p[:,a].std()>0 and p[:,b].std()>0
        correlation.append(dict(views=[a,b], pearson=float(np.corrcoef(p[:,a],p[:,b])[0,1]) if valid else None))
    write(directory / "view_diagnostics.json", dict(split="ood_val", indices=indices, rows=rows,
          overlap=similarities, prediction_correlations=correlation,
          note="Descriptive size-stratified validation subset, not a benchmark AUC or selection criterion"))


def run_queue(spec, out, raw_root, cache_root, device, smoke=False, resume=False, stage_limit="all"):
    from src.experiments.cross_domain import dataset_config, prepare_cache, verify_cache
    from src.experiments.plans import make_plan
    from src.experiments.checkpoints import digest
    counts = validate(spec)
    identity = dict(spec=spec, source_sha256=source_identity(), device=device, synthetic=smoke,
                    raw_root=str(Path(raw_root).resolve()), cache_root=str(Path(cache_root).resolve()))
    out = Path(out).resolve()
    if out.exists() and not resume:
        raise FileExistsError("Fresh output required; use --resume only for the same manifest/source")
    out.mkdir(parents=True, exist_ok=True)
    with queue_lock(out / "queue.lock"):
        state_path = out / "status.json"
        if state_path.exists():
            state = read(state_path)
            if state["identity"] != identity:
                raise ValueError("Source, manifest, paths or device changed; use a fresh queue")
        else:
            state = dict(identity=identity, counts=counts, jobs={}, decisions={},
                         upstream_status=spec["upstream_status"], test_selects=False)
            write(state_path, state)
        cache_configs = {}

        def base_for(dataset):
            if dataset in cache_configs:
                return deepcopy(cache_configs[dataset])
            cache = Path(cache_root).resolve() / dataset
            cfg = dataset_config(dataset, raw_root, cache_root, device)
            if smoke:
                _, fixture = make_plan(out / "fixtures" / dataset, device, True)
                for key in ("dataset", "raw_path", "raw_format", "splits", "label_key", "strict", "batch_size",
                            "num_workers", "pin_memory", "hidden_dim", "gnn_layers"):
                    cfg[key] = fixture[key]
                cfg.update(molecule_id_key="molecule_id", metadata_keys=[], num_views=2,
                           view_fragmentations={"group_graph": 2})
            cfg.update(processed_dir=str(cache), epochs=spec["epochs"],
                       fragment_feature_mode="static_pool", fragment_encoder_layers=2,
                       compare_checkpoint_rules=True, report_within_assay=not dataset.startswith("good_"),
                       log_gradient_influence=True)
            if cache.exists():
                verify_cache(cfg)
            else:
                prepare_cache(cfg)
            cache_configs[dataset] = deepcopy(cfg)
            return cfg

        def job(name, task):
            if source_identity() != identity["source_sha256"]:
                raise ResourceFailure("Source changed during queue; no further jobs launched")
            directory = out / "jobs" / name
            task = dict(task, directory=str(directory))
            token = fingerprint(task)
            prior = state["jobs"].get(name)
            if prior:
                if prior["task_sha256"] != token:
                    raise ValueError("Task specification changed")
                if prior["status"] != "completed":
                    raise RuntimeError(f"{name}: {prior['status']}; no automatic retry of partial/failed work")
                for relative, expected in prior["artifacts"].items():
                    if digest(directory / relative) != expected:
                        raise ValueError("Completed artifact changed")
                return read(directory / "results.json")
            if shutil.disk_usage(out).free < spec["minimum_free_disk_gib"] * 1024 ** 3:
                raise ResourceFailure("Insufficient free disk; no new job launched")
            if task["action"] == "train" and sum(j["action"] == "train" for j in state["jobs"].values()) >= spec["max_training_runs"]:
                raise RuntimeError("Approved manifest run ceiling reached")
            task_file = out / "tasks" / (name + ".json")
            write(task_file, task)
            state["jobs"][name] = dict(status="running", action=task["action"], task_sha256=token,
                                        started_at=datetime.now(timezone.utc).isoformat())
            write(state_path, state)
            log = out / "logs" / (name + ".log"); log.parent.mkdir(exist_ok=True)
            print(f"START {name}: {log}", flush=True)
            with log.open("x", encoding="utf-8") as stream:
                result = subprocess.run([sys.executable, "-u", str(ROOT / "main.py"), "experiment", "study-queue",
                                         "--internal-task", str(task_file)], cwd=ROOT, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False)
            item = state["jobs"][name]
            item.update(exit_code=result.returncode, status="failed" if result.returncode else "completed")
            if not result.returncode:
                item["artifacts"] = {p.relative_to(directory).as_posix(): digest(p) for p in directory.rglob("*")
                                      if p.is_file() and p.suffix in (".json", ".pt", ".npz")}
            write(state_path, state)
            if result.returncode:
                tail = log.read_text(encoding="utf-8", errors="replace")[-20000:].lower()
                if any(message in tail for message in ("out of memory", "no space left", "cuda error", "device-side assert")):
                    raise ResourceFailure(f"Resource/device failure in {name}; inspect {log}")
                raise RuntimeError(f"{name} failed; inspect {log}; no automatic retry")
            return read(directory / "results.json")

        def train(name, cfg, seed):
            return job(name, dict(action="train", config=cfg, seed=seed))

        for feature in spec["feature_modes"]:
            cfg = base_for("ec50_assay")
            cfg.update(fragment_feature_mode=feature, epochs=spec["pilot_epochs"], compare_checkpoint_rules=False)
            train("pilot_" + feature, cfg, spec["screen_seeds"][0])
        if stage_limit == "pilot":
            state["status"] = "pilot_completed"; write(state_path, state)
            return state
        scores = {}
        for feature, preset, seed in itertools.product(spec["feature_modes"], spec["diversity"], spec["screen_seeds"]):
            cfg = base_for("ec50_assay")
            cfg.update(spec["diversity"][preset], fragment_feature_mode=feature,
                       diagnostic_molecules=spec["diagnostic_molecules"])
            r = train(f"screen_{feature}_{preset}_{seed}", cfg, seed)
            scores[feature, preset, seed] = r["best"]["independent"]["auc"]
        diversity = choose_diversity(spec, scores)
        state["decisions"]["diversity"] = diversity; write(state_path, state)
        if stage_limit == "screen":
            state["status"] = "screen_completed"; write(state_path, state)
            return state
        scores = {}
        for feature, coefficient, seed in itertools.product(spec["feature_modes"], spec["contrast_coefficients"], spec["tune_seeds"]):
            cfg = base_for("ec50_assay")
            cfg.update(spec["diversity"][diversity[feature]], fragment_feature_mode=feature, lambda_contrast=coefficient)
            r = train(f"tune_{feature}_{coefficient}_{seed}", cfg, seed)
            scores[feature, coefficient, seed] = r["best"]["independent"]["auc"]
        contrast = choose_contrast(spec, scores)
        state["decisions"]["contrast"] = contrast; write(state_path, state)
        if stage_limit == "tune":
            state["status"] = "tune_completed"; write(state_path, state)
            return state
        frozen = []
        failures = []
        for dataset in spec["confirmation_datasets"]:
            try:
                base = base_for(dataset)
                for arm, seed in itertools.product(spec["feature_modes"] + spec["controls"], spec["confirmation_seeds"]):
                    cfg = deepcopy(base)
                    cfg["diagnostic_molecules"] = spec["diagnostic_molecules"]
                    if arm in spec["feature_modes"]:
                        cfg.update(spec["diversity"][diversity[arm]], fragment_feature_mode=arm, lambda_contrast=contrast[arm])
                    elif arm == "random_learned_gates":
                        # Fixed historical control; not falsely described as an equally tuned method.
                        cfg.update(ranking_mode="random", lambda_diversity=0.0, lambda_contrast=0.1)
                    else:
                        cfg.update(erm_baseline=True, lambda_contrast=0.0, lambda_diversity=0.0)
                    name = f"confirm_{dataset}_{arm}_{seed}"
                    train(name, cfg, seed)
                    frozen.append(dict(name=name, dataset=dataset, arm=arm, seed=seed, cache=cfg["processed_dir"]))
            except ResourceFailure:
                raise
            except Exception as error:
                failures.append(dict(dataset=dataset, error=str(error)))
                write(out / "branch_failures.json", failures)
        # All configurations are frozen before the queue opens a test loader.
        write(out / "frozen_confirmation.json", frozen)
        evaluations = []
        for entry in frozen:
            result = job("eval_" + entry["name"], dict(action="evaluate", seed=entry["seed"],
                cache=entry["cache"], device=device, training_directory=str(out / "jobs" / entry["name"])))
            evaluations.append(dict(entry, metrics=result))
            write(out / "evaluation_progress.json", evaluations)
        groups = {}
        for dataset, arm in itertools.product(spec["confirmation_datasets"], spec["feature_modes"] + spec["controls"]):
            values = [r["metrics"]["best_raw"]["ood_test"]["auc"] for r in evaluations if r["dataset"] == dataset and r["arm"] == arm]
            if values:
                groups[f"{dataset}/{arm}"] = dict(n=len(values), mean=statistics.mean(values),
                    sample_std=statistics.stdev(values) if len(values) > 1 else None,
                    complete=len(values) == len(spec["confirmation_seeds"]))
        paired = {}
        for dataset in spec["confirmation_datasets"]:
            values = {arm: {r["seed"]: r["metrics"]["best_raw"]["ood_test"]["auc"]
                           for r in evaluations if r["dataset"] == dataset and r["arm"] == arm}
                      for arm in spec["feature_modes"] + spec["controls"]}
            for first, second in (("internal_gnn", "static_pool"), ("static_pool", "random_learned_gates"),
                                  ("internal_gnn", "random_learned_gates"), ("static_pool", "erm")):
                seeds = sorted(set(values[first]) & set(values[second]))
                if seeds:
                    differences = [values[first][s] - values[second][s] for s in seeds]
                    sd = statistics.stdev(differences) if len(seeds)>1 else None
                    mean = statistics.mean(differences)
                    # Fixed five-seed confirmation: conditional seed interval only.
                    radius = 2.7764451052 * sd / math.sqrt(5) if len(seeds)==5 else None
                    paired[f"{dataset}/{first}_minus_{second}"] = dict(seeds=seeds, differences=differences,
                        mean=mean, sample_std=sd, positive_seeds=sum(d>0 for d in differences),
                        seed_t95=[mean-radius,mean+radius] if radius is not None else None,
                        note="Seed-only uncertainty; not molecule/assay sampling or benchmark-selection uncertainty")
        write(out / "summary.json", dict(primary_test=groups, paired=paired, runs=evaluations, branch_failures=failures,
              decisions=state["decisions"], synthetic=smoke, upstream_status=spec["upstream_status"],
              note="Current pipeline study, not faithful upstream SuGAr. Fixed controls are not equally tuned. Test never selects."))
        state["status"] = "completed_with_failed_branches" if failures else "completed"
        write(state_path, state)
        return state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="configs/mentor_study_queue.yaml")
    p.add_argument("--output-dir")
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--cache-root", default="data/processed/mentor_study_v1")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--stage-limit", choices=("pilot", "screen", "tune", "all"), default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-from", help="Resume an existing queue while the guarded launcher uses a fresh record directory")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--internal-task", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.internal_task:
        return task_worker(a.internal_task)
    if not a.output_dir:
        p.error("--output-dir required")
    if a.resume_from:
        if not a.resume:
            p.error("--resume-from requires --resume")
        previous = Path(a.resume_from).resolve()
        if not (previous / "status.json").is_file():
            p.error("--resume-from must identify an existing queue")
        record = Path(a.output_dir).resolve()
        if record != previous and not a.dry_run:
            if record.exists():
                p.error("Fresh launcher record output required")
            write(record / "resume_pointer.json", dict(queue=str(previous)))
        a.output_dir = str(previous)
    spec = yaml.safe_load(Path(a.manifest).read_text(encoding="utf-8"))
    if a.smoke:
        spec.update(epochs=5, pilot_epochs=1, screen_seeds=[19501], tune_seeds=[19511],
                    confirmation_seeds=[19521], contrast_coefficients=[0.1, 0.5],
                    confirmation_datasets=["ec50_assay"], minimum_free_disk_gib=0.01,
                    diversity_min_seed_wins=1, diagnostic_molecules=4)
        spec["diversity"] = {k: spec["diversity"][k] for k in ("off", "ramp")}
        spec["diversity"]["ramp"].update(diversity_delay_epochs=1, diversity_ramp_epochs=2)
    if a.dry_run:
        counts = validate(spec)
        print(json.dumps(dict(training_runs=counts, total=sum(counts.values()), spec=spec,
                             upstream_ready=False), indent=2))
        return
    run_queue(spec, a.output_dir, a.raw_root, a.cache_root, a.device, a.smoke, a.resume, a.stage_limit)


if __name__ == "__main__":
    main()
