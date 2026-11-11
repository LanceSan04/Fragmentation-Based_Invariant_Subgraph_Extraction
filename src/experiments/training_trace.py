"""Compare the first five optimizer steps in two fresh training processes."""

import argparse
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

from src.data.dataloader import build_split_dataloaders
from src.data.preprocess import run_preprocessing
from src.experiments.checkpoints import digest
from src.experiments.common import set_all_seeds
from src.experiments.fragment_tuning import contained
from src.experiments.plans import ROOT, make_plan
from src.experiments.sugar_reference_search import write
from src.training.pipeline import (
    Config, build_trainer, build_views_from_feature_dims, infer_feature_dimensions,
    strip_non_tensor_batch_fields,
)
from src.utils.reproducibility import deterministic_training
from src.utils.runtime_spec import runtime_spec

SEED = 11422
STEPS = 5


class FirstBatches:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return islice(self.loader, STEPS)

    def __len__(self):
        return min(len(self.loader), STEPS)


def snapshot(tensors):
    return {name: value.detach().cpu().clone() for name, value in tensors.items()
            if isinstance(value, torch.Tensor)}


def fingerprint(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def trace_worker(config_path, out):
    values = json.loads(config_path.read_text())
    with deterministic_training(values.get("deterministic_training", False)):
        return _trace_worker(values, out)


def _trace_worker(values, out):
    cfg = Config(values)
    torch.set_num_threads(1)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")
    set_all_seeds(SEED)
    loaders = build_split_dataloaders(cfg.processed_dir, cfg)
    loader = loaders["train"]
    molecule_ids = []
    for index in range(len(loader.dataset)):
        molecule = loader.dataset[index]
        molecule_ids.append(str(getattr(molecule, "molecule_id", index)))
        molecule.trace_row = torch.tensor([index])
    strip_non_tensor_batch_fields(loaders)
    dims = infer_feature_dimensions(loader)
    views = build_views_from_feature_dims(cfg, dims)
    trainer = build_trainer(cfg, views, cfg.device)
    trainer.effective_diversity_weight = trainer.lambda_diversity
    out.mkdir(exist_ok=False)
    runtime = runtime_spec(cfg, views, trainer, ("independent",))
    if cfg.device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        runtime["gpu_name"] = properties.name
        runtime["gpu_uuid"] = str(getattr(properties, "uuid", "")) or None
    write(out / "runtime.json", runtime)

    def model_state():
        return {f"view{i}.{name}": value for i, view in enumerate(views)
                for name, value in view.state_dict().items()}

    parameters = {f"view{i}.{name}": param for i, view in enumerate(views)
                  for name, param in view.named_parameters()}
    initial = snapshot(model_state())
    torch.save(initial, out / "initial.pt")
    write(out / "initial_hashes.json", {name: fingerprint(tensor) for name, tensor in initial.items()})
    stages = {}
    batch_order = []

    def capture(step, stage, data):
        if stage == "batch":
            indices = data.trace_row.tolist()
            batch_order.append([molecule_ids[index] for index in indices])
            stages["molecule_order"] = snapshot({"rows": data.trace_row, "labels": data.y})
        elif stage == "forward":
            for key, names in (
                ("sampling_masks", ("fragment_edge_eligible",)),
                ("fragment_scores", ("fragment_scores",)),
                ("selection_masks", ("selected_fragment_mask",)),
                ("forward", ("predictions", "logits", "embeddings")),
            ):
                stages[key] = snapshot({f"view{i}.{name}": result[name]
                                        for i, result in enumerate(data) for name in names})
        elif stage == "loss":
            stages[stage] = snapshot(data)
        elif stage in ("gradients", "clipped_gradients"):
            stages[stage] = snapshot({name: param.grad for name, param in parameters.items()})
        elif stage == "update":
            stages["updated_state"] = snapshot(model_state())
            parameter_names = {id(param): name for name, param in parameters.items()}
            stages["optimizer_state"] = snapshot({
                f"{parameter_names[id(param)]}.{key}": value
                for optimizer in trainer.optimizers for param, state in optimizer.state.items()
                for key, value in state.items()
            })
            torch.save(stages, out / f"batch{step + 1}.pt")
            stages.clear()

    trainer.train_epoch(FirstBatches(loader), trace=capture)
    if len(batch_order) != STEPS:
        raise RuntimeError(f"Expected {STEPS} batches, observed {len(batch_order)}")
    write(out / "batch_molecules.json", batch_order)


def compare_stage(left, right):
    changed = sorted(left.keys() ^ right.keys())
    maximum = 0.0
    comparable = not changed
    for name in sorted(left.keys() & right.keys()):
        a, b = left[name], right[name]
        if a.dtype != b.dtype or a.shape != b.shape:
            changed.append(name)
            comparable = False
        elif fingerprint(a) != fingerprint(b):
            changed.append(name)
            if a.numel():
                delta = (a.double() - b.double()).abs().max().item()
                if math.isfinite(delta):
                    maximum = max(maximum, delta)
                else:
                    comparable = False
    return dict(exact=not changed, differing_tensors=len(changed), examples=changed[:10],
                max_abs_difference=maximum if comparable else None)


def compare_runs(out):
    def load(run, name):
        return torch.load(out / run / name, map_location="cpu", weights_only=True)

    comparison = [{"batch": 0, "stage": "initial_state",
                   **compare_stage(load("run1", "initial.pt"), load("run2", "initial.pt"))}]
    for step in range(1, STEPS + 1):
        left = load("run1", f"batch{step}.pt")
        right = load("run2", f"batch{step}.pt")
        for stage in left:
            comparison.append({"batch": step, "stage": stage, **compare_stage(left[stage], right[stage])})
    first = next((row for row in comparison if not row["exact"]), None)
    return dict(all_exact=first is None, first_observed_difference=first, comparisons=comparison)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--deterministic-training", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    if args.worker_config:
        return trace_worker(args.worker_config, out)
    if args.project_root is None:
        parser.error("--project-root is required")
    if args.smoke and args.device != "cpu":
        parser.error("The synthetic smoke uses CPU")
    protocol = ROOT / "configs/official_original_tuning_20261030.json"
    spec = json.loads(protocol.read_text())
    entry = next(row for row in spec["datasets"] if row["dataset"] == "ec50_assay")
    project = args.project_root.resolve()
    cache = contained(project, entry["cache_path"])
    values = dict(spec["base"], dataset="drugood_lbap_core_ec50_assay", processed_dir=str(cache),
                  raw_path=str(contained(project, entry["raw_path"])), device=args.device,
                  seed=SEED, sampler_seed=SEED, loader_split_names=["train"], loss_log_path=None)
    values.update(dict(zip(spec["columns"], spec["recipes"][0])))
    values["deterministic_training"] = args.deterministic_training
    if not args.smoke:
        marker = json.loads((cache / "cross_domain_cache.json").read_text())
        raw_sha = digest(values["raw_path"])
        if raw_sha != entry["raw_sha256"] or raw_sha != marker["raw_sha256"]:
            raise ValueError("Downloaded EC50 Assay source changed")
        if digest(cache / "train.pt") != marker["files"]["train"]:
            raise ValueError("EC50 Assay training cache changed")
    plan = dict(dataset=entry["dataset"], seed=SEED, recipe=1, processes=2,
                minibatches_per_process=STEPS, device=args.device, synthetic=args.smoke,
                loaded_splits=["train"], deterministic_training=args.deterministic_training,
                protocol_sha256=digest(protocol))
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return plan
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    if args.smoke:
        _, fixture = make_plan(out / "fixture", "cpu", True)
        fixture["splits"] = {"train": fixture["splits"]["train"] * STEPS}
        fixture["split_names"] = ["train"]
        values.update(fixture)
        values.update(seed=SEED, sampler_seed=SEED, loader_split_names=["train"],
                      fragment_feature_mode="static_pool", classifier_feature_mode="raw",
                      num_views=3, view_fragmentations={"group_graph": 3},
                      lambda_contrast=.1, lambda_diversity=.005, lambda_irm=0.)
        run_preprocessing(Config(values))
    config_path = out / "config.json"
    write(config_path, values)
    for index in (1, 2):
        with (out / f"run{index}.log").open("x") as log:
            subprocess.run([sys.executable, "-m", "src.experiments.training_trace",
                            "--worker-config", str(config_path), "--output-dir", str(out / f"run{index}")],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = compare_runs(out)
    write(out / "completed.json", dict(plan=plan, **report))
    print(json.dumps({key: report[key] for key in ("all_exact", "first_observed_difference")}, indent=2))
    return report


if __name__ == "__main__":
    main()
