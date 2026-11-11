"""Two fresh deterministic IC50 Scaffold trajectories using the frozen winning recipe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from src.experiments.checkpoints import evaluate_saved
from src.experiments.context_test import require_exact_predictions
from src.experiments.cross_domain import verify_cache
from src.experiments.fragment_tuning import contained
from src.experiments.plans import ROOT, make_plan
from src.experiments.selection import run_seed, validation_execution, write
from src.experiments.training_trace import fingerprint
from src.data.preprocess import run_preprocessing
from src.training.checkpoint import load_checkpoint, view_state_dicts_from_checkpoint


class EpochTrace:
    def initialize(self, views, trainer, loader):
        self.views = views
        self.trainer = trainer
        self.batch_hashes = []
        for index in range(len(loader.dataset)):
            loader.dataset[index].trace_row = torch.tensor([index])
        self.initial = self.state_hashes()

    def state_hashes(self):
        return {f"view{i}.{name}": fingerprint(value)
                for i, view in enumerate(self.views) for name, value in view.state_dict().items()
                if isinstance(value, torch.Tensor)}

    def capture(self, step, stage, data):
        if stage == "batch":
            self.batch_hashes.append(dict(rows=fingerprint(data.trace_row),
                                          labels=fingerprint(data.y)))
        elif stage == "forward":
            self.batch_hashes[-1]["sampling"] = [
                fingerprint(result["fragment_edge_eligible"]) for result in data]
            self.batch_hashes[-1]["selection"] = [
                fingerprint(result["selected_fragment_mask"]) for result in data]

    def finish_epoch(self, epoch):
        result = dict(model=self.state_hashes(), batches=self.batch_hashes)
        self.batch_hashes = []
        return result


def worker(config, out):
    values = json.loads(config.read_text())
    torch.set_num_threads(1)
    trace = EpochTrace()
    run_seed(values, out, 11321, False, ("independent",), diagnostic=trace)
    write(out / "initial_hashes.json", trace.initial)
    checkpoint = out / "best_independent.pt"
    with validation_execution("cpu"):
        metrics = evaluate_saved(checkpoint, "ood_val", Path(values["processed_dir"]),
                                 "cpu", "independent", 11321,
                                 prediction_path=out / "validation_replay.npz")
    require_exact_predictions(out / "validation_replay.npz",
                              out / "best_independent_validation_predictions.npz")
    write(out / "reload.json", dict(validation=metrics, predictions_exact=True))


def compare(out):
    left, right = out / "run1", out / "run2"
    initial_equal = json.loads((left / "initial_hashes.json").read_text()) == json.loads(
        (right / "initial_hashes.json").read_text())
    histories = [json.loads((p / "history.json").read_text()) for p in (left, right)]
    comparisons = []
    for a, b in zip(*histories):
        comparisons.append(dict(epoch=a["epoch"], training_exact=all(
            a[k] == b[k] for k in ("train_loss", "train_accuracy", "train_components")),
            validation_exact=a["validation"] == b["validation"],
            model_exact=a["reproducibility"]["model"] == b["reproducibility"]["model"],
            batch_order_sampling_selection_exact=(
                a["reproducibility"]["batches"] == b["reproducibility"]["batches"])))
    if len(histories[0]) != len(histories[1]):
        raise ValueError("Trajectory lengths differ")
    checkpoints = [view_state_dicts_from_checkpoint(load_checkpoint(p / "best_independent.pt"))
                   for p in (left, right)]
    checkpoint_exact = checkpoints[0].keys() == checkpoints[1].keys() and all(
        checkpoints[0][i].keys() == checkpoints[1][i].keys() and all(
            torch.equal(checkpoints[0][i][k], checkpoints[1][i][k])
            for k in checkpoints[0][i]) for i in checkpoints[0])
    best = [json.loads((p / "best.json").read_text()) for p in (left, right)]
    import numpy as np
    with np.load(left / "validation_replay.npz") as a, np.load(right / "validation_replay.npz") as b:
        predictions_exact = all(np.array_equal(a[k], b[k]) for k in ("labels", "probabilities"))
    return dict(initial_exact=initial_equal, epochs=comparisons,
                selected_checkpoint_tensors_exact=checkpoint_exact, best=best,
                checkpoint_selection_exact=best[0] == best[1],
                reloaded_predictions_exact=predictions_exact,
                all_exact=initial_equal and checkpoint_exact and best[0] == best[1]
                and predictions_exact and all(all(v for k, v in row.items() if k != "epoch")
                                              for row in comparisons))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker-config", type=Path)
    args = parser.parse_args(argv)
    if args.worker_config:
        return worker(args.worker_config, args.output_dir)
    if args.project_root is None:
        parser.error("--project-root is required")
    project = args.project_root.resolve()
    spec = json.loads((ROOT / "configs/official_original_tuning_20261030.json").read_text())
    entry = next(r for r in spec["datasets"] if r["dataset"] == "ic50_scaffold")
    values = dict(spec["base"], dataset="drugood_lbap_core_ic50_scaffold",
                  raw_path=str(contained(project, entry["raw_path"])),
                  processed_dir=str(contained(project, entry["cache_path"])),
                  device=args.device, epochs=150, deterministic_training=True,
                  seed=11321, sampler_seed=11321)
    values.update(dict(zip(spec["columns"], spec["recipes"][1])))
    if not args.smoke:
        from src.experiments.checkpoints import digest
        if digest(values["raw_path"]) != entry["raw_sha256"]:
            raise ValueError("Pinned IC50 Scaffold source changed")
        verify_cache(values)
        search = json.loads((project / "outputs/cross_domain/ic50_search_20261102_v1/results/completed.json").read_text())
        if search["winners"]["ic50_scaffold"]["recipe"] != 2 or search["test_evaluated"]:
            raise ValueError("Search evidence does not match the approved recipe")
    plan = dict(dataset="ic50_scaffold", recipe=2, seed=11321, runs=2,
                epochs=2 if args.smoke else 150, deterministic_training=True,
                device=args.device, synthetic=args.smoke, test_evaluated=False)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return plan
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.smoke:
        _, fixture = make_plan(out / "fixture", "cpu", True)
        values.update(fixture)
        values.update(epochs=2, hidden_dim=8, gnn_layers=1, num_workers=0,
                      device="cpu", num_views=2, view_fragmentations={"group_graph": 2},
                      validation_device="cpu", loader_split_names=["train", "ood_val"],
                      report_within_assay=False, save_selected_validation_predictions=True)
        from src.training.pipeline import Config
        run_preprocessing(Config(values))
    write(out / "plan.json", plan)
    write(out / "config.json", values)
    for iteration in (1, 2):
        with (out / f"run{iteration}.log").open("x") as log:
            subprocess.run([sys.executable, "-m", "src.experiments.training_repeat",
                            "--worker-config", str(out / "config.json"),
                            "--output-dir", str(out / f"run{iteration}")],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    report = compare(out)
    write(out / "completed.json", dict(plan=plan, **report))
    print(json.dumps({k: v for k, v in report.items() if k != "epochs"}, indent=2))
    return report


if __name__ == "__main__":
    main()
