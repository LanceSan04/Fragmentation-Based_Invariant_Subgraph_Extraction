"""Matched classifier readout pilot on the official Ki Assay split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.dataloader import load_processed_split
from src.data.preprocess import run_preprocessing
from src.evaluation.within_assay import within_assay_report
from src.experiments.checkpoints import digest, evaluate_saved
from src.experiments.context_test import require_exact_predictions
from src.experiments.cross_domain import prepare_cache, verify_cache
from src.experiments.fragment_tuning import contained
from src.experiments.original_tuning import ki_transfer_inputs
from src.experiments.plans import ROOT, make_plan
from src.experiments.selection import run_seed, validation_execution, write
from src.training.pipeline import Config


READOUTS = ("mean", "sum", "mean_count")
SEED = 10801


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    project = args.project_root.resolve()
    protocol = ROOT / "configs/official_original_tuning_20261030.json"
    spec = json.loads(protocol.read_text())
    inputs, selected = ki_transfer_inputs(project, digest(protocol), args.smoke)
    entry = next(row for row in inputs if row["dataset"] == "ki_assay")
    recipe = selected["ki_assay"]["recipe"]
    epochs = 2 if args.smoke else spec["confirmation_epochs"]
    plan = dict(dataset="ki_assay", recipe=recipe, seed=SEED, epochs=epochs,
                readouts=list(READOUTS), deterministic_training=True, device=args.device,
                synthetic=args.smoke, training_runs=len(READOUTS), test_planned=True)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return plan
    raw = contained(project, entry["raw_path"])
    cache = contained(project, entry["cache_path"])
    base = dict(spec["base"], dataset="drugood_lbap_core_ki_assay", device=args.device,
                epochs=epochs, raw_path=str(raw), processed_dir=str(cache),
                deterministic_training=True)
    base.update(dict(zip(spec["columns"], spec["recipes"][recipe - 1])))
    if not args.smoke and digest(raw) != entry["raw_sha256"]:
        raise ValueError("Pinned downloaded Ki Assay file changed")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    torch.set_num_threads(1)
    if args.smoke:
        _, fixture = make_plan(out / "fixture", args.device, True)
        base.update(fixture)
        base.update(epochs=epochs, hidden_dim=8, gnn_layers=1, num_views=2,
                    view_fragmentations={"group_graph": 2}, num_workers=0,
                    loader_split_names=["train", "ood_val"], validation_device="cpu",
                    report_within_assay=False, save_selected_validation_predictions=True)
        run_preprocessing(Config(base))
        cache = Path(base["processed_dir"])
    elif cache.exists():
        verify_cache(base)
    else:
        prepare_cache(base)
    rows = {}
    for readout in READOUTS:
        run = out / readout
        print(f"Training {readout}, seed {SEED}", flush=True)
        result = run_seed(dict(base, classifier_readout=readout), run, SEED,
                          policies=("independent",))
        checkpoint = run / "best_independent.pt"
        reference = run / "best_independent_validation_predictions.npz"
        checkpoint_hash = digest(checkpoint)
        with validation_execution("cpu"):
            replay = evaluate_saved(checkpoint, "ood_val", cache, "cpu", "independent",
                                    SEED, prediction_path=run / "validation_replay.npz")
        require_exact_predictions(run / "validation_replay.npz", reference)
        best = result["best"]["independent"]
        if replay["auc"] != best["auc"]:
            raise ValueError(f"Validation AUC changed after reload: {readout}")
        rows[readout] = dict(validation=replay, selected_epoch=best["epoch"],
                             checkpoint_sha256=checkpoint_hash, resources=result["resources"])
        write(out / "validation_progress.json", rows)
    test_graphs = load_processed_split(cache / "ood_test.pt")["molecule_data"]
    sizes = [str(graph.x.size(0)) for graph in test_graphs]
    expected_labels = [int(graph.y.item()) for graph in test_graphs]
    del test_graphs
    for readout, row in rows.items():
        run = out / readout
        checkpoint = run / "best_independent.pt"
        if digest(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint changed: {readout}")
        prediction_path = run / "test_predictions.npz"
        with validation_execution("cpu"):
            row["test"] = evaluate_saved(checkpoint, "ood_test", cache, "cpu", "independent",
                                         SEED, prediction_path=prediction_path)
        with np.load(prediction_path) as saved:
            labels, scores = saved["labels"], saved["probabilities"]
            if not np.array_equal(labels, expected_labels):
                raise ValueError("Test prediction labels do not align with the graph cache")
            row["within_assay"] = within_assay_report(labels, scores, saved["assays"])
            row["same_atom_count"] = within_assay_report(labels, scores, sizes, min_per_class=1)
        write(out / "test_progress.json", rows)
    report = dict(plan=plan, runs=rows, test_evaluated=True)
    write(out / "completed.json", report)
    return report


if __name__ == "__main__":
    main()
