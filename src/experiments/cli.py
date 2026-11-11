"""Unified experiment CLI; delegates to reusable workflow modules."""

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = {
    "training-trace": "training_trace",
    "training-repeat": "training_repeat",
    "checkpoint-test-replay": "checkpoint_test_replay",
    "ec50-seed-check": "ec50_seed_check",
    "ec50-replication": "ec50_replication",
    "original-tuning": "original_tuning",
    "cpu-frozen-test": "cpu_frozen_test",
    "cpu-validation-audit": "cpu_validation_audit",
    "reload-diagnostic": "reload_diagnostic",
    "context-transition": "context_transition",
    "context-features": "context_features",
    "ki-reference-controls": "ki_reference_controls",
    "ki-readout": "ki_readout",
    "fragment-confirm": "fragment_confirmation",
    "fragment-tuning": "fragment_tuning",
    "fragment-view-screen": "fragment_view_screen",
    "fragment-dataset-search": "fragment_dataset_search",
    "selection-faithfulness": "selection_faithfulness",
    "sugar-inference-compare": "sugar_inference_compare",
    "sugar-reference-replicate": "sugar_reference_replicate",
    "sugar-reference-search": "sugar_reference_search",
    "sugar-reference-eval": "sugar_reference_eval",
    "research-batch": "research_batch",
    "upstream-sugar": "upstream_sugar",
    "dataset-readiness": "dataset_readiness",
    "mentor-confirm": "mentor_confirmation",
    "study-queue": "study_queue",
    "scorer-paths": "scorer_paths",
    "granularity": "granularity",
    "motif-check": "motif_check",
    "select": "selection",
    "cross-domain": "cross_domain",
    "tune": "parameter_search",
    "compare-inference": "inference",
    "compare-views": "views",
    "diagnose": "diagnostics",
    "visualize-views": "visualize_views",
    "selection-gating-probe": "selection_gating_probe",
}
OUTPUT_WORKFLOWS = {"tune", "select", "compare-inference", "compare-views", "diagnose"}


def option(args, name, default=None):
    for i, arg in enumerate(args):
        if arg == name:
            return args[i + 1] if i + 1 < len(args) else default
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return default


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified experiments. Example: main.py experiment select --help"
    )
    parser.add_argument("workflow", choices=WORKFLOWS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parsed = parser.parse_args(argv)
    args = parsed.arguments
    workflow = parsed.workflow
    os.chdir(ROOT)
    output = None
    if workflow in OUTPUT_WORKFLOWS:
        supplied = option(args, "--output-dir")
        if supplied:
            output = Path(supplied).resolve()
        else:
            smoke = "--smoke" in args
            dataset = (
                "smoke"
                if smoke
                else option(
                    args, "--dataset", "assay" if workflow in {"tune", "select"} else "analysis"
                )
            )
            output = (
                ROOT / "outputs" / dataset / workflow / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            args = args + ["--output-dir", str(output)]
    module = importlib.import_module("src.experiments." + WORKFLOWS[workflow])
    dry = "--dry-run" in args or "--help" in args or "-h" in args
    # Never overwrite a previous experiment's metadata, even if a workflow fails early.
    existed = output.exists() if output else False
    record = dict(
        workflow=workflow,
        arguments=args,
        started_at=datetime.now(timezone.utc).isoformat(),
        status="failed",
    )
    try:
        result = module.main(args)
        record["status"] = "returned"  # Inspect native trial/run states for completeness.
        return result
    finally:
        if output and output.is_dir() and not existed and not dry:
            record["result_files"] = [
                str(p.relative_to(output))
                for pattern in [
                    "summary.json",
                    "eval_results.json",
                    "comparison_all.json",
                    "comparison.json",
                    "best_params.json",
                    "leaderboard.csv",
                ]
                for p in output.rglob(pattern)
            ]
            with (output / "experiment.json").open("x") as f:
                json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
