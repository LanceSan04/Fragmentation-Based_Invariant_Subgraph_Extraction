#!/usr/bin/env python
r"""Consolidated CLI for the REU SuGAr-style molecular pipeline.

This entrypoint orchestrates preprocessing, FragmentCIGA/SuGAr training,
checkpoint resume/evaluation, processed-data inspection, and diagnostics.
Most experiment defaults come from configs/experiment.yaml; CLI flags override
those defaults for a single run.

When ``output_root`` is configured, run artifacts are consolidated under:

    outputs/{dataset}/{fragmentation_recipe}/{run_name}/

That run folder contains checkpoints, training logs, final metrics,
diagnostic outputs, the effective config, and run_manifest.json. Processed
``.pt`` split files stay under data/processed/... because they are reusable
dataset artifacts.

Examples:
    # Run the full pipeline: preprocess, train, checkpoint, metrics, diagnostics
    .\.venv\Scripts\python.exe main.py full --config configs\experiment.yaml --validate-loaders --fragment-diagnostics --loss-plots --subgraph-analysis

    # Preprocess the configured dataset and validate PyG loaders
    .\.venv\Scripts\python.exe main.py preprocess --config configs\experiment.yaml --validate-loaders

    # Short CPU smoke training run using a named consolidated output folder
    .\.venv\Scripts\python.exe main.py train --config configs\experiment.yaml --run-name smoke_cpu --epochs 1 --patience 1 --num-views 1 --hidden-dim 8 --gnn-layers 1 --batch-size 512 --device cpu --lambda-contrast 0 --lambda-diversity 0 --lambda-irm 0

    # Evaluate a saved checkpoint without training
    .\.venv\Scripts\python.exe main.py eval outputs\...\checkpoints\best_model.pt --config configs\experiment.yaml --splits ood_test iid_test --device cpu

    # Resume training from a checkpoint
    .\.venv\Scripts\python.exe main.py resume outputs\...\checkpoints\best_model.pt --config configs\experiment.yaml --epochs 5 --run-name resumed_run

    # Run fragmentation diagnostics
    .\.venv\Scripts\python.exe main.py diagnostics fragments data\processed\drugood_lbap_core_ec50_assay\group_graph --splits train ood_val ood_test --output-dir outputs\manual\group_graph_fragmentation_diagnostics

    # Plot loss logs
    .\.venv\Scripts\python.exe main.py diagnostics losses outputs\...\training_epoch_losses.csv --output-dir outputs\...\training_curves

    # Analyze trained invariant subgraphs from a saved run config
    .\.venv\Scripts\python.exe main.py diagnostics subgraphs --config outputs\...\config\effective_config.yaml --subgraph-split ood_test

    # Inspect one processed molecule record
    .\.venv\Scripts\python.exe main.py inspect data\processed\drugood_lbap_core_ec50_assay\group_graph --splits train --index 0
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.data.dataloader import build_split_dataloaders, validate_dataloader_batch
from train_sugar import Config, evaluate_checkpoint, run_training
from src.utils.run_outputs import prepare_run_outputs, resolve_processed_dir, write_run_manifest


ROOT = Path(__file__).resolve().parent


COMMON_OVERRIDES = (
    "dataset",
    "processed_root",
    "processed_dir",
    "device",
    "epochs",
    "patience",
    "num_views",
    "hidden_dim",
    "gnn_layers",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "lambda_contrast",
    "lambda_diversity",
    "lambda_irm",
    "num_envs",
    "env_bucket_strategy",
    "save_model",
    "loss_log_path",
    "final_metrics_path",
    "output_root",
    "run_name",
    "subgraph_split",
    "subgraph_num_samples",
    "subgraph_num_visualize",
    "subgraph_metric_min_available",
    "subgraph_metric_top_k",
    "train_split",
    "validation_split",
    "test_split",
)


def load_yaml_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def apply_common_overrides(config, args):
    """Apply explicit CLI overrides to a config dictionary."""
    merged = dict(config or {})
    explicit_processed_dir = getattr(args, "processed_dir", None) is not None

    for key in COMMON_OVERRIDES:
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value

    explicit_output_paths = [
        key
        for key in ("save_model", "loss_log_path", "final_metrics_path")
        if getattr(args, key, None) is not None
    ]
    if explicit_output_paths:
        merged["_explicit_output_paths"] = explicit_output_paths

    if not explicit_processed_dir and not merged.get("processed_dir"):
        resolve_processed_dir(merged)
        merged.pop("failure_report_path", None)
        print(
            "No processed_dir supplied; using view-fragmentation processed_dir: "
            f"{merged['processed_dir']}"
        )

    return merged


def build_effective_config(args, checkpoint_path=None):
    config_path = getattr(args, "config", None) or "configs/experiment.yaml"
    config = load_yaml_config(config_path)
    if checkpoint_path:
        from src.training.checkpoint import checkpoint_config, load_checkpoint

        checkpoint_payload = load_checkpoint(checkpoint_path, map_location="cpu")
        saved_config = checkpoint_config(checkpoint_payload)
        if saved_config:
            config.update(saved_config)
    return Config(apply_common_overrides(config, args))


def validate_loaders(processed_dir, config):
    loaders = build_split_dataloaders(processed_dir, config)
    for split_name, loader in loaders.items():
        batch = next(iter(loader), None)
        if batch is not None:
            validate_dataloader_batch(batch)
            print(f"Validated loader batch for split {split_name!r}")


def run_preprocess_command(args):
    from src.data.preprocess import run_preprocessing

    config = build_effective_config(args)
    prepare_run_outputs(config, command_name="preprocess")
    result = run_preprocessing(config.to_dict())
    if args.validate_loaders:
        validate_loaders(config.processed_dir, config)
    write_run_manifest(config, command_name="preprocess")
    return result


def run_train_command(args):
    config = build_effective_config(args, checkpoint_path=getattr(args, "resume_from", None))
    return run_training(config, resume_from=getattr(args, "resume_from", None))


def run_full_command(args):
    from src.data.preprocess import run_preprocessing

    config = build_effective_config(args, checkpoint_path=getattr(args, "resume_from", None))
    prepare_run_outputs(config, command_name="full")
    if not args.skip_preprocess:
        run_preprocessing(config.to_dict())
        if args.validate_loaders:
            validate_loaders(config.processed_dir, config)

    result = run_training(config, resume_from=getattr(args, "resume_from", None))

    if args.fragment_diagnostics:
        run_fragment_diagnostics(
            processed_dir=Path(config.processed_dir),
            splits=args.diagnostic_splits or config.get("split_names", None),
            output_dir=(
                Path(args.fragment_diagnostics_output)
                if args.fragment_diagnostics_output
                else Path(config.get("fragmentation_diagnostics_dir"))
                if config.get("fragmentation_diagnostics_dir", None)
                else None
            ),
            max_molecules=args.max_molecules,
            no_plots=args.no_plots,
            top_k=args.top_k,
        )

    if args.loss_plots:
        loss_log = args.loss_log or config.get("loss_log_path", None)
        if loss_log:
            run_loss_diagnostics(
                loss_log,
                args.loss_output_dir or config.get("training_diagnostics_dir", None),
            )
        else:
            print("No loss log path configured; skipping loss plots.")

    if args.subgraph_analysis:
        run_subgraph_diagnostics(
            config=config,
            model_path=args.subgraph_model_path,
            processed_dir=args.subgraph_processed_dir,
            split=args.subgraph_split,
            num_samples=args.subgraph_num_samples,
            num_visualize=args.subgraph_num_visualize,
            output_dir=args.subgraph_output_dir,
            metric_min_available=args.subgraph_metric_min_available,
            metric_top_k=args.subgraph_metric_top_k,
            batch_size=args.subgraph_batch_size,
            device=args.subgraph_device,
        )

    write_run_manifest(config, command_name="full")
    return result


def run_eval_command(args):
    config = build_effective_config(args, checkpoint_path=args.checkpoint)
    results = evaluate_checkpoint(
        config,
        args.checkpoint,
        split_names=args.splits,
        use_selected_views=args.use_selected_views,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote evaluation metrics: {output_path}")
    return results


def run_fragment_diagnostics(
    processed_dir,
    splits=None,
    output_dir=None,
    max_molecules=None,
    no_plots=False,
    top_k=20,
):
    from scripts.fragmentation_diagnostics import (
        collect_diagnostics,
        default_output_dir,
        default_splits,
        make_plots,
        write_csv,
        write_json,
    )

    processed_dir = Path(processed_dir)
    splits = splits or default_splits(processed_dir)
    diagnostics = collect_diagnostics(
        processed_dir,
        splits,
        max_molecules=max_molecules,
        top_k=top_k,
    )
    output_dir = Path(output_dir) if output_dir else default_output_dir(diagnostics, processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    molecule_path = output_dir / "molecule_fragmentation_stats.csv"
    fragment_path = output_dir / "fragment_stats.csv"
    summary_path = output_dir / "split_summary.csv"
    metadata_path = output_dir / "metadata.json"
    write_csv(molecule_path, diagnostics["molecules"])
    write_csv(fragment_path, diagnostics["fragments"])
    write_csv(summary_path, diagnostics["summary"])
    write_json(metadata_path, diagnostics["metadata"])

    plot_paths = [] if no_plots else make_plots(diagnostics, output_dir, top_k=top_k)
    print(f"Analyzed splits: {', '.join(splits)}")
    print(f"Wrote fragmentation diagnostics: {output_dir}")
    if plot_paths:
        print(f"Wrote {len(plot_paths)} plot(s) under {output_dir / 'plots'}")
    return diagnostics


def run_loss_diagnostics(loss_log, output_dir):
    from scripts.plot_loss_logs import load_rows, make_plots, write_summary

    rows = load_rows(loss_log)
    output_dir = Path(output_dir or "reports/training_diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "final_epoch_summary.txt"
    write_summary(rows, summary_path)
    plot_paths = make_plots(rows, output_dir)
    print(f"Loaded {len(rows)} epoch row(s) from {loss_log}")
    print(f"Wrote loss diagnostics: {output_dir}")
    if plot_paths:
        print(f"Wrote {len(plot_paths)} plot(s)")


def run_subgraph_diagnostics(
    config=None,
    *,
    model_path=None,
    processed_dir=None,
    split=None,
    num_samples=None,
    num_visualize=None,
    output_dir=None,
    metric_min_available=None,
    metric_top_k=None,
    batch_size=None,
    device=None,
):
    from scripts.analyze_subgraphs import run_subgraph_analysis

    return run_subgraph_analysis(
        config or {},
        model_path=model_path,
        processed_dir=processed_dir,
        split=split,
        num_samples=num_samples,
        num_visualize=num_visualize,
        output_dir=output_dir,
        metric_min_available=metric_min_available,
        metric_top_k=metric_top_k,
        batch_size=batch_size,
        device=device,
    )


def run_inspect_command(args):
    from scripts.inspect_processed_pt import load_payload, paths_to_inspect, print_sectioned_record

    split_paths = paths_to_inspect(Path(args.path), args.splits)
    if not split_paths:
        raise FileNotFoundError(f"No .pt files found under {args.path}")
    for split_path in split_paths:
        payload = load_payload(split_path)
        print_sectioned_record(split_path, payload, args.index, args)


def add_common_arguments(parser):
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--processed-root", dest="processed_root")
    parser.add_argument("--processed-dir", dest="processed_dir")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--num-views", dest="num_views", type=int)
    parser.add_argument("--hidden-dim", dest="hidden_dim", type=int)
    parser.add_argument("--gnn-layers", dest="gnn_layers", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float)
    parser.add_argument("--lambda-contrast", dest="lambda_contrast", type=float)
    parser.add_argument("--lambda-diversity", dest="lambda_diversity", type=float)
    parser.add_argument("--lambda-irm", dest="lambda_irm", type=float)
    parser.add_argument("--num-envs", dest="num_envs", type=int)
    parser.add_argument(
        "--env-bucket-strategy",
        dest="env_bucket_strategy",
        choices=["value", "size"],
    )
    parser.add_argument("--save-model", dest="save_model")
    parser.add_argument("--loss-log-path", dest="loss_log_path")
    parser.add_argument("--final-metrics-path", dest="final_metrics_path")
    parser.add_argument("--output-root", dest="output_root")
    parser.add_argument("--run-name", dest="run_name")
    parser.add_argument("--subgraph-split", dest="subgraph_split")
    parser.add_argument("--subgraph-num-samples", dest="subgraph_num_samples", type=int)
    parser.add_argument("--subgraph-num-visualize", dest="subgraph_num_visualize", type=int)
    parser.add_argument(
        "--subgraph-metric-min-available",
        dest="subgraph_metric_min_available",
        type=int,
    )
    parser.add_argument("--subgraph-metric-top-k", dest="subgraph_metric_top_k", type=int)
    parser.add_argument("--train-split", dest="train_split")
    parser.add_argument("--validation-split", dest="validation_split")
    parser.add_argument("--test-split", dest="test_split")


def add_subgraph_diagnostic_arguments(parser, include_analysis_flag=False):
    if include_analysis_flag:
        parser.add_argument("--subgraph-analysis", action="store_true")
    parser.add_argument("--subgraph-model-path", dest="subgraph_model_path")
    parser.add_argument("--subgraph-processed-dir", dest="subgraph_processed_dir")
    parser.add_argument("--subgraph-output-dir", dest="subgraph_output_dir")
    parser.add_argument("--subgraph-batch-size", dest="subgraph_batch_size", type=int)
    parser.add_argument("--subgraph-device", dest="subgraph_device")


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Run dataset preprocessing.")
    add_common_arguments(preprocess)
    preprocess.add_argument("--validate-loaders", action="store_true")
    preprocess.set_defaults(func=run_preprocess_command)

    train = subparsers.add_parser("train", help="Train FragmentCIGA views and evaluate.")
    add_common_arguments(train)
    train.add_argument("--resume-from", dest="resume_from")
    train.set_defaults(func=run_train_command)

    resume = subparsers.add_parser("resume", help="Resume training from a checkpoint.")
    add_common_arguments(resume)
    resume.add_argument("checkpoint")
    resume.set_defaults(
        func=lambda args: run_training(
            build_effective_config(args, checkpoint_path=args.checkpoint),
            resume_from=args.checkpoint,
        )
    )

    evaluate = subparsers.add_parser("eval", help="Evaluate a saved checkpoint without training.")
    add_common_arguments(evaluate)
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--splits", nargs="+")
    evaluate.add_argument("--use-selected-views", action="store_true")
    evaluate.add_argument("--output")
    evaluate.set_defaults(func=run_eval_command)

    full = subparsers.add_parser("full", help="Run preprocessing, training, and optional diagnostics.")
    add_common_arguments(full)
    full.add_argument("--resume-from", dest="resume_from")
    full.add_argument("--skip-preprocess", action="store_true")
    full.add_argument("--validate-loaders", action="store_true")
    full.add_argument("--fragment-diagnostics", action="store_true")
    full.add_argument("--fragment-diagnostics-output")
    full.add_argument("--diagnostic-splits", nargs="+")
    full.add_argument("--max-molecules", type=int)
    full.add_argument("--top-k", type=int, default=20)
    full.add_argument("--no-plots", action="store_true")
    full.add_argument("--loss-plots", action="store_true")
    full.add_argument("--loss-log")
    full.add_argument("--loss-output-dir")
    add_subgraph_diagnostic_arguments(full, include_analysis_flag=True)
    full.set_defaults(func=run_full_command)

    diagnostics = subparsers.add_parser("diagnostics", help="Run report/diagnostic helpers.")
    diagnostic_subparsers = diagnostics.add_subparsers(dest="diagnostic_command", required=True)

    frag = diagnostic_subparsers.add_parser("fragments", help="Summarize processed fragment outputs.")
    frag.add_argument("processed_dir")
    frag.add_argument("--splits", nargs="+")
    frag.add_argument("--output-dir")
    frag.add_argument("--max-molecules", type=int)
    frag.add_argument("--top-k", type=int, default=20)
    frag.add_argument("--no-plots", action="store_true")
    frag.set_defaults(
        func=lambda args: run_fragment_diagnostics(
            args.processed_dir,
            splits=args.splits,
            output_dir=args.output_dir,
            max_molecules=args.max_molecules,
            no_plots=args.no_plots,
            top_k=args.top_k,
        )
    )

    losses = diagnostic_subparsers.add_parser("losses", help="Plot trainer CSV loss logs.")
    losses.add_argument("loss_log")
    losses.add_argument("--output-dir", default="reports/training_diagnostics")
    losses.set_defaults(func=lambda args: run_loss_diagnostics(args.loss_log, args.output_dir))

    subgraphs = diagnostic_subparsers.add_parser(
        "subgraphs",
        help="Analyze invariant subgraphs from a trained checkpoint.",
    )
    add_common_arguments(subgraphs)
    add_subgraph_diagnostic_arguments(subgraphs)
    subgraphs.set_defaults(
        func=lambda args: run_subgraph_diagnostics(
            config=build_effective_config(args),
            model_path=args.subgraph_model_path,
            processed_dir=args.subgraph_processed_dir,
            split=args.subgraph_split,
            num_samples=args.subgraph_num_samples,
            num_visualize=args.subgraph_num_visualize,
            output_dir=args.subgraph_output_dir,
            metric_min_available=args.subgraph_metric_min_available,
            metric_top_k=args.subgraph_metric_top_k,
            batch_size=args.subgraph_batch_size,
            device=args.subgraph_device,
        )
    )

    inspect = subparsers.add_parser("inspect", help="Inspect processed .pt files.")
    inspect.add_argument("path")
    inspect.add_argument("--splits", nargs="+")
    inspect.add_argument("--index", type=int, default=0)
    inspect.add_argument("--max-items", type=int, default=8)
    inspect.add_argument("--max-chars", type=int, default=220)
    inspect.add_argument("--metadata-json", action="store_true")
    inspect.add_argument("--full-lists", action="store_true")
    inspect.add_argument("--all-fields", action="store_true")
    inspect.set_defaults(func=run_inspect_command)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
