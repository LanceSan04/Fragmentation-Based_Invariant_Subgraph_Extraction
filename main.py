#!/usr/bin/env python
"""Consolidated CLI for preprocessing, training, checkpoints, and diagnostics."""

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.data.dataloader import build_split_dataloaders, validate_dataloader_batch
from train_sugar import Config, evaluate_checkpoint, run_training


ROOT = Path(__file__).resolve().parent


COMMON_OVERRIDES = (
    "dataset",
    "fragmentation",
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
    "save_model",
    "loss_log_path",
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
    old_processed_dir = merged.get("processed_dir")

    for key in COMMON_OVERRIDES:
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value

    if (
        getattr(args, "fragmentation", None)
        and not explicit_processed_dir
        and old_processed_dir
    ):
        merged["processed_dir"] = str(Path(old_processed_dir).parent / args.fragmentation)
        merged.pop("failure_report_path", None)
        print(
            "No --processed-dir supplied; using fragmenter-specific processed_dir: "
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
    result = run_preprocessing(config.to_dict())
    if args.validate_loaders:
        validate_loaders(config.processed_dir, config)
    return result


def run_train_command(args):
    config = build_effective_config(args, checkpoint_path=getattr(args, "resume_from", None))
    return run_training(config, resume_from=getattr(args, "resume_from", None))


def run_full_command(args):
    from src.data.preprocess import run_preprocessing

    config = build_effective_config(args, checkpoint_path=getattr(args, "resume_from", None))
    if not args.skip_preprocess:
        run_preprocessing(config.to_dict())
        if args.validate_loaders:
            validate_loaders(config.processed_dir, config)

    result = run_training(config, resume_from=getattr(args, "resume_from", None))

    if args.fragment_diagnostics:
        run_fragment_diagnostics(
            processed_dir=Path(config.processed_dir),
            splits=args.diagnostic_splits or config.get("split_names", None),
            output_dir=Path(args.fragment_diagnostics_output) if args.fragment_diagnostics_output else None,
            max_molecules=args.max_molecules,
            no_plots=args.no_plots,
            top_k=args.top_k,
        )

    if args.loss_plots:
        loss_log = args.loss_log or config.get("loss_log_path", None)
        if loss_log:
            run_loss_diagnostics(loss_log, args.loss_output_dir)
        else:
            print("No loss log path configured; skipping loss plots.")

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
    parser.add_argument("--fragmentation")
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
    parser.add_argument("--save-model", dest="save_model")
    parser.add_argument("--loss-log-path", dest="loss_log_path")
    parser.add_argument("--train-split", dest="train_split")
    parser.add_argument("--validation-split", dest="validation_split")
    parser.add_argument("--test-split", dest="test_split")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
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
    full.add_argument("--loss-output-dir", default="reports/training_diagnostics")
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
