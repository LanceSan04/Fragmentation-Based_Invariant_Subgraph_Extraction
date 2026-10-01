#!/usr/bin/env python
"""Plot SuGAr/FragmentCIGA per-epoch loss CSV logs.

CLI arguments:
    loss_log
        Path to a CSV written by SuGARTrainer.

CLI options:
    --output-dir PATH
        Directory for summary text and plots. Default:
        reports/training_diagnostics.

Example:
    .\.venv\Scripts\python.exe scripts\plot_loss_logs.py ^
        outputs/logs/sugar_training_losses.csv ^
        --output-dir reports\training_diagnostics
"""

import argparse
import csv
from pathlib import Path


def load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Loss log has no rows: {path}")
    return rows


def numeric_series(rows, column):
    values = [to_float(row.get(column)) for row in rows]
    return values if any(value is not None for value in values) else None


def pipe_series(rows, column):
    parsed = []
    max_len = 0
    for row in rows:
        raw = row.get(column, "")
        if raw == "":
            values = []
        else:
            values = [to_float(part) for part in raw.split("|")]
        parsed.append(values)
        max_len = max(max_len, len(values))
    if max_len == 0:
        return []
    series = []
    for index in range(max_len):
        series.append([values[index] if index < len(values) else None for values in parsed])
    return series


def series_values(series):
    values = []
    for _, maybe_values in series:
        if maybe_values is None:
            continue
        values.extend(value for value in maybe_values if value is not None)
    return values


def needs_symlog_scale(series, ratio_threshold=100.0):
    values = [abs(value) for value in series_values(series) if value != 0]
    if not values:
        return False
    return max(values) / min(values) >= ratio_threshold


def plot_lines(plt, epochs, series, title, ylabel, output_path, yscale=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, values in series:
        if values is None:
            continue
        ax.plot(epochs, values, marker="o", linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_auxiliary_group(plt, epochs, series, title, ylabel, output_path):
    """Plot combined auxiliary losses, using symlog when scales differ wildly."""
    plot_lines(
        plt,
        epochs,
        series,
        title,
        ylabel,
        output_path,
        yscale="symlog" if needs_symlog_scale(series) else None,
    )


def write_summary(rows, output_path):
    final = rows[-1]
    columns = [
        "epoch",
        "train_loss",
        "train_prediction_loss",
        "train_contrastive_loss_weighted",
        "train_diversity_loss_weighted",
        "train_irm_loss_weighted",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_auc",
        "val_selection_metric",
        "val_selection_metric_name",
        "val_acc_per_view",
        "val_auc_per_view",
    ]
    output_path.write_text(
        "\n".join(f"{column}: {final.get(column, '')}" for column in columns) + "\n",
        encoding="utf-8",
    )


def make_plots(rows, output_dir):
    plt = load_matplotlib()
    if plt is None:
        print("matplotlib is not installed; skipping PNG plots.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [int(float(row["epoch"])) for row in rows]
    written = []

    path = output_dir / "total_loss.png"
    plot_lines(
        plt,
        epochs,
        [
            ("train_loss", numeric_series(rows, "train_loss")),
            ("val_loss", numeric_series(rows, "val_loss")),
        ],
        "Total Loss",
        "loss",
        path,
    )
    written.append(path)

    path = output_dir / "prediction_loss.png"
    plot_lines(
        plt,
        epochs,
        [
            ("train_prediction_loss", numeric_series(rows, "train_prediction_loss")),
            ("val_prediction_loss", numeric_series(rows, "val_prediction_loss")),
        ],
        "Prediction Loss",
        "loss",
        path,
    )
    written.append(path)

    weighted_auxiliary_series = [
        ("train_contrastive_weighted", numeric_series(rows, "train_contrastive_loss_weighted")),
        ("train_diversity_weighted", numeric_series(rows, "train_diversity_loss_weighted")),
        ("train_irm_weighted", numeric_series(rows, "train_irm_loss_weighted")),
        ("val_contrastive_weighted", numeric_series(rows, "val_contrastive_loss_weighted")),
        ("val_diversity_weighted", numeric_series(rows, "val_diversity_loss_weighted")),
        ("val_irm_weighted", numeric_series(rows, "val_irm_loss_weighted")),
    ]
    path = output_dir / "weighted_auxiliary_losses.png"
    plot_auxiliary_group(
        plt,
        epochs,
        weighted_auxiliary_series,
        "Weighted Auxiliary Loss Contributions",
        "weighted loss (symlog if needed)",
        path,
    )
    written.append(path)

    raw_auxiliary_series = [
        ("train_contrastive", numeric_series(rows, "train_contrastive_loss")),
        ("train_diversity", numeric_series(rows, "train_diversity_loss")),
        ("train_irm", numeric_series(rows, "train_irm_loss")),
        ("val_contrastive", numeric_series(rows, "val_contrastive_loss")),
        ("val_diversity", numeric_series(rows, "val_diversity_loss")),
        ("val_irm", numeric_series(rows, "val_irm_loss")),
    ]
    path = output_dir / "raw_auxiliary_losses.png"
    plot_auxiliary_group(
        plt,
        epochs,
        raw_auxiliary_series,
        "Raw Auxiliary Losses",
        "raw loss (symlog if needed)",
        path,
    )
    written.append(path)

    auxiliary_specs = [
        (
            "contrastive_loss.png",
            "Contrastive Loss",
            [
                ("train_contrastive", numeric_series(rows, "train_contrastive_loss")),
                ("val_contrastive", numeric_series(rows, "val_contrastive_loss")),
            ],
        ),
        (
            "diversity_loss.png",
            "Diversity Loss",
            [
                ("train_diversity", numeric_series(rows, "train_diversity_loss")),
                ("val_diversity", numeric_series(rows, "val_diversity_loss")),
            ],
        ),
        (
            "irm_loss.png",
            "IRM Loss",
            [
                ("train_irm", numeric_series(rows, "train_irm_loss")),
                ("val_irm", numeric_series(rows, "val_irm_loss")),
            ],
        ),
        (
            "weighted_contrastive_loss.png",
            "Weighted Contrastive Loss Contribution",
            [
                (
                    "train_contrastive_weighted",
                    numeric_series(rows, "train_contrastive_loss_weighted"),
                ),
                ("val_contrastive_weighted", numeric_series(rows, "val_contrastive_loss_weighted")),
            ],
        ),
        (
            "weighted_diversity_loss.png",
            "Weighted Diversity Loss Contribution",
            [
                ("train_diversity_weighted", numeric_series(rows, "train_diversity_loss_weighted")),
                ("val_diversity_weighted", numeric_series(rows, "val_diversity_loss_weighted")),
            ],
        ),
        (
            "weighted_irm_loss.png",
            "Weighted IRM Loss Contribution",
            [
                ("train_irm_weighted", numeric_series(rows, "train_irm_loss_weighted")),
                ("val_irm_weighted", numeric_series(rows, "val_irm_loss_weighted")),
            ],
        ),
    ]
    for filename, title, series in auxiliary_specs:
        path = output_dir / filename
        plot_lines(plt, epochs, series, title, "loss", path)
        written.append(path)

    path = output_dir / "accuracy_auc.png"
    plot_lines(
        plt,
        epochs,
        [
            ("train_acc", numeric_series(rows, "train_acc")),
            ("val_acc", numeric_series(rows, "val_acc")),
            ("val_auc", numeric_series(rows, "val_auc")),
        ],
        "Accuracy And AUC",
        "metric",
        path,
    )
    written.append(path)

    view_auc_series = pipe_series(rows, "val_auc_per_view")
    if view_auc_series:
        path = output_dir / "per_view_val_auc.png"
        plot_lines(
            plt,
            epochs,
            [(f"view_{index}_auc", values) for index, values in enumerate(view_auc_series)],
            "Per-View Validation AUC",
            "AUC",
            path,
        )
        written.append(path)

    view_acc_series = pipe_series(rows, "val_acc_per_view")
    if view_acc_series:
        path = output_dir / "per_view_val_accuracy.png"
        plot_lines(
            plt,
            epochs,
            [(f"view_{index}_acc", values) for index, values in enumerate(view_acc_series)],
            "Per-View Validation Accuracy",
            "accuracy",
            path,
        )
        written.append(path)

    return written


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("loss_log", help="Path to a CSV written by SuGARTrainer.")
    parser.add_argument("--output-dir", default="reports/training_diagnostics")
    args = parser.parse_args()

    rows = load_rows(args.loss_log)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "final_epoch_summary.txt"
    write_summary(rows, summary_path)
    plot_paths = make_plots(rows, output_dir)

    print(f"Loaded {len(rows)} epoch row(s) from {args.loss_log}")
    print(f"Wrote final summary: {summary_path}")
    if plot_paths:
        print(f"Wrote {len(plot_paths)} plot(s) under {output_dir}")


if __name__ == "__main__":
    main()
