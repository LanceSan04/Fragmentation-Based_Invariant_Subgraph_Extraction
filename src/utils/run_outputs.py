"""Resolve and record consolidated experiment output directories."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def config_to_dict(config):
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    return {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_")
    }


def set_config_value(config, key, value):
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


def get_config_value(config, key, default=None):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def safe_path_part(value, fallback="unknown"):
    text = str(value or "").strip()
    if not text:
        text = fallback
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in text)


def auto_run_name(config, command_name="run"):
    dataset = safe_path_part(get_config_value(config, "dataset", "dataset"))
    fragmentation = safe_path_part(get_config_value(config, "fragmentation", "fragmentation"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset}__{fragmentation}__{command_name}__{timestamp}"


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def resolve_run_directory(config, command_name="run"):
    """Mutate config with run_dir and default consolidated output paths."""
    output_root = get_config_value(config, "output_root", None)
    if not output_root:
        return None

    dataset = safe_path_part(get_config_value(config, "dataset", "dataset"))
    fragmentation = safe_path_part(get_config_value(config, "fragmentation", "fragmentation"))
    run_name = get_config_value(config, "run_name", None)
    if not run_name or str(run_name).lower() == "auto":
        run_name = auto_run_name(config, command_name=command_name)

    run_dir = Path(output_root) / dataset / fragmentation / safe_path_part(run_name)
    set_config_value(config, "run_name", run_name)
    set_config_value(config, "run_dir", str(run_dir))

    explicit = set(get_config_value(config, "_explicit_output_paths", []) or [])
    default_paths = {
        "save_model": run_dir / "checkpoints" / "best_model.pt",
        "loss_log_path": run_dir / "training_epoch_losses.csv",
        "final_metrics_path": run_dir / "final_evaluation_metrics.json",
        "training_diagnostics_dir": run_dir / "training_curves",
        "fragmentation_diagnostics_dir": run_dir / "preprocessing_fragmentation_diagnostics",
        "subgraph_analysis_dir": run_dir / "subgraph_analysis",
    }
    for key, path in default_paths.items():
        if key not in explicit:
            set_config_value(config, key, str(path))

    return run_dir


def write_effective_config(config):
    run_dir = get_config_value(config, "run_dir", None)
    if not run_dir:
        return None
    path = Path(run_dir) / "config" / "effective_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config_to_dict(config)
    data.pop("_explicit_output_paths", None)
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    return path


def build_manifest(config, command_name="run", extra=None):
    data = config_to_dict(config)
    manifest = {
        "command": command_name,
        "dataset": data.get("dataset"),
        "fragmentation": data.get("fragmentation"),
        "run_name": data.get("run_name"),
        "run_dir": data.get("run_dir"),
        "processed_dir": data.get("processed_dir"),
        "train_split": data.get("train_split"),
        "validation_split": data.get("validation_split"),
        "test_split": data.get("test_split"),
        "report_splits": data.get("report_splits"),
        "num_views": data.get("num_views"),
        "checkpoint_path": data.get("save_model"),
        "loss_log_path": data.get("loss_log_path"),
        "final_metrics_path": data.get("final_metrics_path"),
        "training_diagnostics_dir": data.get("training_diagnostics_dir"),
        "fragmentation_diagnostics_dir": data.get("fragmentation_diagnostics_dir"),
        "subgraph_analysis_dir": data.get("subgraph_analysis_dir"),
        "git_commit": git_commit(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_run_manifest(config, command_name="run", extra=None):
    run_dir = get_config_value(config, "run_dir", None)
    if not run_dir:
        return None
    path = Path(run_dir) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            build_manifest(config, command_name=command_name, extra=extra),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def prepare_run_outputs(config, command_name="run"):
    run_dir = resolve_run_directory(config, command_name=command_name)
    if run_dir is None:
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    write_effective_config(config)
    write_run_manifest(config, command_name=command_name)
    return run_dir
