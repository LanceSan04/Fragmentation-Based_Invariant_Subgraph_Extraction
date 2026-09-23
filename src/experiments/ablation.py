#!/usr/bin/env python
"""Staged validation-only runs. Review the views stage before running auxiliary.

python scripts/run_multiview_ablation.py views --dry-run
python scripts/run_multiview_ablation.py views
python scripts/run_multiview_ablation.py auxiliary
"""
import argparse
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
STAGES = {
    "views": ["multiview_prediction"],
    "auxiliary": ["multiview_diversity", "multiview_contrastive", "multiview_irm"],
}


def build_commands(stage, device="cuda"):
    commands = []
    for name in STAGES[stage]:
        config_path = ROOT / "configs" / f"{name}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if (config.get("test_split") is not None or config.get("report_splits") or
                config.get("log_test_curve") or config.get("greedy_model_selection") or
                config.get("selection_ma_window") != 1 or config.get("early_stopping") is not False or
                config.get("selection_weight_avg") or config.get("compare_selection_policies")):
            raise ValueError(f"Unexpected experiment protocol in {config_path}")
        for seed in (11, 22, 33):
            run_name = f"{name}_s{seed}"
            destination = ROOT / config["output_root"] / config["dataset"] / "group_graph" / run_name
            commands.append((destination, [
                sys.executable, str(ROOT / "main.py"), "train", "--config", str(config_path),
                "--device", device, "--epochs", "100", "--selection-ma-window", "1",
                "--seed", str(seed), "--sampler-seed", str(seed), "--run-name", run_name,
            ]))
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    commands = build_commands(args.stage, args.device)
    print(f"Stage {args.stage}: {len(commands)} runs, 100 epochs each. Test evaluation disabled.")
    for destination, command in commands:
        print(" ".join(command))
        if not args.dry_run and destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}; use a new output_root for a rerun")
    if args.dry_run:
        return
    for _, command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
