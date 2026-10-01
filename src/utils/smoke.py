#!/usr/bin/env python
"""Small synthetic end-to-end check; no DrugOOD files are read or changed.

Run: python -m src.utils.smoke [--device cpu|cuda]
Temporary preprocessing/checkpoints are removed on completion.
The labels are artificial: AUC here is an execution check, not a benchmark.
"""

import argparse
import math
import tempfile
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import torch
import yaml
from src.data.preprocess import run_preprocessing
from src.training.pipeline import Config, run_training, evaluate_checkpoint


def run_smoke(device="cpu", compare_selection=False):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Python environment")
    torch.set_num_threads(1)
    base = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    molecules = {
        "train": ["CCOC(=O)C", "c1ccccc1CC(=O)NC", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CCN(CC)CC"],
        "valid": ["CCOC(=O)CC", "CCc1ccccc1"],
        "test": ["CC(=O)Nc1ccccc1", "CCCOC"],
    }
    splits = {
        name: [
            dict(smiles=smi, label=i % 2, domain_id=i // 2, molecule_id=f"{name}_{i}")
            for i, smi in enumerate(smiles)
        ]
        for name, smiles in molecules.items()
    }
    cases = [
        ("erm", True, {"group_graph": 1}),
        ("fragment", False, {"group_graph": 2}),
        ("mixed", False, {"group_graph": 1, "brics": 1}),
    ]
    summaries = []
    with tempfile.TemporaryDirectory(prefix="reu_smoke_") as directory:
        for name, erm, recipe in cases:
            values = deepcopy(base)
            values.update(
                dataset="synthetic_smoke",
                raw_path=None,
                splits=splits,
                raw_format=None,
                label_key="label",
                environment_key="domain_id",
                split_names=list(splits),
                train_split="train",
                validation_split="valid",
                test_split="test",
                report_splits=[],
                output_root="",
                processed_dir=str(Path(directory) / name / "processed"),
                save_model=str(Path(directory) / name / "model.pt"),
                final_metrics_path=None,
                loss_log_path=str(Path(directory) / name / "loss.csv"),
                debug=False,
                strict=True,
                num_workers=0,
                pin_memory=False,
                num_views=sum(recipe.values()),
                view_fragmentations=recipe,
                erm_baseline=erm,
                hidden_dim=16,
                gnn_layers=2,
                dropout=0.0,
                batch_size=4,
                epochs=2,
                patience=2,
                selection_ma_window=2,
                selection_weight_avg=False,
                log_test_curve=False,
                greedy_model_selection=False,
                fragment_sample_ratio=0.8,
                lambda_contrast=0.0 if erm else 0.1,
                lambda_diversity=0.0 if erm else 0.1,
                lambda_irm=0.0 if erm else 0.1,
                num_envs=1 if erm else 2,
                seed=17,
                sampler_seed=17,
                device=device,
            )
            if compare_selection:
                values.update(
                    compare_selection_policies=True,
                    test_split=None,
                    output_root=str(Path(directory) / "outputs"),
                    run_name=name,
                )
            cfg = Config(values)
            processed = run_preprocessing(cfg)
            assert not any(processed["failures"].values()), processed["failures"]
            result = run_training(cfg)
            history = result["history"]
            assert len(history["train_loss"]) == 2
            assert history["selected_checkpoint_epoch"] == 2
            assert history["val_selection_metric_smoothed"][0] is None
            assert all(math.isfinite(v) for v in history["train_loss"])
            assert all(optimizer.state for optimizer in result["trainer"].optimizers)
            restored = evaluate_checkpoint(
                Config(cfg.to_dict()),
                cfg.save_model,
                ["valid"] if compare_selection else ["valid", "test"],
            )
            for split in restored:
                for metric in ("loss", "acc", "auc"):
                    before, after = result["metrics"][split][metric], restored[split][metric]
                    assert before is not None and math.isfinite(before), (split, metric, before)
                    assert abs(before - after) < 1e-6, (name, split, metric, before, after)
            if compare_selection:
                import json

                report = json.loads(
                    (Path(cfg.run_dir) / "selection_comparison/comparison.json").read_text()
                )
                assert report["completed_epochs"] == 2
                assert set(report["policies"]) == {"best_raw", "best_ma_endpoint"}
                for policy in report["policies"].values():
                    evaluated = evaluate_checkpoint(
                        Config(cfg.to_dict()), policy["checkpoint"], ["valid"]
                    )
                    assert (
                        abs(evaluated["valid"]["auc"] - policy["validation_auc_at_selection"])
                        < 1e-5
                    )
            summaries.append(
                f"{name}: preprocessing, 2 training epochs, evaluation and checkpoint parity PASS"
            )
    print("\n" + "\n".join(summaries))
    print(f"Device: {device}; synthetic labels only; no benchmark claim.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--compare-selection",
        action="store_true",
        help="Also test both fixed-budget checkpoint policies",
    )
    args = parser.parse_args(argv)
    run_smoke(args.device, compare_selection=args.compare_selection)


if __name__ == "__main__":
    main()
