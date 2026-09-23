"""Shared frozen benchmark configuration and synthetic experiment fixtures."""
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]

def make_plan(output, device, smoke=False):
    base = yaml.safe_load((ROOT / "configs/multiview_irm.yaml").read_text(encoding="utf-8"))
    base.update(processed_dir=str(output / "processed" if smoke else ROOT / "data/processed/overnight" / Path(output).name), output_root="",
                split_names=["train", "ood_val", "ood_test"], test_split=None,
                report_splits=[], device=device, debug=False)
    spec = dict(output_dir=str(output), study_name="factorial_losses_v2", sampler="grid",
                grid={"lambda_contrast": [0.0, 0.1], "lambda_diversity": [0.0, 0.5],
                      "lambda_irm": [0.0, 0.5, 1.0, 2.0]}, n_trials=16,
                search_repeats=2, common_search_seeds=True, base_seed=101,
                sampler_seed=7, auc_ma_window=1, maximize=True, eval_runs=3,
                eval_base_seed=1001, test_split="ood_test", retain_eval_artifacts=True,
                overrides={"epochs": 100})
    if smoke:
        molecules = ["CCOC(=O)C", "c1ccccc1CC(=O)NC", "CCN(CC)CC", "CCCOC"]
        splits = {name: [dict(smiles=smi, label=i % 2, domain_id=i // 2,
                             molecule_id=f"{name}_{i}") for i, smi in enumerate(molecules)]
                  for name in base["split_names"]}
        base.update(dataset="synthetic_overnight", raw_path=None, raw_format=None,
                    splits=splits, label_key="label", strict=True, batch_size=4,
                    num_workers=0, pin_memory=False, hidden_dim=8, gnn_layers=1)
        spec["grid"]["lambda_irm"] = [0.0, 2.0]
        spec.update(n_trials=8, eval_runs=2,
                    overrides={"epochs": 2})
    return spec, base


def base_config(dataset, cache, device):
    base = yaml.safe_load((ROOT/'configs/parameter_winner_test_base.yaml').read_text())
    if dataset != 'assay':
        base.update(dataset=f'drugood_lbap_core_ec50_{dataset}',
                    raw_path=str(ROOT/f'data/raw/drugood/lbap_core_ec50_{dataset}.json'),
                    processed_dir=str(cache))
    base.update(device=device, loader_split_names=['train', 'ood_val', 'ood_test'],
                split_names=['train', 'ood_val', 'ood_test'], test_split='ood_test',
                log_test_curve=False, report_splits=[])
    return base



def workflow_defaults():
    """Load versioned workflow budgets and search spaces."""
    return yaml.safe_load((ROOT/"configs/workflows.yaml").read_text())
