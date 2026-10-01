"""Record resolved execution choices without environment variables or secrets."""

from importlib.metadata import version, PackageNotFoundError
import platform
import torch


def runtime_spec(config, views, trainer, policies):
    versions = {}
    for package in ["torch", "torch-geometric", "numpy", "scikit-learn", "rdkit"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return dict(
        config=config.to_dict(),
        python=platform.python_version(),
        packages=versions,
        device=str(trainer.device),
        torch_threads=torch.get_num_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        checkpoint_selection_policies=list(policies),
        evaluation_policies=list(policies),
        topk_tie_policy="stable_original_edge_order",
        gradient_clip_max_norm=1.0,
        views=[
            dict(
                scoring_mode=v.fragment_masker.scoring_mode,
                selection_mode=v.selection_mode,
                erm=v.erm_baseline,
                selection_budget=v.selection_budget,
                ranking_mode=v.ranking_mode,
                atom_retention_ratio=v.atom_retention_ratio,
                fragmentation_method=v.fragmentation_method,
                edge_ratio=v.fragment_masker.ratio,
                fragment_sample_ratio=v.fragment_edge_sampler.keep_ratio,
                sampler_seed=v.fragment_edge_sampler.seed,
            )
            for v in views
        ],
        optimizers=[
            dict(
                type=type(o).__name__,
                groups=[{k: v for k, v in g.items() if k != "params"} for g in o.param_groups],
            )
            for o in trainer.optimizers
        ],
    )
