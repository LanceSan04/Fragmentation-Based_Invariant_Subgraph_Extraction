#!/usr/bin/env python
"""Train and evaluate SuGAR on configured molecular split files.

The script expects preprocessed ``.pt`` files under ``processed_dir``. Split
selection is config-driven: ``train_split`` is optimized, ``validation_split``
drives validation/checkpointing, ``test_split`` is the primary final report,
and ``report_splits`` can add side-by-side IID/OOD diagnostics.

Training flow:
  1. Create one FragmentCIGA model per view and train them independently.
  2. After training, wrap the trained views in SuGAREnsemble for inference.
  3. Evaluate the ensemble on the test set.
"""

import torch
import torch.nn.functional as F
import argparse
import yaml
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

try:
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from src.data.dataloader import build_split_dataloaders
from src.models.fragment_ciga import FragmentCIGA
from src.models.sugar_ensemble import SuGAREnsemble
from src.training.sugar_trainer import SuGARTrainer
from src.training.model_selection import greedy_view_selection, _accuracy_on_loader, _auc_on_loader


@dataclass
class Config:
    """Configuration object."""

    def __init__(self, config_dict=None):
        if config_dict:
            for key, val in config_dict.items():
                setattr(self, key, val)

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_config_and_cli(config_dict, args):
    """Use YAML as defaults and explicit CLI values as overrides."""
    merged = dict(config_dict or {})
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            merged[key] = value
    return merged


def build_config(args):
    """Build the effective training config from YAML defaults plus CLI overrides."""
    config_dict = load_config(args.config) if args.config else {}
    return Config(merge_config_and_cli(config_dict, args))


def build_shared_initialized_views(
    config,
    atom_feat_dim,
    fragment_feat_dim,
    atom_edge_dim,
    fragment_edge_dim,
):
    """Build FragmentCIGA views from one shared initial state."""
    num_views = config.num_views
    shared_view = FragmentCIGA(
        atom_feature_dim=atom_feat_dim,
        fragment_feature_dim=fragment_feat_dim,
        num_classes=config.num_classes,
        num_gnn_layers=config.gnn_layers,
        gnn_hidden_dim=config.hidden_dim,
        edge_ratio=config.edge_ratio,
        dropout=config.dropout,
        fragment_edge_dim=fragment_edge_dim,
        atom_edge_dim=atom_edge_dim,
        fragment_sample_ratio=config.fragment_sample_ratio,
        sampler_seed=config.sampler_seed,
    )
    shared_initial_state = deepcopy(shared_view.state_dict())

    views = []
    for i in range(num_views):
        view = FragmentCIGA(
            atom_feature_dim=atom_feat_dim,
            fragment_feature_dim=fragment_feat_dim,
            num_classes=config.num_classes,
            num_gnn_layers=config.gnn_layers,
            gnn_hidden_dim=config.hidden_dim,
            edge_ratio=config.edge_ratio,
            dropout=config.dropout,
            fragment_edge_dim=fragment_edge_dim,
            atom_edge_dim=atom_edge_dim,
            fragment_sample_ratio=config.fragment_sample_ratio,
            sampler_seed=(
                None
                if config.sampler_seed is None
                else int(config.sampler_seed) + i
            ),
        )
        view.load_state_dict(shared_initial_state)
        views.append(view)
    return views


@torch.no_grad()
def _evaluate_ensemble(ensemble, loader, device, track_roc_auc=True):
    """Evaluate a SuGAREnsemble on a data loader.

    Returns:
        avg_loss: Average NLL loss.
        accuracy: Classification accuracy.
        roc_auc: Macro-averaged ROC-AUC, or None if unavailable.
    """
    ensemble.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    compute_auc = track_roc_auc and _SKLEARN_AVAILABLE
    all_probs = [] if compute_auc else None
    all_labels_list = [] if compute_auc else None

    for batch in loader:
        batch = batch.to(device)
        predictions = ensemble(batch)  # log-probs [B, C]

        labels = batch.y
        if labels.dim() == 2:
            labels = labels.squeeze(-1)
        labels = labels.long()

        total_loss += F.nll_loss(predictions, labels).item()
        pred_labels = predictions.argmax(dim=-1)
        correct += (pred_labels == labels).sum().item()
        total += labels.size(0)
        num_batches += 1

        if compute_auc:
            all_probs.append(torch.softmax(predictions, dim=-1).cpu())
            all_labels_list.append(labels.cpu())

    avg_loss = total_loss / num_batches if num_batches else 0.0
    accuracy = correct / total if total else 0.0

    roc_auc = None
    if compute_auc:
        all_probs_t = torch.cat(all_probs, dim=0).numpy()
        all_labels_t = torch.cat(all_labels_list, dim=0).numpy()
        num_classes = all_probs_t.shape[1]
        try:
            if num_classes == 2:
                roc_auc = _sklearn_roc_auc(all_labels_t, all_probs_t[:, 1])
            else:
                roc_auc = _sklearn_roc_auc(
                    all_labels_t, all_probs_t, multi_class="ovr", average="macro"
                )
        except ValueError:
            roc_auc = None

    return avg_loss, accuracy, roc_auc


def main(args):
    """Main training function."""
    config = build_config(args)

    device_name = config.get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Using device: {device}")

    print(f"Loading data from {config.processed_dir}...")
    dataloaders = build_split_dataloaders(config.processed_dir, config)

    train_split = config.get("train_split", "train")
    validation_split = config.get("validation_split", config.get("valid_split", "valid"))
    test_split = config.get("test_split", "test")

    train_loader = dataloaders.get(train_split)
    val_loader = dataloaders.get(validation_split)
    test_loader = dataloaders.get(test_split)

    if train_loader is None:
        raise ValueError(f"No training data found for split {train_split!r} in {config.processed_dir}")

    keys_to_remove = ["fragment_to_atoms", "smiles", "molecule_id", "metadata"]
    for loader in dataloaders.values():
        if loader is not None and hasattr(loader, "dataset"):
            for data in loader.dataset:
                for key in keys_to_remove:
                    if hasattr(data, key):
                        delattr(data, key)
                    if key in data._store:
                        del data._store[key]

    # Infer feature dimensions from first batch
    print("Building FragmentCIGA views...")
    first_batch = next(iter(train_loader))
    atom_feat_dim = first_batch.x.size(1)
    fragment_feat_dim = (
        first_batch.fragment_x_static.size(1)
        if hasattr(first_batch, "fragment_x_static")
        else atom_feat_dim
    )
    atom_edge_dim = first_batch.edge_attr.size(1)
    fragment_edge_dim = first_batch.fragment_edge_attr.size(1)

    print(f"  Atom feature dim:     {atom_feat_dim}")
    print(f"  Fragment feature dim: {fragment_feat_dim}")
    print(f"  Atom edge dim:        {atom_edge_dim}")
    print(f"  Fragment edge dim:    {fragment_edge_dim}")

    # Create one FragmentCIGA model per view. All views share the same initial
    # weights, while their sampler seeds differ so training still sees different
    # fragment-edge drops across views.
    num_views = config.num_views
    views = build_shared_initialized_views(
        config,
        atom_feat_dim,
        fragment_feat_dim,
        atom_edge_dim,
        fragment_edge_dim,
    )
    print(f"  Created {num_views} FragmentCIGA view(s) with shared initialization")

    # Build trainer
    trainer = SuGARTrainer(
        views,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lambda_contrast=config.lambda_contrast,
        lambda_diversity=config.lambda_diversity,
        lambda_irm=config.lambda_irm,
        num_envs=config.num_envs,
        contrast_temperature=config.contrast_temperature,
        contrast_sampling=config.contrast_sampling,
        track_roc_auc=config.track_roc_auc,
        loss_log_path=config.get("loss_log_path", "logs/sugar_training_losses.csv"),
    )

    # Train — per-view best weights are tracked and restored automatically
    if val_loader is None and validation_split is not None:
        print(f"Validation split {validation_split!r} was not found; training without validation.")
    print(f"Starting training on split {train_split!r}...")
    history = trainer.train(
        train_loader,
        val_loader=val_loader,
        num_epochs=config.epochs,
        patience=config.patience,
        restore_best_weights=True,
    )

    # Wrap trained views in the ensemble for inference
    ensemble = SuGAREnsemble(
        trainer.views,
        aggregation_mode=config.get("aggregation_mode", config.get("aggregation", "mean")),
    )

    # Evaluate ensemble on test set
    if test_loader is not None:
        print(f"\nEvaluating ensemble on test split {test_split!r}...")
        test_loss, test_acc, test_auc = _evaluate_ensemble(
            ensemble, test_loader, device, track_roc_auc=config.track_roc_auc
        )
        auc_str = f" | {test_split} AUC: {test_auc:.4f}" if test_auc is not None else ""
        print(f"{test_split} Loss: {test_loss:.4f} | {test_split} Accuracy: {test_acc:.4f}{auc_str}")

    for split_name in config.get("report_splits", []) or []:
        if split_name in {validation_split, test_split}:
            continue
        loader = dataloaders.get(split_name)
        if loader is None:
            print(f"Report split {split_name!r} was not found; skipping.")
            continue
        loss, acc, auc = _evaluate_ensemble(
            ensemble, loader, device, track_roc_auc=config.track_roc_auc
        )
        auc_str = f" | {split_name} AUC: {auc:.4f}" if auc is not None else ""
        print(f"{split_name} Loss: {loss:.4f} | {split_name} Accuracy: {acc:.4f}{auc_str}")

    # Greedy view selection
    if config.get("greedy_model_selection", False) and val_loader is not None:
        print("\nRunning greedy view selection on validation set...")
        selected_views = greedy_view_selection(
            ensemble, val_loader, device, use_auc=trainer.track_roc_auc
        )

        if test_loader is not None:
            print("\nEvaluating greedy-selected ensemble on test set...")
            test_acc_greedy = _accuracy_on_loader(
                ensemble, test_loader, device, view_indices=selected_views
            )
            print(
                f"{test_split} Accuracy (greedy-selected views {selected_views}): "
                f"{test_acc_greedy:.4f}"
            )
            if trainer.track_roc_auc:
                test_auc_greedy = _auc_on_loader(
                    ensemble, test_loader, device, view_indices=selected_views
                )
                if test_auc_greedy is not None:
                    print(
                        f"{test_split} AUC (greedy-selected views {selected_views}): "
                        f"{test_auc_greedy:.4f}"
                    )

    # Save model — store each view's state dict keyed by view index
    if config.save_model:
        model_path = Path(config.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {i: v.state_dict() for i, v in enumerate(trainer.views)},
            model_path,
        )
        print(f"Model saved to {model_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SuGAR ensemble for molecular property prediction")

    # Data arguments
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--validation_split", type=str, default=None)
    parser.add_argument("--test_split", type=str, default=None)

    # Model arguments
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--gnn_layers", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--edge_ratio", type=float, default=None)
    parser.add_argument("--num_views", type=int, default=None)
    parser.add_argument(
        "--aggregation_mode", type=str, default=None, choices=["mean", "max"]
    )
    parser.add_argument("--fragment_sample_ratio", type=float, default=None)
    parser.add_argument("--sampler_seed", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lambda_contrast", type=float, default=None)
    parser.add_argument("--lambda_diversity", type=float, default=None)
    parser.add_argument("--lambda_irm", type=float, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--contrast_temperature", type=float, default=None)
    parser.add_argument(
        "--contrast_sampling", type=str, default=None, choices=["mul", "var", "cnc"]
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_model", type=str, default=None)
    parser.add_argument("--loss_log_path", type=str, default=None)
    parser.add_argument(
        "--track_roc_auc",
        type=lambda x: x.lower() not in ("0", "false", "no"),
        default=None,
        metavar="BOOL",
    )
    parser.add_argument(
        "--greedy_model_selection",
        type=lambda x: x.lower() not in ("0", "false", "no"),
        default=None,
        metavar="BOOL",
    )

    args = parser.parse_args()
    main(args)
