#!/usr/bin/env python
"""Main training script for SuGAR ensemble on molecular property prediction."""

import torch
import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass

from src.data.dataloader import build_split_dataloaders
from src.models.sugar_ensemble import SuGAREnsemble
from src.training.sugar_trainer import SuGARTrainer
from src.training.model_selection import greedy_view_selection, _accuracy_on_loader


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


def main(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Load configuration. YAML values provide defaults; explicit CLI values
    # override those defaults.
    config = build_config(args)

    # Build dataloaders
    print(f"Loading data from {config.processed_dir}...")
    dataloaders = build_split_dataloaders(config.processed_dir, config)

    train_loader = dataloaders.get("train")
    val_loader = dataloaders.get("valid")
    test_loader = dataloaders.get("test")

    if train_loader is None:
        raise ValueError(f"No training data found in {config.processed_dir}")

    keys_to_remove = ["fragment_to_atoms", "smiles", "molecule_id"]
    
    for loader in [train_loader, val_loader, test_loader]:
        if loader is not None and hasattr(loader, 'dataset'):
            for data in loader.dataset:
                for key in keys_to_remove:
                    if hasattr(data, key):
                        delattr(data, key)
                    if key in data._store:
                        del data._store[key]
    # Build model
    print("Building SuGAR ensemble model...")
    
    # Infer feature dimensions from first batch
    first_batch = next(iter(train_loader))
    atom_feat_dim = first_batch.x.size(1)
    fragment_feat_dim = first_batch.fragment_x_static.size(1) if hasattr(first_batch, 'fragment_x_static') else atom_feat_dim
    atom_edge_dim = first_batch.edge_attr.size(1)
    fragment_edge_dim = first_batch.fragment_edge_attr.size(1)
    
    print(f"  Atom feature dimension: {atom_feat_dim}")
    print(f"  Fragment feature dimension: {fragment_feat_dim}")
    print(f"  Atom edge feature dimension: {atom_edge_dim}")
    print(f"  Fragment edge feature dimension: {fragment_edge_dim}")
    
    model = SuGAREnsemble(
        atom_feature_dim=atom_feat_dim,
        fragment_feature_dim=fragment_feat_dim,
        num_classes=config.num_classes,
        num_gnn_layers=config.gnn_layers,
        gnn_hidden_dim=config.hidden_dim,
        edge_ratio=config.edge_ratio,
        num_views=config.num_views,
        dropout=config.dropout,
        fragment_edge_dim=fragment_edge_dim,
        atom_edge_dim=atom_edge_dim,
        fragment_sample_ratio=config.fragment_sample_ratio,
        sampler_seed=config.sampler_seed,
    )

    # Build trainer
    trainer = SuGARTrainer(
        model,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lambda_contrast=config.lambda_contrast,
        lambda_diversity=config.lambda_diversity,
        lambda_irm=config.lambda_irm,
        num_envs=config.num_envs,
        contrast_temperature=config.contrast_temperature,
        contrast_sampling=config.contrast_sampling,
    )

    # Train
    print("Starting training...")
    history = trainer.train(
        train_loader,
        val_loader=val_loader,
        num_epochs=config.epochs,
        patience=config.patience,
    )

    # Evaluate on test set
    if test_loader is not None:
        print("\nEvaluating on test set...")
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Greedy view selection; requires num_views > 1 and a validation set to score candidate subsets.
    if config.get("greedy_model_selection", False) and val_loader is not None:
        print("\nRunning greedy view selection on validation set...")
        selected_views = greedy_view_selection(model, val_loader, device)

        if test_loader is not None:
            print("\nEvaluating greedy-selected ensemble on test set...")
            test_acc_greedy = _accuracy_on_loader(model, test_loader, device, view_indices=selected_views)
            print(f"Test Accuracy (greedy-selected views {selected_views}): {test_acc_greedy:.4f}")

    # Save model
    if config.save_model:
        model_path = Path(config.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SuGAR ensemble for molecular property prediction")

    # Data arguments
    parser.add_argument("--processed_dir", type=str, default=None, help="Directory with processed data")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of data loading workers")

    # Model arguments
    parser.add_argument("--num_classes", type=int, default=None, help="Number of output classes")
    parser.add_argument("--gnn_layers", type=int, default=None, help="Number of GNN layers")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--edge_ratio", type=float, default=None, help="Ratio of edges to keep in invariant subgraph")
    parser.add_argument("--num_views", type=int, default=None, help="Number of SuGAr views/base models")
    parser.add_argument(
        "--fragment_sample_ratio",
        type=float,
        default=None,
        help="Training-time SuGAr sampler ratio for eligible fragment edges",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=None,
        help="Optional seed for reproducible fragment-edge sampling views",
    )
    parser.add_argument("--dropout", type=float, default=None, help="Dropout rate")

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--lambda_contrast", type=float, default=None, help="Supervised contrastive loss weight")
    parser.add_argument("--lambda_diversity", type=float, default=None, help="Pairwise diversity loss weight")
    parser.add_argument("--lambda_irm", type=float, default=None, help="IRM penalty weight")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environment partitions for IRM")
    parser.add_argument("--contrast_temperature", type=float, default=None, help="Supervised contrastive temperature")
    parser.add_argument(
        "--contrast_sampling",
        type=str,
        default=None,
        choices=["mul", "var", "cnc"],
        help="Supervised contrastive sampling mode",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--save_model", type=str, default=None, help="Path to save trained model")

    args = parser.parse_args()
    main(args)
