#!/usr/bin/env python
"""Main training script for SuGAR ensemble on molecular property prediction."""

import torch
import torch.nn as nn
import argparse
import yaml
from pathlib import Path
from dataclasses import dataclass

from src.data.dataloader import build_split_dataloaders
from src.models.sugar_ensemble import SuGAREnsemble
from src.training.sugar_trainer import SuGARTrainer


@dataclass
class Config:
    """Configuration object."""

    def __init__(self, args=None, config_dict=None):
        if args:
            self.processed_dir = args.processed_dir
            self.batch_size = args.batch_size
            self.num_workers = args.num_workers
        if config_dict:
            for key, val in config_dict.items():
                setattr(self, key, val)

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    if args.config:
        config_dict = load_config(args.config)
        config = Config(config_dict=config_dict)
    else:
        config = Config(args=args)

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
        num_classes=args.num_classes,
        num_gnn_layers=args.gnn_layers,
        gnn_hidden_dim=args.hidden_dim,
        edge_ratio=args.edge_ratio,
        num_views=1,  # Single view for now
        dropout=args.dropout,
        fragment_edge_dim=fragment_edge_dim,
        atom_edge_dim=atom_edge_dim,
    )

    # Build trainer
    trainer = SuGARTrainer(
        model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Train
    print("Starting training...")
    history = trainer.train(
        train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        patience=args.patience,
    )

    # Evaluate on test set
    if test_loader is not None:
        print("\nEvaluating on test set...")
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    # Save model
    if args.save_model:
        model_path = Path(args.save_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SuGAR ensemble for molecular property prediction")

    # Data arguments
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Directory with processed data")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")

    # Model arguments
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes")
    parser.add_argument("--gnn_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--edge_ratio", type=float, default=0.5, help="Ratio of edges to keep in invariant subgraph")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save_model", type=str, default=None, help="Path to save trained model")

    args = parser.parse_args()
    main(args)
