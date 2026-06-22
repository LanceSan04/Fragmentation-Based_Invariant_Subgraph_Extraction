"""Training pipeline for SuGAr ensemble with multi-view fragment learning."""

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm


class SuGARTrainer:
    """Trainer for SuGAR ensemble model."""

    def __init__(
        self,
        model,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
    ):
        """
        Args:
            model: SuGAREnsemble model
            device: Device to train on (cpu/cuda)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader):
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader with batches of MoleculeData
            
        Returns:
            avg_loss: Average training loss for the epoch
            accuracy: Training classification accuracy for the epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)

            # Forward pass
            predictions = self.model(batch)

            # Get labels
            labels = batch.y
            if labels.dim() == 2:
                labels = labels.squeeze(-1)
            labels = labels.long()

            # Compute loss
            loss = self.loss_fn(predictions, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
            # Compute running accuracy
            pred_labels = predictions.argmax(dim=-1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar with both loss and live accuracy
            current_acc = correct / total if total > 0 else 0.0
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.4f}"})

        avg_loss = total_loss / num_batches
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, eval_loader):
        """Evaluate on validation/test set.
        
        Args:
            eval_loader: DataLoader with batches of MoleculeData
            
        Returns:
            avg_loss: Average loss
            accuracy: Classification accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        for batch in eval_loader:
            batch = batch.to(self.device)

            # Forward pass
            predictions = self.model(batch)

            # Get labels
            labels = batch.y
            if labels.dim() == 2:
                labels = labels.squeeze(-1)
            labels = labels.long()

            # Compute loss
            loss = self.loss_fn(predictions, labels)
            total_loss += loss.item()

            # Compute accuracy
            pred_labels = predictions.argmax(dim=-1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)
            num_batches += 1

        avg_loss = total_loss / num_batches
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def train(self, train_loader, val_loader=None, num_epochs=100, patience=20):
        """Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            num_epochs: Number of epochs to train
            patience: Early stopping patience
            
        Returns:
            dict: Training history containing train_loss, train_acc, val_loss, val_acc
        """
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f}"
                )

        return history