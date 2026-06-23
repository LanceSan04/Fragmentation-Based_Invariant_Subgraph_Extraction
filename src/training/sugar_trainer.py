"""Training pipeline for SuGAr ensemble with multi-view fragment learning."""

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.losses.prediction import prediction_loss
from src.losses.supervised_contrastive import supervised_contrastive_loss
from src.losses.diversity import pairwise_diversity_loss
from src.losses.irm import invariant_risk_loss


class SuGARTrainer:
    """Trainer for SuGAR ensemble model."""

    def __init__(
        self,
        model,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        lambda_contrast=1.0,
        lambda_diversity=1.0,
        lambda_irm=1.0,
        contrast_temperature=0.5,
        contrast_sampling="mul",
    ):
        """
        Args:
            model: SuGAREnsemble model
            device: Device to train on (cpu/cuda)
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            lambda_contrast: Weight for supervised contrastive loss (0 disables it)
            lambda_diversity: Weight for pairwise diversity loss (0 disables it)
            lambda_irm: Weight for IRM penalty (0 disables it). The default
                is intentionally left as-is from the incoming loss branch, but
                random environment assignments are a provisional SuGAr-style
                fallback. Prefer real environment/scaffold/assay groups when
                available.
            contrast_temperature: Temperature for supervised contrastive loss
            contrast_sampling: Sampling mode for contrastive loss ('mul', 'var', 'cnc')
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.lambda_contrast = lambda_contrast
        self.lambda_diversity = lambda_diversity
        self.lambda_irm = lambda_irm
        self.contrast_temperature = contrast_temperature
        self.contrast_sampling = contrast_sampling

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

        use_aux = self.lambda_contrast > 0 or self.lambda_diversity > 0 or self.lambda_irm > 0

        # Pre-generate random binary environment assignments for the full dataset,
        # mirroring SuGAr main.py. Sliced per-batch below.
        if self.lambda_irm > 0:
            dataset_size = len(train_loader.dataset)
            env_idx = (torch.sigmoid(torch.randn(dataset_size)) > 0.5).long()
        else:
            env_idx = None

        pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(pbar):
            batch = batch.to(self.device)

            # Forward pass — collect intermediates only when auxiliary losses are active
            if use_aux:
                predictions, intermediates = self.model(batch, return_intermediates=True)
            else:
                predictions = self.model(batch)

            # Get labels
            labels = batch.y
            if labels.dim() == 2:
                labels = labels.squeeze(-1)
            labels = labels.long()

            # Prediction loss
            loss = prediction_loss(predictions, labels)

            if use_aux:
                # Supervised contrastive loss over pooled per-view embeddings.
                # Concatenate views along the batch axis so every molecule
                # appears once per view, giving more positive/negative pairs.
                if self.lambda_contrast > 0:
                    all_emb = torch.cat(intermediates["embeddings"], dim=0)   # [N*V, D]
                    all_labels = labels.repeat(self.model.num_views)           # [N*V]
                    if self.contrast_sampling.lower() == "cnc":
                        y_pred = torch.cat(
                            [
                                view_logits.detach().argmax(dim=-1)
                                for view_logits in intermediates["per_view_logits"]
                            ],
                            dim=0,
                        )
                    else:
                        y_pred = None
                    loss = loss + self.lambda_contrast * supervised_contrastive_loss(
                        all_emb,
                        all_labels,
                        temperature=self.contrast_temperature,
                        sampling=self.contrast_sampling,
                        y_pred=y_pred,
                    )

                # Pairwise diversity loss over aligned fragment-edge score vectors.
                if self.lambda_diversity > 0 and len(intermediates["fragment_scores"]) > 1:
                    loss = loss + self.lambda_diversity * pairwise_diversity_loss(
                        intermediates["fragment_scores"]
                    )

                # IRM penalty applied to each view's invariant-subgraph logits,
                # averaged across views (mirrors SuGAr: irm applied to causal_pred).
                if self.lambda_irm > 0:
                    batch_size = labels.size(0)
                    start = step * train_loader.batch_size
                    batch_env = env_idx[start: start + batch_size].to(self.device)
                    irm_terms = []
                    for view_logits in intermediates["per_view_logits"]:
                        irm_terms.append(invariant_risk_loss(view_logits, labels, batch_env))
                    if irm_terms:
                        loss = loss + self.lambda_irm * torch.stack(irm_terms).mean()

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

            loss = prediction_loss(predictions, labels)
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
