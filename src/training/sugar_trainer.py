"""Training pipeline for SuGAR ensemble with multi-view fragment learning."""

from copy import deepcopy

import torch
from torch.optim import Adam
from tqdm import tqdm

from src.losses.prediction import prediction_loss
from src.losses.supervised_contrastive import supervised_contrastive_loss
from src.losses.diversity import pairwise_diversity_loss
from src.losses.irm import invariant_risk_loss


class SuGARTrainer:
    """Trainer for a list of FragmentCIGA views forming the SuGAR ensemble.

    Each view is trained independently: its masker identifies fragment edges that
    form an invariant subgraph, reconstruction extracts that atom-level subgraph,
    and its classifier is trained on those atoms. Auxiliary losses (contrastive,
    diversity, IRM) encourage the views to attend to complementary substructures.

    After training, wrap the views in SuGAREnsemble for inference-time aggregation.
    """

    def __init__(
        self,
        views,
        device="cpu",
        learning_rate=0.001,
        weight_decay=0.0,
        lambda_contrast=1.0,
        lambda_diversity=1.0,
        lambda_irm=0.0,
        num_envs=1,
        contrast_temperature=0.5,
        contrast_sampling="mul",
        track_roc_auc=True,
    ):
        """
        Args:
            views: List of FragmentCIGA models (one per ensemble view).
            device: Device string ('cpu' or 'cuda').
            learning_rate: Adam learning rate, applied per-view.
            weight_decay: Adam weight decay.
            lambda_contrast: Weight for supervised contrastive loss (0 disables).
            lambda_diversity: Weight for pairwise diversity loss (0 disables).
            lambda_irm: Weight for IRM penalty. Requires num_envs > 1 to activate.
            num_envs: Number of environment partitions for IRM.
            contrast_temperature: Temperature for contrastive loss.
            contrast_sampling: Contrastive sampling mode ('mul', 'var', 'cnc').
            track_roc_auc: Whether to track ROC-AUC during per-view evaluation.
        """
        self.views = [v.to(device) for v in views]
        self.device = device
        # One optimizer per view — mirrors SuGAr/main.py
        self.optimizers = [
            Adam(v.parameters(), lr=learning_rate, weight_decay=weight_decay)
            for v in self.views
        ]
        self.lambda_contrast = lambda_contrast
        self.lambda_diversity = lambda_diversity
        self.lambda_irm = lambda_irm
        self.num_envs = num_envs
        self.contrast_temperature = contrast_temperature
        self.contrast_sampling = contrast_sampling
        self.track_roc_auc = track_roc_auc

    def uses_irm(self):
        return self.lambda_irm > 0 and self.num_envs > 1

    def train_epoch(self, train_loader):
        """Train all views for one epoch.

        Each view runs its own masker → reconstruction → classifier forward pass.
        Per-view prediction losses are summed; auxiliary losses are added on top.
        One backward pass updates all views jointly.

        Returns:
            avg_loss: Average total loss for the epoch.
            accuracy: Training accuracy (averaged per-view softmax probs).
        """
        for v in self.views:
            v.train()

        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        use_irm = self.uses_irm()
        use_aux = self.lambda_contrast > 0 or self.lambda_diversity > 0 or use_irm

        if use_irm:
            dataset_size = len(train_loader.dataset)
            env_idx = (torch.sigmoid(torch.randn(dataset_size)) > 0.5).long()
        else:
            env_idx = None

        pbar = tqdm(train_loader, desc="Training")
        for step, batch in enumerate(pbar):
            batch = batch.to(self.device)

            labels = batch.y
            if labels.dim() == 2:
                labels = labels.squeeze(-1)
            labels = labels.long()

            # Per-view forward — each view runs its own masker → reconstruct → classify
            per_view = []
            for view in self.views:
                if use_aux:
                    pred, interm = view(batch, return_intermediates=True)
                    per_view.append({"predictions": pred, **interm})
                else:
                    pred = view(batch)
                    per_view.append({"predictions": pred})

            # Sum per-view prediction losses (mirrors SuGAr/main.py total_loss = sum(batch_losses))
            per_view_losses = [prediction_loss(pv["predictions"], labels) for pv in per_view]
            loss = torch.stack(per_view_losses).sum()

            if use_aux:
                # Supervised contrastive over pooled per-view embeddings
                if self.lambda_contrast > 0:
                    all_emb = torch.cat([pv["embeddings"] for pv in per_view], dim=0)
                    all_labels = labels.repeat(len(self.views))
                    if self.contrast_sampling.lower() == "cnc":
                        y_pred = torch.cat(
                            [pv["logits"].detach().argmax(dim=-1) for pv in per_view],
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

                # Pairwise diversity over aligned fragment-edge score vectors
                if self.lambda_diversity > 0 and len(self.views) > 1:
                    loss = loss + self.lambda_diversity * pairwise_diversity_loss(
                        [pv["fragment_scores"] for pv in per_view]
                    )

                # IRM penalty applied per-view, averaged
                if use_irm:
                    batch_size = labels.size(0)
                    start = step * train_loader.batch_size
                    batch_env = env_idx[start: start + batch_size].to(self.device)
                    irm_terms = [
                        invariant_risk_loss(pv["logits"], labels, batch_env)
                        for pv in per_view
                    ]
                    loss = loss + self.lambda_irm * torch.stack(irm_terms).mean()

            # One backward pass through all views; each optimizer steps independently
            for opt in self.optimizers:
                opt.zero_grad()
            loss.backward()
            for v in self.views:
                torch.nn.utils.clip_grad_norm_(v.parameters(), max_norm=1.0)
            for opt in self.optimizers:
                opt.step()

            total_loss += loss.item()
            num_batches += 1

            # Accuracy: average per-view softmax probs → argmax
            avg_probs = torch.stack(
                [pv["predictions"].exp() for pv in per_view], dim=0
            ).mean(dim=0)
            pred_labels = avg_probs.argmax(dim=-1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

            current_acc = correct / total if total > 0 else 0.0
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.4f}"})

        return total_loss / num_batches, correct / total if total > 0 else 0.0

    @torch.no_grad()
    def evaluate_single_view(self, eval_loader, view_idx):
        """Evaluate one view independently (its masker → its reconstruction → its classifier).

        Args:
            eval_loader: DataLoader with batches of MoleculeData.
            view_idx: Index into self.views.

        Returns:
            accuracy: Classification accuracy for this view.
        """
        view = self.views[view_idx]
        view.eval()
        correct = 0
        total = 0

        for batch in eval_loader:
            batch = batch.to(self.device)
            predictions = view(batch)
            labels = batch.y
            if labels.dim() == 2:
                labels = labels.squeeze(-1)
            labels = labels.long()
            correct += (predictions.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        return correct / total if total > 0 else 0.0

    def train(self, train_loader, val_loader=None, num_epochs=100, patience=20, restore_best_weights=True):
        """Full training loop with per-view best-weight tracking.

        Each view's best weights are saved independently whenever that view
        achieves a new personal best validation accuracy, mirroring SuGAr/main.py.
        Early stopping is driven by the average per-view validation accuracy.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader (optional).
            num_epochs: Maximum epochs.
            patience: Early stopping patience (epochs without avg val acc improvement).
            restore_best_weights: If True, restore each view's personal best
                weights after training completes.

        Returns:
            dict: History with 'train_loss', 'train_acc', 'val_acc_per_view', 'val_acc_avg'.
        """
        num_views = len(self.views)
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_acc_per_view": [],
            "val_acc_avg": [],
        }

        per_view_best_val_acc = [-float("inf")] * num_views
        per_view_best_weights = [None] * num_views
        best_avg_val_acc = -float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            if val_loader is not None:
                view_val_accs = []
                for i in range(num_views):
                    acc = self.evaluate_single_view(val_loader, i)
                    view_val_accs.append(acc)
                    if acc > per_view_best_val_acc[i]:
                        per_view_best_val_acc[i] = acc
                        per_view_best_weights[i] = deepcopy(self.views[i].state_dict())

                avg_val_acc = sum(view_val_accs) / num_views
                history["val_acc_per_view"].append(view_val_accs)
                history["val_acc_avg"].append(avg_val_acc)

                view_acc_str = "  ".join(
                    f"v{i}={a:.4f}" for i, a in enumerate(view_val_accs)
                )
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Acc [{view_acc_str}] | Avg: {avg_val_acc:.4f}"
                )

                if avg_val_acc > best_avg_val_acc:
                    best_avg_val_acc = avg_val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                )

        if restore_best_weights:
            restored = 0
            for i in range(num_views):
                if per_view_best_weights[i] is not None:
                    self.views[i].load_state_dict(per_view_best_weights[i])
                    restored += 1
            if restored:
                print(
                    f"Restored best per-view weights for {restored}/{num_views} views "
                    f"(best avg val acc = {best_avg_val_acc:.4f})"
                )

        return history
