"""Greedy view selection for SuGAR ensemble post-training.

Mirrors the greedy checkpoint selection in SuGAr/ens.py:

  1. Score each view independently on the validation loader.
  2. Sort views descending by individual validation accuracy.
  3. Greedily add views to the ensemble; keep a view only if it improves
     the ensemble validation accuracy.

The result is a compact subset of views that the ensemble uses at test time.
"""

import torch

try:
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


@torch.no_grad()
def _accuracy_on_loader(model, loader, device, view_indices=None):
    """Classification accuracy for the given view subset over a full loader."""
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        predictions = model(batch, view_indices=view_indices)
        labels = batch.y
        if labels.dim() == 2:
            labels = labels.squeeze(-1)
        labels = labels.long()
        pred_labels = predictions.argmax(dim=-1)
        correct += (pred_labels == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0


@torch.no_grad()
def _auc_on_loader(model, loader, device, view_indices=None):
    """Macro-averaged ROC-AUC for the given view subset over a full loader.

    Returns None if AUC is undefined (single class present) or sklearn is missing.
    """
    if not _SKLEARN_AVAILABLE:
        return None
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        predictions = model(batch, view_indices=view_indices)
        labels = batch.y
        if labels.dim() == 2:
            labels = labels.squeeze(-1)
        labels = labels.long()
        # The ensemble returns log-probabilities, not logits.
        probs = predictions.exp()
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    num_classes = all_probs.shape[1]
    try:
        if num_classes == 2:
            return _sklearn_roc_auc(all_labels, all_probs[:, 1])
        return _sklearn_roc_auc(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        return None


def greedy_view_selection(model, val_loader, device, use_auc=False):
    """Select the subset of views that maximises ensemble validation metric.

    Args:
        model: Trained SuGAREnsemble.
        val_loader: Validation DataLoader used to score candidate subsets.
        device: Torch device the model lives on.
        use_auc: Score by ROC-AUC instead of accuracy. Falls back to accuracy
            if AUC is unavailable (sklearn missing or single class).

    Returns:
        list[int]: Indices of the selected views, in the order they were added.
    """
    num_views = model.num_views
    if num_views == 1:
        print("Greedy view selection: only one view — returning [0].")
        return [0]

    def score(view_indices):
        if use_auc:
            auc = _auc_on_loader(model, val_loader, device, view_indices=view_indices)
            if auc is not None:
                return auc, "auc"
        return _accuracy_on_loader(model, val_loader, device, view_indices=view_indices), "acc"

    metric_label = "auc" if (use_auc and _SKLEARN_AVAILABLE) else "acc"

    # Step 1: score each view individually
    print(f"Greedy view selection — scoring individual views (metric: val_{metric_label}):")
    view_scores = []
    for i in range(num_views):
        value, label = score([i])
        view_scores.append((i, value))
        print(f"  View {i}: val_{label} = {value:.4f}")

    # Step 2: sort descending by individual score
    view_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [idx for idx, _ in view_scores]

    # Step 3: greedy forward selection
    selected = []
    best_score = -float("inf")
    for candidate in sorted_indices:
        candidate_set = selected + [candidate]
        value, label = score(candidate_set)
        if value > best_score:
            best_score = value
            selected = candidate_set
            print(f"  + View {candidate} kept   → ensemble val_{label} = {value:.4f}")
        else:
            print(f"  - View {candidate} skipped ({label}={value:.4f} ≤ best={best_score:.4f})")

    print(f"Selected views: {selected}  (val_{metric_label} = {best_score:.4f})")
    return selected
