"""Greedy view selection for SuGAR ensemble post-training.

Mirrors the greedy checkpoint selection in SuGAr/ens.py:

  1. Score each view independently on the validation loader.
  2. Sort views descending by individual validation accuracy.
  3. Greedily add views to the ensemble; keep a view only if it improves
     the ensemble validation accuracy.

The result is a compact subset of views that the ensemble uses at test time.
"""

import torch


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


def greedy_view_selection(model, val_loader, device):
    """Select the subset of views that maximises ensemble validation accuracy.

    Args:
        model: Trained SuGAREnsemble.
        val_loader: Validation DataLoader used to score candidate subsets.
        device: Torch device the model lives on.

    Returns:
        list[int]: Indices of the selected views, in the order they were added.
    """
    num_views = model.num_views
    if num_views == 1:
        print("Greedy view selection: only one view — returning [0].")
        return [0]

    # Step 1: score each view individually
    print("Greedy view selection — scoring individual views:")
    view_scores = []
    for i in range(num_views):
        acc = _accuracy_on_loader(model, val_loader, device, view_indices=[i])
        view_scores.append((i, acc))
        print(f"  View {i}: val_acc = {acc:.4f}")

    # Step 2: sort descending by individual accuracy, matching ens.py's
    # `sorted_checkpoints` ordering before the greedy loop
    view_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [idx for idx, _ in view_scores]

    # Step 3: greedy forward selection — keep a candidate view only if it
    # strictly improves the current best ensemble accuracy 
    selected = []
    best_acc = -float("inf")
    for candidate in sorted_indices:
        candidate_set = selected + [candidate]
        acc = _accuracy_on_loader(model, val_loader, device, view_indices=candidate_set)
        if acc > best_acc:
            best_acc = acc
            selected = candidate_set
            print(f"  + View {candidate} kept   → ensemble val_acc = {acc:.4f}")
        else:
            print(f"  - View {candidate} skipped (acc={acc:.4f} ≤ best={best_acc:.4f})")

    print(f"Selected views: {selected}  (val_acc = {best_acc:.4f})")
    return selected
