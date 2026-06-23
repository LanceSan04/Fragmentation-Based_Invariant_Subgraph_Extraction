import torch
import torch.nn.functional as F


def invariant_risk_loss(causal_pred, labels, batch_env_idx, criterion=F.cross_entropy):
    """IRM gradient penalty (Arjovsky et al. 2019) across two environments.

    Minimizing this penalty forces the invariant-subgraph predictor to produce
    loss gradients that agree across both environment partitions, discouraging
    spurious-feature reliance.

    Mirrors get_irm_loss in SuGAr.

    Args:
        causal_pred:   [N, C] logits from the invariant-subgraph classifier
        labels:        [N] integer class indices
        batch_env_idx: [N] binary LongTensor (0 or 1) — environment assignment
        criterion:     loss fn(pred, labels) — default cross_entropy

    Returns:
        Scalar IRM penalty. Returns 0 when either environment is empty in the batch.
    """
    device = causal_pred.device
    env0 = batch_env_idx == 0
    env1 = batch_env_idx == 1

    if env0.sum() == 0 or env1.sum() == 0:
        return torch.tensor(0.0, device=device)

    dummy_w = torch.tensor(1.0, device=device).requires_grad_()
    loss_0 = criterion(causal_pred[env0] * dummy_w, labels[env0])
    loss_1 = criterion(causal_pred[env1] * dummy_w, labels[env1])
    grad_0 = torch.autograd.grad(loss_0, dummy_w, create_graph=True)[0]
    grad_1 = torch.autograd.grad(loss_1, dummy_w, create_graph=True)[0]
    return torch.sum(grad_0 * grad_1)
