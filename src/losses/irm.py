import torch
import torch.nn.functional as F


def invariant_risk_loss(causal_pred, labels, batch_env_idx, criterion=F.cross_entropy):
    """IRM gradient penalty (Arjovsky et al. 2019, IRMv1 practical penalty).

    For each environment present in the batch, computes the gradient of that
    environment's loss with respect to a fixed dummy classifier scale of 1.0,
    then penalizes the sum of squared per-environment gradients. This is zero
    only when every environment's gradient is near zero, i.e. the
    invariant-subgraph predictor is simultaneously near-optimal for every
    environment, discouraging spurious-feature reliance.

    Note: this differs from SuGAr's get_irm_loss, which computed a signed
    grad_0 * grad_1 product. That formula is only non-pathological when
    environments 0/1 are a random split of the same distribution (there it is
    an unbiased estimator of the squared population gradient). Once
    environments represent genuinely different distributions (e.g. grouped by
    domain_id), an unsquared signed product has no floor at zero, so
    minimizing it can reward the optimizer for making per-environment
    gradients diverge in opposite directions instead of driving them to zero.
    Squaring each environment's gradient keeps the penalty non-negative
    regardless of how environments are constructed.

    Args:
        causal_pred:   [N, C] logits from the invariant-subgraph classifier
        labels:        [N] integer class indices
        batch_env_idx: [N] LongTensor of environment ids — environment assignment
        criterion:     loss fn(pred, labels) — default cross_entropy

    Returns:
        Scalar IRM penalty >= 0. Returns 0 when fewer than two environments
        are present in the batch.
    """
    device = causal_pred.device
    env_ids = torch.unique(batch_env_idx)

    grads = []
    for env_id in env_ids:
        env_mask = batch_env_idx == env_id
        dummy_w = torch.tensor(1.0, device=device).requires_grad_()
        env_loss = criterion(causal_pred[env_mask] * dummy_w, labels[env_mask])
        grads.append(torch.autograd.grad(env_loss, dummy_w, create_graph=True)[0])

    if len(grads) < 2:
        return torch.tensor(0.0, device=device)

    return torch.sum(torch.stack([grad ** 2 for grad in grads]))
