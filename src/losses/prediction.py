import torch.nn.functional as F


def prediction_loss(log_probs, labels):
    """NLL loss between soft-vote log-probabilities and integer class labels.

    Args:
        log_probs: [N, num_classes] — log-probabilities from SuGAREnsemble.forward
        labels: [N] integer class indices

    Returns:
        Scalar NLL loss.
    """
    return F.nll_loss(log_probs, labels.long())
