import torch.nn.functional as F


def prediction_loss(logits, labels):
    """Cross-entropy between aggregated logits and integer class labels.

    Args:
        logits: [N, num_classes] — output of SuGAREnsemble aggregator
        labels: [N] integer class indices

    Returns:
        Scalar cross-entropy loss.
    """
    return F.cross_entropy(logits, labels.long())
