import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=1.0,
    normalize=True,
    sampling="mul",
    y_pred=None,
):
    """Supervised contrastive loss adapted from SuGAr/models/losses.py.

    Supports three sampling strategies mirroring the SuGAr reference:
      'mul'  — standard SupCon (Khosla et al. 2020): all same-class pairs as positives
      'var'  — SupCon + intra-class variance penalty to discourage cluster collapse
      'cnc'  — Correct-N-Contrast (Yao et al. 2022): hard positives (misclassified,
               same label) vs. hard negatives (correctly classified, different label)

    Args:
        embeddings: [N, D] representation vectors (e.g., graph-level pooled features)
        labels:     [N] integer class indices
        temperature: dot-product scaling factor
        normalize:  L2-normalize embeddings before computing similarities
        sampling:   one of 'mul', 'var', 'cnc'
        y_pred:     [N] predicted class indices — required for 'cnc'

    Returns:
        Scalar contrastive loss.
    """
    if normalize:
        embeddings = F.normalize(embeddings, dim=1)

    device = embeddings.device
    N = labels.size(0)

    # Positive mask: same label, self excluded
    same_label = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
    logits_mask = torch.ones(N, N, device=device).fill_diagonal_(0)
    pos_mask = same_label * logits_mask

    # Compute logits with numerical stability
    logits = torch.div(embeddings @ embeddings.T, temperature)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    if sampling.lower() in ("mul", "var"):
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        is_valid = pos_mask.sum(1) != 0
        mean_log_prob_pos = (pos_mask * log_prob).sum(1)[is_valid] / pos_mask.sum(1)[is_valid]
        loss = -mean_log_prob_pos.mean()
        if sampling.lower() == "var":
            loss = loss + mean_log_prob_pos.var()
        return loss

    elif sampling.lower() == "cnc":
        if y_pred is None:
            raise ValueError("y_pred is required for 'cnc' sampling")

        incorrect = (y_pred != labels).float().to(device)  # [N]
        correct = (y_pred == labels).float().to(device)    # [N]

        # Hard positive: same label AND incorrectly predicted
        hard_pos = pos_mask * incorrect.unsqueeze(1)
        # Hard negative: different label AND correctly predicted
        neg_base = torch.logical_not(same_label).float() * logits_mask
        hard_neg = neg_base * correct.unsqueeze(1)

        # Fall back to all same/diff-label pairs if hard sets are empty
        if hard_neg.sum() == 0:
            hard_neg = neg_base
        if hard_pos.sum() == 0:
            hard_pos = pos_mask

        pos_exp = torch.exp(logits) * hard_pos
        neg_exp = torch.exp(logits) * hard_neg
        log_prob = logits - torch.log(pos_exp.sum(1, keepdim=True) + neg_exp.sum(1, keepdim=True) + 1e-12)

        is_valid = hard_pos.sum(1) != 0
        mean_log_prob_pos = (hard_pos * log_prob).sum(1)[is_valid] / hard_pos.sum(1)[is_valid]
        return -mean_log_prob_pos.mean()

    else:
        raise ValueError(f"Unknown sampling strategy: '{sampling}'. Choose 'mul', 'var', or 'cnc'.")
