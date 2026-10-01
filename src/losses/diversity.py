import torch


def _pair_cosine(s_a, s_b, edge_batch, num_graphs, weight, center, eps):
    """Mean per-molecule cosine of two score vectors.

    Edges are masked multiplicatively by ``weight`` (1.0 where both views scored
    the edge, else 0.0) instead of indexed.
    Molecules with fewer than 2 weighted edges are excluded.
    """
    device = s_a.device
    G = num_graphs
    S = torch.stack((s_a, s_b))  # [2, E]
    counts = torch.zeros(G, device=device).index_add_(0, edge_batch, weight)
    valid = (counts >= 2).to(S.dtype)

    if center:  # subtract each view's per-molecule mean over its scored edges
        mean = torch.zeros(2, G, device=device).index_add_(1, edge_batch, S * weight)
        S = S - mean.div_(counts.clamp(min=1))[:, edge_batch]
    S = S * weight  # dropped edges -> 0

    norm = torch.zeros(2, G, device=device).index_add_(1, edge_batch, S * S)
    S_hat = S / norm.clamp(min=eps * eps).sqrt()[:, edge_batch]
    cos = torch.zeros(G, device=device).index_add_(0, edge_batch, S_hat[0] * S_hat[1])
    # Penalize similarity only (clamp >= 0); average over molecules with >= 2
    # scored edges, guarding a pair whose molecules are all invalid.
    return (cos.clamp(min=0.0) * valid).sum() / valid.sum().clamp(min=1.0)


def pairwise_diversity_loss(
    fragment_scores,
    edge_batch,
    eligible_masks=None,
    num_graphs=None,
    center=True,
    eps=1e-8,
):
    """Pairwise diversity loss over aligned fragment-edge score vectors.

    Penalizes similarity between per-view score vectors so views attend to
    different edges. Cosine is measured per molecule and averaged over
    molecules; ``eligible_masks`` restricts each view pair to the edges both
    views scored. Scores are mean-centered per molecule and the per-molecule
    cosine is clamped at 0, so the loss stays in [0, 1].

    Args:
        fragment_scores: list of V tensors, returned by a view's forward.
        edge_batch: LongTensor ``[num_edges]`` mapping each fragment edge to its
            molecule index (``intermediates['fragment_edge_batch']``).
        eligible_masks: optional list of V boolean tensors ``[num_edges]`` marking
            which edges each view scored (``intermediates['fragment_edge_eligible']``).
        num_graphs: molecule count; inferred from ``edge_batch`` if None.
        center: mean-center each view's scores per molecule before the cosine.
        eps: numerical floor for the L2 normalization.

    Returns:
        Scalar loss in [0, 1]: mean non-negative per-molecule cosine across all
        view pairs.
    """
    V = len(fragment_scores)
    if V < 2:
        device = fragment_scores[0].device if fragment_scores else torch.device("cpu")
        return torch.zeros((), device=device)

    S = torch.stack(fragment_scores)  # [V, E]
    device = S.device
    E = S.size(1)
    if E == 0:
        return torch.zeros((), device=device)

    edge_batch = edge_batch.to(device=device, dtype=torch.long)
    if num_graphs is None:
        num_graphs = int(edge_batch.max().item()) + 1

    ones = torch.ones(E, device=device, dtype=S.dtype)
    if eligible_masks is not None:
        eligible_masks = [m.to(device=device, dtype=S.dtype) for m in eligible_masks]

    total = torch.zeros((), device=device)
    num_pairs = 0
    for a in range(V):
        for b in range(a + 1, V):
            weight = ones if eligible_masks is None else eligible_masks[a] * eligible_masks[b]
            total = total + _pair_cosine(S[a], S[b], edge_batch, num_graphs, weight, center, eps)
            num_pairs += 1
    return total / num_pairs
