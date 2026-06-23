import torch
import torch.nn.functional as F


def pairwise_diversity_loss(fragment_scores):
    """Pairwise diversity loss over aligned fragment-edge score vectors.

    All views in SuGAREnsemble score the same fragment graph, so the score
    vectors are positionally aligned (same edge index → same position).
    This loss penalizes high cosine similarity between per-view score vectors,
    pushing the views to attend to different edges.

    Args:
        fragment_scores: list of V tensors, each [num_edges], from
                         intermediates['fragment_scores'] returned by
                         SuGAREnsemble.forward(..., return_intermediates=True).

    Returns:
        Scalar loss: mean pairwise cosine similarity across all view pairs.
        Minimizing this encourages diverse edge selection across views.
    """
    V = len(fragment_scores)
    if V < 2:
        device = fragment_scores[0].device if fragment_scores else torch.device("cpu")
        return torch.tensor(0.0, device=device)

    # [V, num_edges]
    S = torch.stack(fragment_scores, dim=0)
    S_norm = F.normalize(S, dim=1)

    # [V, V] pairwise cosine similarities
    sim = S_norm @ S_norm.T

    # Average over strictly upper-triangle (each unordered pair counted once)
    idx = torch.triu_indices(V, V, offset=1, device=S.device)
    return sim[idx[0], idx[1]].mean()
