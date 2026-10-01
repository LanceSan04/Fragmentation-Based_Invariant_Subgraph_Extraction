"""Whole-fragment prefix selection targeting a retained atom count."""
import math
import torch


def atom_budget_mask(scores, edge_index, fragment_batch, fragment_sizes,
                     atom_counts, eligible, ratio):
    """Closest nonempty ranked prefix; ties prefer fewer atoms, then edges.

    Each fragment contributes its atom count only at its first incident edge
    in the ranked prefix. No atoms are split, and no extra bonds are inserted.
    Graphs without eligible edges return an empty mask (reconstruction owns
    the existing edgeless-graph fallback).
    """
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('atom retention ratio must be in (0, 1]')
    mask = torch.zeros_like(scores, dtype=torch.bool)
    ids = torch.where(eligible)[0]
    if ids.numel() == 0:
        return mask
    # Stable two-stage sort: graph first, score descending, original id ties.
    ids = ids[torch.argsort(scores[ids], descending=True, stable=True)]
    groups = fragment_batch[edge_index[0, ids]]
    order = torch.argsort(groups, stable=True)
    ids, groups = ids[order], groups[order]
    n = ids.numel()
    positions = torch.arange(n, device=scores.device)
    first = torch.full_like(fragment_sizes, n)
    first.scatter_reduce_(0, edge_index[:, ids].reshape(-1),
                          positions.repeat(2), reduce='amin', include_self=True)
    increments = torch.zeros(n + 1, dtype=torch.long, device=scores.device)
    increments.index_add_(0, first, fragment_sizes.long())
    cumulative = increments[:n].cumsum(0)
    counts = torch.bincount(groups, minlength=atom_counts.numel())
    starts = counts.cumsum(0) - counts
    before = torch.cat((cumulative.new_zeros(1), cumulative))[starts]
    retained = cumulative - before[groups]
    target = torch.ceil(atom_counts.double() * ratio).long()
    distance = (retained - target[groups]).abs()
    best_distance = torch.full_like(atom_counts, torch.iinfo(torch.long).max)
    best_distance.scatter_reduce_(0, groups, distance, reduce='amin', include_self=True)
    candidates = distance == best_distance[groups]
    best_atoms = torch.full_like(best_distance, torch.iinfo(torch.long).max)
    best_atoms.scatter_reduce_(0, groups[candidates], retained[candidates], reduce='amin', include_self=True)
    candidates &= retained == best_atoms[groups]
    last = torch.full_like(atom_counts, n)
    last.scatter_reduce_(0, groups[candidates], positions[candidates], reduce='amin', include_self=True)
    mask[ids] = positions <= last[groups]
    return mask
