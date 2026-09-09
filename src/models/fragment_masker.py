"""Fragment-level CIGA masker for scoring and selecting fragment-edge instances."""

import torch
import torch.nn as nn
from src.data.atom_graph import ATOM_FDIM, BOND_FDIM
from src.models.gnn_encoders import SimpleGNN


FRAGMENT_EDGE_DIM = (2 * ATOM_FDIM) + BOND_FDIM + 2


def reverse_fragment_edge_attr(edge_attr):
    """Swap the src/dst blocks of a fragment-edge feature for the reverse direction.

    Layout is ``[src atom | dst atom | bond | src size | dst size]``, built by
    ``group_graph.build_fragment_edge_features``. Widths that do not match that
    layout are returned unchanged.
    """
    if edge_attr is None or edge_attr.size(-1) != FRAGMENT_EDGE_DIM:
        return edge_attr
    atom = ATOM_FDIM
    return torch.cat(
        [
            edge_attr[:, atom : 2 * atom],
            edge_attr[:, 0:atom],
            edge_attr[:, 2 * atom : 2 * atom + BOND_FDIM],
            edge_attr[:, -1:],
            edge_attr[:, -2:-1],
        ],
        dim=-1,
    )


def segment_topk_mask(scores, edge_batch, ratio, eligible_mask=None):
    """Vectorized per-graph top-k edge selection.

    For each graph (a segment in ``edge_batch``) keep the top
    ``max(1, floor(count * ratio))`` edges by score, where ``count`` is the
    graph's *total* edge count. When ``eligible_mask`` is given, only eligible
    edges are candidates, but ``k`` is still sized from the full edge count and
    then clipped to the number of eligible edges. Sizing ``k`` from the eligible
    subset instead would make a sampled training graph select a smaller subgraph
    than the full graph seen at eval, shifting the input distribution the
    classifier's BatchNorm statistics are fitted to. Replaces the per-graph
    Python loop with a single segmented sort so the whole batch is handled in a
    few vectorized ops.

    Ties in score are broken arbitrarily, as with ``torch.topk``.
    """
    device = scores.device
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if scores.numel() == 0:
        return mask

    edge_batch = edge_batch.to(device)
    num_graphs = int(edge_batch.max().item()) + 1
    if eligible_mask is not None:
        eligible_mask = eligible_mask.to(device=device, dtype=torch.bool)
        cand_idx = eligible_mask.nonzero(as_tuple=True)[0]
    else:
        cand_idx = torch.arange(scores.numel(), device=device)
    if cand_idx.numel() == 0:
        return mask

    g = edge_batch[cand_idx]
    s = scores[cand_idx]

    # Offsets index the candidate array, so they follow the eligible counts,
    # while k follows the full per-graph edge count.
    counts_all = torch.bincount(edge_batch, minlength=num_graphs)
    counts_eligible = torch.bincount(g, minlength=num_graphs)
    offsets = torch.cumsum(counts_eligible, dim=0) - counts_eligible
    k = torch.clamp((counts_all.to(torch.float32) * ratio).to(torch.long), min=1)
    k = torch.minimum(k, counts_eligible)

    # Order candidates by (graph asc, score desc); the second sort is stable so
    # the score ordering is preserved within each graph.
    order_s = torch.argsort(s, descending=True)
    order_g = torch.argsort(g[order_s], stable=True)
    perm = order_s[order_g]

    sorted_g = g[perm]
    positions = torch.arange(perm.numel(), device=device)
    within_rank = positions - offsets[sorted_g]
    keep = within_rank < k[sorted_g]

    mask[cand_idx[perm[keep]]] = True
    return mask


class FragmentMasker(nn.Module):
    """CIGA-style module: fragment graph -> edge weights + invariant edge mask.

    The hard top-k mask chooses which fragment-edge instances are reconstructed.
    The continuous scores must still be passed downstream as selected edge
    weights so prediction loss can train the scorer.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.0,
        ratio=0.5,
        edge_dim=None,
    ):
        super().__init__()
        self.ratio = ratio
        self.gnn_encoder = SimpleGNN(input_dim, hidden_dim, num_layers, dropout, edge_dim=edge_dim)
        self.edge_att = nn.Sequential(
            nn.Linear((hidden_dim * 2) + (edge_dim or 0), hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, 1),
        )

    def forward(self, data):
        """
        Args:
            data: Batched PyG Data for the fragment graph.

        Returns:
            soft_weights: Continuous edge scores [num_edges] in (0, 1).
            bool_mask: Boolean top-k mask [num_edges], True for kept edges.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)

        # The stored fragment graph keeps one canonical column per inter-fragment
        # bond (src <= dst), so message passing over it as-is would only ever flow
        # from low to high fragment id. Symmetrize for the encoder only; scoring
        # below stays on the canonical columns, which the top-k selection,
        # reconstruction provenance and diversity loss all index by.
        encoder_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        encoder_edge_attr = (
            None
            if edge_attr is None
            else torch.cat([edge_attr, reverse_fragment_edge_attr(edge_attr)], dim=0)
        )

        h, _ = self.gnn_encoder(x, encoder_edge_index, batch, edge_attr=encoder_edge_attr)

        src, dst = edge_index
        edge_rep_pieces = [h[src], h[dst]]
        if self.gnn_encoder.edge_dim is not None:
            edge_rep_pieces.append(edge_attr)
        edge_rep = torch.cat(edge_rep_pieces, dim=-1)
        soft_weights = torch.sigmoid(self.edge_att(edge_rep).view(-1))

        # Top-k indices are discrete. Detaching here only isolates mask
        # construction; selected soft_weights are still passed downstream.
        bool_mask = self._topk_mask(soft_weights.detach(), batch[src], self.ratio)
        return soft_weights, bool_mask

    @staticmethod
    def _topk_mask(scores, edge_batch, ratio):
        """Boolean top-k mask keeping `ratio` fraction of edges per graph."""
        return segment_topk_mask(scores, edge_batch, ratio)
