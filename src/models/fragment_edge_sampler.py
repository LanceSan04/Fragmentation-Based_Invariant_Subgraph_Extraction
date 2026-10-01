"""Training-time random sampling for fragment-edge instances.

This module is the project version of SuGAr's random subgraph sampler. It
does not change preprocessing and it does not remove atom nodes, atom bonds,
fragment nodes, or saved molecule records. It only marks a random subset of
preprocessed fragment-edge instances as eligible for one model forward pass.

The sampled graph preserves the original fragment nodes and stores
``full_fragment_edge_id`` so CIGA selections on the sampled graph can be mapped
back to the full preprocessed edge order before atom reconstruction.
"""

import torch
from torch_geometric.data import Data


def fragment_edge_batch(fragment_batch):
    """Return the molecule id for every fragment-edge instance in a batch."""
    if fragment_batch.edge_index.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=fragment_batch.x.device)
    return fragment_batch.batch[fragment_batch.edge_index[0]]


def random_fragment_edge_mask(
    edge_batch,
    keep_ratio=1.0,
    training=True,
    seed=None,
    ensure_one_edge=True,
):
    """Sample a full-edge-aligned eligibility mask per molecule.

    Args:
        edge_batch: Tensor assigning each fragment edge to a molecule.
        keep_ratio: Fraction of fragment-edge instances to keep.
        training: When false, return the full graph.
        seed: Optional deterministic seed for tests or reproducible views.
        ensure_one_edge: Keep at least one edge for molecules that have edges.

    Returns:
        Boolean mask with one value per original fragment-edge instance.
    """
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in [0, 1]")

    mask = torch.ones(edge_batch.numel(), dtype=torch.bool, device=edge_batch.device)
    if edge_batch.numel() == 0 or not training or keep_ratio >= 1.0:
        return mask

    mask.fill_(False)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=edge_batch.device)
        generator.manual_seed(int(seed))

    # Assign each edge a random key and keep the lowest-key ``keep_count`` edges
    # per graph via one segmented sort.
    rand = torch.rand(edge_batch.numel(), device=edge_batch.device, generator=generator)
    counts = torch.bincount(edge_batch)
    offsets = torch.cumsum(counts, dim=0) - counts
    keep_count = (counts.to(torch.float32) * keep_ratio).to(torch.long)
    if ensure_one_edge and keep_ratio > 0.0:
        keep_count = torch.where(counts > 0, torch.clamp(keep_count, min=1), keep_count)
    keep_count = torch.minimum(keep_count, counts)

    order_r = torch.argsort(rand)
    order_g = torch.argsort(edge_batch[order_r], stable=True)
    perm = order_r[order_g]

    sorted_g = edge_batch[perm]
    positions = torch.arange(perm.numel(), device=edge_batch.device)
    within_rank = positions - offsets[sorted_g]
    keep = within_rank < keep_count[sorted_g]
    mask[perm[keep]] = True

    return mask


def filter_fragment_graph_edges(fragment_batch, eligible_edge_mask):
    """Build a sampled fragment graph while preserving full-edge ids."""
    eligible_edge_mask = torch.as_tensor(
        eligible_edge_mask,
        dtype=torch.bool,
        device=fragment_batch.edge_index.device,
    )
    num_edges = fragment_batch.edge_index.size(1)
    if eligible_edge_mask.numel() != num_edges:
        raise ValueError("eligible_edge_mask must contain one value per fragment edge")

    full_edge_ids = getattr(fragment_batch, "full_fragment_edge_id", None)
    if full_edge_ids is None:
        full_edge_ids = torch.arange(
            num_edges, dtype=torch.long, device=fragment_batch.edge_index.device
        )

    return Data(
        x=fragment_batch.x,
        edge_index=fragment_batch.edge_index[:, eligible_edge_mask],
        edge_attr=fragment_batch.edge_attr[eligible_edge_mask],
        y=getattr(fragment_batch, "y", None),
        batch=fragment_batch.batch,
        ptr=getattr(fragment_batch, "ptr", None),
        full_fragment_edge_id=full_edge_ids[eligible_edge_mask],
        full_num_fragment_edges=torch.tensor(
            num_edges,
            dtype=torch.long,
            device=fragment_batch.edge_index.device,
        ),
        num_nodes=fragment_batch.x.size(0),
    )


class FragmentEdgeSampler(torch.nn.Module):
    """Randomly drop fragment-edge instances during training."""

    def __init__(self, keep_ratio=1.0, seed=None, ensure_one_edge=True):
        super().__init__()
        if not 0.0 <= keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in [0, 1]")
        self.keep_ratio = keep_ratio
        self.seed = seed
        self.ensure_one_edge = ensure_one_edge
        self.register_buffer("_call_count", torch.zeros((), dtype=torch.long), persistent=False)

    def forward(self, fragment_batch):
        """Return ``(sampled_fragment_batch, eligible_full_edge_mask)``."""
        edge_batch = fragment_edge_batch(fragment_batch)
        if self.seed is None:
            seed = None
        else:
            seed = int(self.seed) + int(self._call_count.item())

        eligible_mask = random_fragment_edge_mask(
            edge_batch,
            keep_ratio=self.keep_ratio,
            training=self.training,
            seed=seed,
            ensure_one_edge=self.ensure_one_edge,
        )

        if self.training:
            self._call_count += 1

        return filter_fragment_graph_edges(fragment_batch, eligible_mask), eligible_mask
