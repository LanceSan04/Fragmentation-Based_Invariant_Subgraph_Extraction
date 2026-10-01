"""Uniform random fragment-edge ranking with batch-invariant evaluation draws."""

import torch


def random_edge_ranks(fragment_batch, seed, training):
    """Give each edge a random rank; evaluation depends only on its molecule.

    Training draws fresh ranks on every forward pass. Evaluation hashes the
    molecule's ordered fragment features and its local edge identifiers, so a
    molecule gets the same draw regardless of batch order or batch size.
    This is a control, not a learned chemical importance score.
    """
    edge_index = fragment_batch.edge_index
    count = edge_index.size(1)
    if not count:
        return fragment_batch.x.new_empty((0,))
    if training:
        return torch.rand(count, device=edge_index.device, dtype=fragment_batch.x.dtype)

    node_batch = fragment_batch.batch.long()
    edge_batch = node_batch[edge_index[0]]
    num_graphs = int(node_batch.max().item()) + 1
    node_counts = torch.bincount(node_batch, minlength=num_graphs)
    node_offsets = torch.cumsum(node_counts, 0) - node_counts
    edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
    edge_offsets = torch.cumsum(edge_counts, 0) - edge_counts

    # Fragment features are stable preprocessed molecular descriptors. Integer
    # arithmetic makes this fingerprint independent of CUDA reduction order.
    features = torch.round(fragment_batch.x.detach() * 1000).long()
    feature_columns = torch.arange(1, features.size(1) + 1, device=features.device)
    local_nodes = torch.arange(features.size(0), device=features.device) - node_offsets[node_batch]
    node_keys = (features * feature_columns).sum(1) + 104729 * (local_nodes + 1)
    graph_keys = torch.zeros(num_graphs, dtype=torch.long, device=features.device)
    graph_keys.index_add_(0, node_batch, node_keys)

    local_edges = torch.arange(count, device=edge_index.device) - edge_offsets[edge_batch]
    local_src = edge_index[0] - node_offsets[edge_batch]
    local_dst = edge_index[1] - node_offsets[edge_batch]
    key = (graph_keys[edge_batch] + (int(seed) + 1) * 8191
           + (local_edges + 1) * 131071 + (local_src + 1) * 1009
           + (local_dst + 1) * 9176) & 0x7fffffff
    # Mix each 31-bit key before converting it to a rank in [0, 1).
    for _ in range(3):
        key = ((key ^ (key >> 16)) * 0x45D9F3B) & 0x7fffffff
    return (key.to(torch.float64) / float(1 << 31)).to(fragment_batch.x.dtype)
