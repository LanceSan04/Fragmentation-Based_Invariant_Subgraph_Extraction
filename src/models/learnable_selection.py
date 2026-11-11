"""Budget-constrained smooth selection and straight-through graph masks.

The hard forward decision is unchanged. Backward uses the derivative of a
sigmoid threshold constrained to sum to k; it is a surrogate, not the true
derivative of hard top-k. Sampling-ineligible edges receive no selection weight.
"""

import torch
from torch_geometric.data import Data
from src.models.atom_reconstructor import fragment_soft_importance
from src.models.fragment_graph import get_fragment_view_value


def gumbel_ranks(scores, seed):
    """Log-odds plus unit-scale Gumbel noise; local RNG, differentiable scores.

    This is a Gumbel-perturbed straight-through budget-sigmoid variant,
    not a reproduction of a published recursive subset relaxation.
    """
    generator = torch.Generator(device=scores.device).manual_seed(int(seed))
    eps = torch.finfo(scores.dtype).eps
    uniform = torch.rand(scores.shape, device=scores.device, dtype=scores.dtype,
                         generator=generator).clamp(eps, 1 - eps)
    return torch.logit(scores.clamp(eps, 1 - eps)) - torch.log(-torch.log(uniform))


class _BudgetSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, graphs, budgets, temperature):
        # Solve all molecule thresholds together; no per-edge dense adjacency
        # or repeated CPU synchronization inside the bisection loop.
        low = (scores.min() - 40 * temperature).expand_as(budgets).clone()
        high = (scores.max() + 40 * temperature).expand_as(budgets).clone()
        for _ in range(60):
            threshold = (low + high) / 2
            weights = torch.sigmoid((scores - threshold[graphs]) / temperature)
            totals = scores.new_zeros(budgets.numel()).index_add(0, graphs, weights)
            above = totals > budgets
            low = torch.where(above, threshold, low)
            high = torch.where(above, high, threshold)
        weights = torch.sigmoid((scores - ((low + high) / 2)[graphs]) / temperature)
        counts = torch.bincount(graphs, minlength=budgets.numel())
        weights = torch.where(budgets[graphs] == counts[graphs], torch.ones_like(weights), weights)
        weights = torch.where(budgets[graphs] == 0, torch.zeros_like(weights), weights)
        ctx.save_for_backward(weights, graphs)
        ctx.num_graphs = budgets.numel()
        ctx.temperature = temperature
        return weights

    @staticmethod
    def backward(ctx, gradient):
        weights, graphs = ctx.saved_tensors
        slope = weights * (1 - weights) / ctx.temperature
        denominator = slope.new_zeros(ctx.num_graphs).index_add(0, graphs, slope)
        numerator = slope.new_zeros(ctx.num_graphs).index_add(0, graphs, gradient * slope)
        mean = numerator / denominator.clamp_min(1e-20)
        return slope * (gradient - mean[graphs]), None, None, None


def budget_soft_mask(scores, edge_batch, hard_mask, eligible, temperature):
    if not 0 < temperature < float("inf"):
        raise ValueError("selection_temperature must be finite and positive")
    soft = torch.zeros_like(scores)
    indices = eligible.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return soft
    graphs = edge_batch[indices]
    num_graphs = int(edge_batch.max()) + 1
    budgets = scores.new_zeros(num_graphs).index_add(0, edge_batch, hard_mask.to(scores.dtype))
    soft[indices] = _BudgetSigmoid.apply(scores[indices], graphs, budgets, temperature)
    return soft


def masked_atom_graph(batch, fragment_graph, scores, hard_mask, eligible, method, temperature,
                      message_gate_mode="learned", selection_scores=None):
    soft = budget_soft_mask(scores if selection_scores is None else selection_scores, fragment_graph.batch[fragment_graph.edge_index[0]],
                            hard_mask, eligible, temperature)
    selection = hard_mask.to(scores.dtype) + (soft - soft.detach())
    endpoints = fragment_graph.edge_index
    n = fragment_graph.x.size(0)
    hard_nodes = torch.zeros(n, dtype=torch.bool, device=scores.device)
    hard_nodes[endpoints[:, hard_mask].reshape(-1)] = True
    # A fragment is present if ANY incident selected connection is present.
    complement = torch.ones(n, dtype=scores.dtype, device=scores.device)
    if endpoints.numel():
        complement = complement.scatter_reduce(
            0, endpoints.reshape(-1), (1 - soft).repeat(2), reduce="prod", include_self=True)
    soft_nodes = 1 - complement
    graph_has_edge = torch.zeros(batch.num_graphs, dtype=torch.bool, device=scores.device)
    graph_has_edge[fragment_graph.batch[endpoints[0]]] = True
    fallback = ~graph_has_edge[fragment_graph.batch]
    hard_nodes = hard_nodes | fallback
    soft_nodes = torch.where(fallback, torch.ones_like(soft_nodes), soft_nodes)
    node_mask = hard_nodes.to(scores.dtype) + (soft_nodes - soft_nodes.detach())
    atom_to_fragment = get_fragment_view_value(batch, "atom_to_fragment", method)
    atom_mask = node_mask[atom_to_fragment]
    src, dst = batch.bond_index
    internal = atom_to_fragment[src] == atom_to_fragment[dst]
    importance = fragment_soft_importance(n, endpoints, scores)
    bond_weight = scores.new_zeros(src.numel())
    internal_weights = node_mask if message_gate_mode == "unit" else importance * node_mask
    bond_weight[internal] = internal_weights[atom_to_fragment[src[internal]]]
    bond_ids = get_fragment_view_value(batch, "fragment_edge_to_bond_index", method)
    bond_weight[bond_ids] = selection if message_gate_mode == "unit" else selection * scores
    return Data(x=batch.x, edge_index=torch.cat([batch.bond_index, batch.bond_index.flip(0)], 1),
                edge_attr=batch.bond_attr.repeat(2, 1), edge_weight=bond_weight.repeat(2),
                batch=batch.batch, node_selection_mask=atom_mask), soft
