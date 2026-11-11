"""Frozen-model interventions for the constructed motif diagnostic only."""
import torch
from sklearn.metrics import roc_auc_score
from src.models.atom_reconstructor import reconstruct_batch_fast


def information_paths(model, batch):
    """Keep selection fixed while changing gates or classifier atom inputs.

    These distribution-changing interventions identify sensitivities, not the
    performance of retrained ablations or causal rationale faithfulness.
    """
    model.eval()
    with torch.no_grad():
        pred, details = model(batch, return_intermediates=True)
        mask = details["selected_fragment_mask"]
        scores = details["fragment_scores"]
        weights = mask.float() if scores is None else scores
        graph = reconstruct_batch_fast(batch, mask, selected_fragment_edge_weight=weights)
        original = model.atom_classifier(graph).softmax(-1)[:, 1]
        torch.testing.assert_close(original, pred.exp()[:, 1], atol=1e-6, rtol=1e-5)
        variants = {"original": original}
        for name, value in (("unit_gates_fixed_selection", 1.0),
                            ("half_gates_fixed_selection", 0.5)):
            changed = graph.clone()
            changed.edge_weight = torch.full_like(graph.edge_weight, value)
            variants[name] = model.atom_classifier(changed).softmax(-1)[:, 1]
        changed = graph.clone()
        changed.x = torch.zeros_like(graph.x)
        changed.edge_attr = torch.zeros_like(graph.edge_attr)
        variants["zero_atom_and_bond_features_keep_gates"] = model.atom_classifier(changed).softmax(-1)[:, 1]
        changed.edge_weight = torch.ones_like(graph.edge_weight)
        variants["zero_atom_and_bond_features_unit_gates"] = model.atom_classifier(changed).softmax(-1)[:, 1]
        # Counterfactual classifier inputs: exchange features of each matched
        # positive/negative pair without recomputing fragment scores/selection.
        # Original fragment partition and connection provenance stay frozen.
        counterfactual = batch.clone()
        for i in range(0, batch.num_graphs, 2):
            a, b, c = [int(batch.ptr[j]) for j in (i, i + 1, i + 2)]
            if b - a != c - b:
                raise ValueError("Requires paired equal atom counts")
            counterfactual.x[a:b] = batch.x[b:c]
            counterfactual.x[b:c] = batch.x[a:b]
            first = (batch.batch[batch.bond_index[0]] == i).nonzero().flatten()
            second = (batch.batch[batch.bond_index[0]] == i + 1).nonzero().flatten()
            if not torch.equal(batch.bond_index[:, first] - a, batch.bond_index[:, second] - b):
                raise ValueError("Requires identical paired atom connectivity/order")
            counterfactual.bond_attr[first] = batch.bond_attr[second]
            counterfactual.bond_attr[second] = batch.bond_attr[first]
        swapped = reconstruct_batch_fast(counterfactual, mask, selected_fragment_edge_weight=weights)
        torch.testing.assert_close(swapped.selected_original_atom_indices, graph.selected_original_atom_indices)
        variants["swap_pair_classifier_features_keep_selection_and_gates"] = model.atom_classifier(swapped).softmax(-1)[:, 1]
        swapped.edge_weight = torch.ones_like(swapped.edge_weight)
        variants["swap_pair_classifier_features_unit_gates"] = model.atom_classifier(swapped).softmax(-1)[:, 1]
        atom_fractions = []
        retained = set(graph.selected_original_atom_indices.tolist())
        for i in range(batch.num_graphs):
            a, b = int(batch.ptr[i]), int(batch.ptr[i + 1])
            atom_fractions.append(sum(j in retained for j in range(a, b)) / (b - a))
        labels = batch.y.reshape(-1).numpy()
        return dict(
            note="Frozen forward interventions, not retrained controls. Zero features are out of distribution. Pair swaps keep the original fragment partition.",
            variants={name: dict(auc=float(roc_auc_score(labels, p.numpy())),
                mean_absolute_probability_change=float((p - original).abs().mean()),
                probabilities=p.tolist()) for name, p in variants.items()},
            retained_atom_fraction=atom_fractions,
            positive_mean_atom_fraction=sum(atom_fractions[::2]) / len(atom_fractions[::2]),
            negative_mean_atom_fraction=sum(atom_fractions[1::2]) / len(atom_fractions[1::2]),
            labels=labels.tolist())
