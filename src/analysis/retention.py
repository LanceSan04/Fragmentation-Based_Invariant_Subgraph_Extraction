"""Frozen-weight geometry probes; sampling-only mode is not benchmark inference."""

from contextlib import contextmanager
from itertools import combinations
import numpy as np
import torch
from torch_geometric.data import Batch

from src.analysis.fragment_structure import describe, topology
from src.analysis.view_overlap import overlap


def size_bin(atoms):
    return "small_1_25" if atoms <= 25 else "medium_26_50" if atoms <= 50 else "large_51_plus"


@contextmanager
def probe_settings(view, edge_ratio, sample_ratio=None, seed=17):
    """Only enable the sampler; leave encoders/dropout/normalization in eval mode."""
    modes = {m: m.training for m in view.modules()}
    sampler = view.fragment_edge_sampler
    previous = (
        view.fragment_masker.ratio,
        sampler.keep_ratio,
        sampler.seed,
        sampler._call_count.clone(),
    )
    try:
        view.eval()
        view.fragment_masker.ratio = edge_ratio
        if sample_ratio is not None:
            sampler.keep_ratio = sample_ratio
            sampler.seed = seed
            sampler._call_count.zero_()
            sampler.train()
        yield
    finally:
        view.fragment_masker.ratio, sampler.keep_ratio, sampler.seed = previous[:3]
        sampler._call_count.copy_(previous[3])
        for module, training in modes.items():
            module.training = training


def stratified_indices(data, count):
    ordered = sorted(range(len(data)), key=lambda i: (data[i].x.size(0), i))
    return [
        ordered[i] for i in np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int)
    ]


def summarize_rows(rows):
    keys = [
        "atom_fraction",
        "bond_fraction",
        "components",
        "selected_connections",
        "eligible_connections",
        "full_connections",
        "selected_connection_fraction",
        "eligible_connection_fraction",
    ]
    return {key: describe([r[key] for r in rows if r[key] is not None]) for key in keys}


def measure(views, config, data, edge_ratio=0.5, sample_ratio=None, count=100, draw=0):
    """Use paired molecule IDs and random draws across ratios; do not calculate AUC."""
    indices = stratified_indices(data, count)
    rows, pairs = [], []
    with torch.no_grad():
        for index in indices:
            original = data[index]
            molecule = original.clone()
            for key in ["fragment_to_atoms", "smiles", "molecule_id", "metadata"]:
                if key in molecule:
                    del molecule[key]
            batch = Batch.from_data_list([molecule], follow_batch=config.follow_batch)
            states = []
            for number, view in enumerate(views):
                # Stable across ratios, but different across molecules/views/draws.
                seed = 17000000 + draw * 1000000 + index * 10 + number
                with probe_settings(view, edge_ratio, sample_ratio, seed):
                    _, trace = view(batch, return_intermediates=True)
                atoms = trace["selected_atom_ids"].tolist()
                bonds = trace["selected_bond_ids"].tolist()
                atom_map = {a: j for j, a in enumerate(atoms)}
                edges = [
                    (atom_map[a], atom_map[b]) for a, b in original.bond_index[:, bonds].T.tolist()
                ]
                full = trace["selected_fragment_mask"].numel()
                selected = int(trace["selected_fragment_mask"].sum())
                eligible = int(trace["fragment_edge_eligible"].sum())
                row = dict(
                    index=index,
                    view=number,
                    method=view.fragmentation_method,
                    original_atoms=original.x.size(0),
                    original_bonds=original.bond_index.size(1),
                    size_bin=size_bin(original.x.size(0)),
                    atom_ids=atoms,
                    bond_ids=bonds,
                    full_connections=full,
                    eligible_connections=eligible,
                    selected_connections=selected,
                    selected_connection_fraction=selected / full if full else None,
                    eligible_connection_fraction=eligible / full if full else None,
                    atom_fraction=len(atoms) / original.x.size(0),
                    bond_fraction=(
                        len(bonds) / original.bond_index.size(1)
                        if original.bond_index.size(1)
                        else None
                    ),
                    components=topology(len(atoms), edges)["components"],
                )
                rows.append(row)
                states.append(row)
            for a, b in combinations(states, 2):
                pairs.append(
                    dict(
                        index=index,
                        a=a["view"],
                        b=b["view"],
                        atom_jaccard=overlap(a["atom_ids"], b["atom_ids"]),
                        bond_jaccard=overlap(a["bond_ids"], b["bond_ids"]),
                    )
                )
    return dict(
        edge_ratio=edge_ratio,
        sample_ratio=sample_ratio,
        draw=draw,
        indices=indices,
        mode="evaluation" if sample_ratio is None else "sampler_only_probe",
        note="Frozen weights, dropout off, no gradients. Sampling probe is not a training run or benchmark prediction.",
        sample_rule="Evenly spaced ranks by original atom count; descriptive, not a random sample.",
        summary=summarize_rows(rows),
        by_size={
            key: summarize_rows([r for r in rows if r["size_bin"] == key])
            for key in sorted({r["size_bin"] for r in rows})
        },
        overlap={
            key: describe([p[key] for p in pairs if p[key] is not None])
            for key in ["atom_jaccard", "bond_jaccard"]
        },
        rows=rows,
        pairs=pairs,
    )
