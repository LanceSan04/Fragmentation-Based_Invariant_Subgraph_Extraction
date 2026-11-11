"""Synthetic motif-presence engineering check; not an OOD benchmark."""

import argparse
import hashlib
import json
import statistics
from itertools import combinations
from copy import copy
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Batch
from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data
from src.fragmentation import build_fragmenter
from src.models.fragment_ciga import FragmentCIGA
from src.experiments.selection import write

TRAIN = ("CCOC(=O)C", "CC(=O)NC", "CCC(=O)O", "CC(=O)CC", "CCOC", "CCNCC", "CCCCO", "CC(C)CC")
HELD_OUT = ("CCCC(=O)OC", "CCNC(=O)CC", "CCOCC", "CCNCCC")


def fixture(smiles, noise, seed, motif_smarts="[CX3]=[OX1]"):
    generator = torch.Generator().manual_seed(seed)
    query = Chem.MolFromSmarts(motif_smarts)
    data, motifs = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        matches = mol.GetSubstructMatches(query)
        fragments = build_fragmenter("group_graph").fragment(mol)
        target = sorted({int(fragments.atom_to_fragment[atom]) for match in matches for atom in match})
        label = int(bool(matches))
        if torch.rand((), generator=generator) < noise:
            label = 1 - label
        data.append(molecule_example_to_data(build_molecule_example(smi, label=label,
            fragmenter=build_fragmenter("group_graph"))))
        motifs.append(target)
    return Batch.from_data_list(data, follow_batch=["fragment_x_static"]), motifs


def selection_trace(batch, targets, details):
    """Count complete motif fragments and retain provenance per molecule."""
    chosen = set(details["selected_atom_ids"].tolist())
    rows = []
    for i, fragments in enumerate(targets):
        start, end = int(batch.ptr[i]), int(batch.ptr[i + 1])
        fragment_ids = batch.atom_to_fragment[start:end] - int(batch.fragment_x_static_ptr[i])
        motif_atoms = [start + j for j, f in enumerate(fragment_ids.tolist()) if f in fragments]
        retained = [a - start for a in chosen if start <= a < end]
        rows.append(dict(index=i, atom_ids=sorted(retained), atom_fraction=len(retained)/(end-start),
                         motif_retained=all(a in chosen for a in motif_atoms) if motif_atoms else None))
    positive = [row["motif_retained"] for row in rows if row["motif_retained"] is not None]
    return dict(motif_retention=sum(positive)/len(positive),
                mean_atom_fraction=statistics.mean(row["atom_fraction"] for row in rows),
                selected_edge_ids=details["selected_fragment_mask"].nonzero().flatten().tolist(),
                per_molecule=rows)


def frozen_selection_draws(model, batch, targets, seed, draws=20):
    """Freeze encoders/gates; alter only edge priorities, never train-mode BN."""
    from src.models.learnable_selection import gumbel_ranks
    results = []
    original = model.select_fragment_edges
    ranking_mode = model.ranking_mode
    model.eval()
    try:
        # Explicit priorities also serve random models; avoid their rank override.
        model.ranking_mode = "learned"
        for draw in range(draws):
            def select(data, fragments, scores, eligible, ranking_scores=None):
                if ranking_mode == "random" or model.selection_mode == "random":
                    generator = torch.Generator(device=scores.device).manual_seed(seed + 700003 + draw)
                    ranks = torch.rand(scores.shape, device=scores.device, generator=generator)
                else:
                    ranks = gumbel_ranks(scores, seed + 700003 + draw)
                return original(data, fragments, scores, eligible, ranking_scores=ranks)
            with torch.no_grad(), patch.object(model, "select_fragment_edges", side_effect=select):
                pred, details = model(batch, return_intermediates=True)
            results.append(dict(draw=draw, **selection_trace(batch, targets, details),
                auc=float(roc_auc_score(batch.y.reshape(-1).numpy(), pred.exp()[:, 1].numpy()))))
    finally:
        model.ranking_mode = ranking_mode
    return dict(draws=results, mean_retention=statistics.mean(r["motif_retention"] for r in results),
        mean_auc=statistics.mean(r["auc"] for r in results),
        note="Mean per-draw metrics, not an averaged-prediction ensemble. Full encoders in eval mode; only priorities change.")


def run_seed_check(a, out, train_smiles, held_smiles):
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    motif = "[CX2]#[NX1]" if getattr(a, "motif", "carbonyl") == "nitrile" else "[CX3]=[OX1]"
    train, train_targets = fixture(train_smiles, a.label_noise, a.seed, motif)
    held, targets = fixture(held_smiles, 0, a.seed, motif)
    reference = FragmentCIGA(atom_feature_dim=train.x.size(1),
        fragment_feature_dim=train.fragment_x_static.size(1),
        fragment_edge_dim=train.fragment_edge_attr.size(1), atom_edge_dim=train.edge_attr.size(1),
        gnn_hidden_dim=16, num_gnn_layers=2, edge_ratio=a.edge_ratio, fragment_sample_ratio=1.0,
        sampler_seed=a.seed, dropout=0.0)
    report = dict(synthetic=True, purpose="gradient/selection engineering check, not OOD evidence",
        motif=motif + "; retain all fragments containing motif atoms",
        epochs=a.epochs, seed=a.seed, noise=a.label_noise, train_smiles=train_smiles,
        held_out_smiles=held_smiles, edge_ratio=a.edge_ratio, sample_ratio=1.0, arms={})
    arms = ["hard", "straight_through", "random"] + (["gumbel_st"] if getattr(a, "include_gumbel", False) else [])
    if getattr(a, "selection_traces", False):
        arms.append("random_learned_gates")
    for arm in arms:
        model = deepcopy(reference)
        model.selection_estimator = "hard" if arm in ("random", "random_learned_gates") else arm
        model.selection_mode = "random" if arm == "random" else "learned"
        model.ranking_mode = "random" if arm == "random_learned_gates" else "learned"
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        history = []
        for epoch in range(1, a.epochs + 1):
            model.train(); optimizer.zero_grad()
            pred, details = model(train, return_intermediates=True)
            loss = F.nll_loss(pred, train.y.reshape(-1).long())
            loss.backward()
            gradients = sum(float(p.grad.norm()) for p in model.fragment_masker.parameters() if p.grad is not None)
            spread = []
            if details["fragment_scores"] is not None:
                scores = details["fragment_scores"].detach()
                for graph in torch.unique(details["fragment_edge_batch"]):
                    subset = scores[details["fragment_edge_batch"] == graph]
                    spread.append(float(subset.max() - subset.min()))
            history.append(dict(loss=float(loss.detach()), scorer_gradient_norm_sum=gradients,
                                within_molecule_score_ranges=spread))
            if getattr(a, "selection_traces", False):
                history[-1]["selection"] = selection_trace(train, train_targets, details)
            optimizer.step()
            if getattr(a, "selection_traces", False) and epoch in {25, 50, 75, a.epochs}:
                torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    epoch=epoch, arm=arm, seed=a.seed, edge_ratio=a.edge_ratio,
                    train_smiles=train_smiles, held_smiles=held_smiles, motif=motif,
                    ranking_call_count=model._ranking_call_count,
                    architecture=dict(hidden_dim=16, layers=2, temperature=model.selection_temperature)),
                    out / f"{arm}_epoch{epoch}.pt")
        model.eval()
        with torch.no_grad():
            pred, details = model(held, return_intermediates=True)
        selected = set(details["selected_atom_ids"].tolist())
        positive = []
        molecule_rows = []
        for i, fragments in enumerate(targets):
            if fragments:
                mol = Chem.MolFromSmiles(held_smiles[i])
                result = build_fragmenter("group_graph").fragment(mol)
                atoms = [int(held.ptr[i]) + atom for f in fragments for atom in result.fragment_to_atoms[f].tolist()]
                retained = all(atom in selected for atom in atoms)
                positive.append(retained)
                molecule_rows.append(dict(index=i, smiles=held_smiles[i], motif_fragments=fragments,
                    motif_fragment_atoms=len(atoms), motif_retained=retained))
        report["arms"][arm] = dict(history=history, held_out_accuracy=float(
            (pred.argmax(1) == held.y.reshape(-1)).float().mean()),
            motif_fragment_retention=sum(positive) / len(positive), positive_molecules=len(positive),
            retained_atoms=details["selected_atom_ids"].numel(), total_atoms=held.x.size(0),
            held_out_auc=float(roc_auc_score(held.y.numpy().reshape(-1), pred.exp()[:, 1].numpy())),
            per_positive_molecule=molecule_rows,
            predictions=pred.exp()[:, 1].tolist())
        if getattr(a, "selection_traces", False):
            report["arms"][arm]["deterministic_trace"] = selection_trace(held, targets, details)
            report["arms"][arm]["frozen_noisy_draws"] = frozen_selection_draws(model, held, targets, a.seed)
            with torch.no_grad():
                repeated = model(held)
            torch.testing.assert_close(pred, repeated, rtol=0, atol=0)
        if getattr(a, "information_path_check", False):
            from src.experiments.motif_information import information_paths
            report["arms"][arm]["information_paths"] = information_paths(model, held)
            torch.save(dict(state_dict=model.state_dict(), arm=arm, seed=a.seed,
                            edge_ratio=a.edge_ratio, epochs=a.epochs), out / f"{arm}.pt")
    write(out / "motif_check.json", report)
    print(f"Saved engineering check: {out / 'motif_check.json'}")
    return report


TRAIN_PREFIXES = ("CC", "CCC", "CCCC", "CC(C)C", "CC(C)CC", "c1ccccc1C", "C1CCCCC1C", "COCC")
HELD_PREFIXES = ("CCCCC", "CC(C)(C)C", "c1ccncc1C", "C1CCCC1C")


def expanded_fixture():
    """Fixed valid molecular structures, with matched positive/negative atom counts.

    These are constructed structures, not experimentally labelled molecules.
    Prefix families are held out; canonical identity overlap is forbidden.
    Similar chemistry across splits remains intentional in this engineering task.
    """
    seen = set()
    result = []
    for prefixes in (TRAIN_PREFIXES, HELD_PREFIXES):
        split = []
        for prefix in prefixes:
            for length in range(1, 5):
                suffix = "C" * length
                pair = (prefix + "C(=O)" + suffix, prefix + "OC" + suffix)
                mols = [Chem.MolFromSmiles(s) for s in pair]
                if any(m is None for m in mols):
                    raise ValueError("Invalid generated molecular structure")
                canonical = [Chem.MolToSmiles(m) for m in mols]
                if any(s in seen for s in canonical):
                    continue
                if mols[0].GetNumAtoms() != mols[1].GetNumAtoms():
                    raise AssertionError("Matched pair atom counts differ")
                seen.update(canonical)
                split.extend(pair)
        result.append(tuple(split))
    return tuple(result)


def challenging_fixture():
    """Longer multi-fragment structures with atom/topology-matched label pairs.

    Replace a carbonyl double bond with an alcohol single bond. Both structures
    have identical atom counts and unlabelled atom connectivity. Fragmentation
    can still differ; this remains an engineering task, not a causal benchmark.
    """
    seen, result = set(), []
    for prefixes in (TRAIN_PREFIXES, HELD_PREFIXES):
        split = []
        for prefix in prefixes:
            for length in range(1, 5):
                suffix = "C" * length
                pair = (prefix + "OCCNCCOC(=O)CCO" + suffix,
                        prefix + "OCCNCCOC(O)CCO" + suffix)
                mols = [Chem.MolFromSmiles(s) for s in pair]
                if any(m is None for m in mols):
                    raise ValueError("Invalid challenging fixture")
                canonical = [Chem.MolToSmiles(m) for m in mols]
                if any(s in seen for s in canonical):
                    continue
                if mols[0].GetNumAtoms() != mols[1].GetNumAtoms():
                    raise AssertionError("Atom counts differ")
                seen.update(canonical)
                split.extend(pair)
        result.append(tuple(split))
    return tuple(result)


def nitrile_fixture():
    """Terminal nitrile/amine pairs with matched heavy-atom counts and held prefixes."""
    seen, result = set(), []
    for prefixes in (TRAIN_PREFIXES, HELD_PREFIXES):
        split = []
        for prefix in prefixes:
            for length in range(1, 5):
                stem = prefix + "OCCNCCO" + "C" * length
                pair = (stem + "C#N", stem + "CN")
                mols = [Chem.MolFromSmiles(s) for s in pair]
                if any(m is None for m in mols):
                    raise ValueError("Invalid nitrile structure")
                canonical = [Chem.MolToSmiles(m) for m in mols]
                if len(set(canonical)) != 2 or any(s in seen for s in canonical):
                    raise ValueError("Nitrile identity overlap")
                if mols[0].GetNumAtoms() != mols[1].GetNumAtoms():
                    raise ValueError("Nitrile atom counts differ")
                seen.update(canonical)
                split.extend(pair)
        result.append(tuple(split))
    return tuple(result)


def retention_geometry(smiles, edge_ratio=0.5, motif_smarts="[CX3]=[OX1]"):
    """Exact uniform random top-k motif retention for this small fixed fixture."""
    query = Chem.MolFromSmarts(motif_smarts)
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        matches = mol.GetSubstructMatches(query)
        if not matches:
            continue
        fragments = build_fragmenter("group_graph").fragment(mol)
        target = {int(fragments.atom_to_fragment[a]) for match in matches for a in match}
        edges = fragments.fragment_edge_index.t().tolist()
        k = min(len(edges), max(1, int(len(edges) * edge_ratio)))
        retained, total = 0, 0
        for selection in combinations(range(len(edges)), k):
            nodes = {node for i in selection for node in edges[i]} if edges else set(range(fragments.num_fragments))
            retained += int(target <= nodes)
            total += 1
        rows.append(dict(smiles=smi, fragments=fragments.num_fragments,
            connections=len(edges), selected_connections=k,
            motif_fragments=len(target), possible_selections=total,
            uniform_random_retention_probability=retained / total))
    return dict(per_positive_molecule=rows,
        mean_uniform_random_retention=statistics.mean(r["uniform_random_retention_probability"] for r in rows),
        always_retained_molecules=sum(r["uniform_random_retention_probability"] == 1 for r in rows),
        positive_molecules=len(rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--label-noise", type=float, default=0.1)
    p.add_argument("--expanded", action="store_true")
    p.add_argument("--challenging", action="store_true")
    p.add_argument("--edge-ratio", type=float, default=0.5)
    p.add_argument("--information-path-check", action="store_true")
    p.add_argument("--include-gumbel", action="store_true", help="Add experimental noisy straight-through arm")
    p.add_argument("--motif", choices=["carbonyl", "nitrile"], default="carbonyl")
    p.add_argument("--selection-traces", action="store_true", help="Save fixed snapshots and frozen noisy draws; add random learned gates")
    a = p.parse_args(argv)
    if a.selection_traces and (not a.include_gumbel or a.information_path_check):
        p.error("Selection traces require Gumbel and no information-path interventions")
    if a.motif == "nitrile" and (not a.challenging or a.information_path_check):
        p.error("Nitrile requires challenging fixture and no information-path check")
    if a.include_gumbel and a.information_path_check:
        p.error("Gumbel information-path interventions are not yet validated")
    if a.information_path_check and not a.challenging:
        p.error("Information-path check requires the paired challenging fixture")
    seeds = a.seeds or [a.seed]
    if (a.epochs < 1 or not 0 <= a.label_noise < 0.5 or len(set(seeds)) != len(seeds)
            or not 0 < a.edge_ratio <= 1 or (a.expanded and a.challenging)):
        p.error("Positive epochs, unique seeds, and noise in [0, 0.5) required")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    train_smiles, held_smiles = (nitrile_fixture() if a.motif == "nitrile" else challenging_fixture() if a.challenging else
                                expanded_fixture() if a.expanded else (TRAIN, HELD_OUT))
    plan = dict(synthetic=True, expanded=a.expanded, challenging=a.challenging,
        edge_ratio=a.edge_ratio, seeds=seeds, epochs=a.epochs,
        information_path_check=a.information_path_check,
        arms=["hard", "straight_through", "random"] + (["gumbel_st"] if a.include_gumbel else []) +
            (["random_learned_gates"] if a.selection_traces else []),
        training_runs=((4 if a.include_gumbel else 3) + int(a.selection_traces)) * len(seeds),
        train_molecules=len(train_smiles), held_out_molecules=len(held_smiles), noise=a.label_noise,
        dataset_sha256=hashlib.sha256(json.dumps([train_smiles, held_smiles]).encode()).hexdigest(),
        checkpoint_rule="final fixed epoch, no held-out selection", no_ood_claim=True,
        interpretation="Functional learning check; no seed/epoch search; do not tune against held-out scores")
    if a.challenging:
        plan["proposed_engineering_support_rule"] = dict(
            st_minus_hard_retention_mean_at_least=0.1,
            st_minus_random_retention_mean_at_least=0.1,
            st_beats_hard_seeds_at_least=4,
            note="Fixed five-seed screening bar, not statistical significance or OOD evidence")
    if a.selection_traces:
        plan["gumbel_diagnostic_rule"] = "Final deterministic motif retention beats random learned gates by >=.10 mean and positive in >=4/5 seeds on EACH motif task. All other epochs/draws descriptive; no held-out selection or tuning. Not OOD evidence."
    write(out / "plan.json", plan)
    motif = "[CX2]#[NX1]" if a.motif == "nitrile" else "[CX3]=[OX1]"
    plan["motif_smarts"] = motif
    write(out / "plan.json", plan)
    geometry = retention_geometry(held_smiles, a.edge_ratio, motif)
    if a.challenging and (geometry["always_retained_molecules"] or
                         geometry["mean_uniform_random_retention"] > 0.5):
        raise ValueError("Challenging diagnostic requires discardable motifs and random retention <= 0.5")
    write(out / "retention_geometry.json", geometry)
    records = []
    for seed in seeds:
        args = copy(a)
        args.seed = seed
        target = out if len(seeds) == 1 else out / f"seed{seed}"
        if target != out:
            target.mkdir()
        print(f"Motif seed {seed}: {len(train_smiles)} train / {len(held_smiles)} held-out", flush=True)
        record = run_seed_check(args, target, train_smiles, held_smiles)
        records.append(record)
    def stats(values):
        return dict(mean=statistics.mean(values), sample_std=statistics.stdev(values) if len(values) > 1 else None,
                    values=values, n=len(values))
    summary = {arm: {metric: stats([r["arms"][arm][metric] for r in records])
               for metric in ["motif_fragment_retention", "held_out_auc", "held_out_accuracy"]}
               for arm in plan["arms"]}
    differences = [r["arms"]["straight_through"]["motif_fragment_retention"] -
                   r["arms"]["hard"]["motif_fragment_retention"] for r in records]
    write(out / "summary.json", dict(plan=plan, summary=summary,
        retention_geometry=geometry,
        paired_st_minus_hard_retention=stats(differences),
        positive_seeds=sum(v > 0 for v in differences),
        conclusion="Not automatically a gate pass; inspect retention feasibility and size confounding"))
    print(f"Complete: {out / 'summary.json'}", flush=True)
    return summary
