"""Validation-only frozen-checkpoint ranking/gating factorial diagnostic."""
import argparse
import json
from pathlib import Path
import statistics

import torch
from sklearn.metrics import roc_auc_score

from src.data.dataloader import build_split_dataloaders
from src.experiments.assay_followup import digest, restore, require_parity
from src.experiments.common import set_all_seeds
from src.experiments.selection import write
from src.models.atom_reconstructor import reconstruct_batch_fast
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.random_fragment_selection import random_edge_ranks
from train_sugar import Config, strip_non_tensor_batch_fields


def unit_gates(graph):
    """Change messages only, preserving every selected atom/bond and feature."""
    result = graph.clone()
    result.edge_weight = torch.ones_like(graph.edge_weight)
    return result


def run(entry, device, out):
    checkpoint = Path(entry['checkpoint'])
    if digest(checkpoint) != entry['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint hash mismatch')
    seed = entry['seed']
    set_all_seeds(seed)
    config, views = restore(checkpoint, entry['cache'], device)
    if any(v.erm_baseline or v.selection_mode != 'learned' or
           v.selection_budget != 'edges' or v.ranking_mode != 'learned' for v in views):
        raise ValueError('Expected learned edge-budget reference')
    config = Config(dict(config.to_dict(), loader_split_names=['ood_val']))
    loaders = build_split_dataloaders(entry['cache'], config)
    strip_non_tensor_batch_fields(loaders)
    modes = ['learned_rank_learned_gate', 'learned_rank_unit_gate'] + [
        f'random{draw}_rank_{gate}_gate' for draw in range(3) for gate in ['learned', 'unit']]
    predictions = {mode: [] for mode in modes}
    labels = []
    with torch.no_grad():
        for batch in loaders['ood_val']:
            batch = batch.to(device)
            probs = {mode: [] for mode in modes}
            for number, view in enumerate(views):
                fragment = build_fragment_graph_batch(batch, method=view.fragmentation_method).to(device)
                baseline, trace = view(batch, return_intermediates=True, fragment_batch=fragment)
                scores = trace['fragment_scores']
                mask = trace['selected_fragment_mask']
                original = reconstruct_batch_fast(batch, mask, scores, fragmentation_method=view.fragmentation_method)
                reconstructed = view.atom_classifier(original).softmax(-1)[:, 1]
                torch.testing.assert_close(reconstructed, baseline.exp()[:, 1], atol=1e-6, rtol=1e-5)
                probs[modes[0]].append(baseline.exp()[:, 1])
                probs[modes[1]].append(view.atom_classifier(unit_gates(original)).softmax(-1)[:, 1])
                edge_batch = trace['fragment_edge_batch']
                counts = torch.bincount(edge_batch[mask], minlength=batch.num_graphs)
                for draw in range(3):
                    ranks = random_edge_ranks(fragment, seed * 100 + number * 10 + draw, False)
                    random_mask = view.select_fragment_edges(batch, fragment, ranks, trace['fragment_edge_eligible'])
                    if not torch.equal(counts, torch.bincount(edge_batch[random_mask], minlength=batch.num_graphs)):
                        raise AssertionError('Random ranking changed fragment-edge budget')
                    graph = reconstruct_batch_fast(batch, random_mask, scores, fragmentation_method=view.fragmentation_method)
                    for gate, selected in [('learned', graph), ('unit', unit_gates(graph))]:
                        probs[f'random{draw}_rank_{gate}_gate'].append(view.atom_classifier(selected).softmax(-1)[:, 1])
            labels.extend(batch.y.reshape(-1).cpu().tolist())
            for mode in modes:
                predictions[mode].extend(torch.stack(probs[mode]).mean(0).cpu().tolist())
    aucs = {mode: float(roc_auc_score(labels, values)) for mode, values in predictions.items()}
    require_parity(aucs[modes[0]], entry['selected_auc'])
    if digest(checkpoint) != entry['checkpoint_sha256']:
        raise ValueError('Checkpoint changed during probe')
    result = dict(seed=seed, auc=aucs, molecules=len(labels), parity_passed=True,
                  checkpoint_sha256=entry['checkpoint_sha256'], effective_config=config.to_dict())
    write(out / 'metrics.json', result)
    write(out / 'predictions.json', dict(labels=labels, predictions=predictions))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    root, out = Path(args.reference_dir).resolve(), Path(args.output_dir).resolve()
    complete = json.loads((root / 'complete.json').read_text())
    if not complete['complete'] or bool(complete['synthetic']) != args.smoke:
        raise ValueError('Reference must be complete and match smoke/real mode')
    entries = [r for r in json.loads((root / 'frozen_checkpoints.json').read_text())
               if r['dataset'] == 'good_hiv_scaffold' and r['arm'] == 'learned']
    if sorted(r['seed'] for r in entries) != ([8700] if args.smoke else list(range(8701, 8706))):
        raise ValueError('Unexpected reference seeds')
    expected = json.loads((root / 'data_manifest.json').read_text())
    # Verify only the validation file; this diagnostic never deserializes test data.
    for entry in entries:
        cache = Path(entry['cache'])
        if digest(cache / 'ood_val.pt') != expected['good_hiv_scaffold']['files']['ood_val']:
            raise ValueError('Validation cache hash changed')
        if digest(entry['checkpoint']) != entry['checkpoint_sha256']:
            raise ValueError('Reference checkpoint changed')
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'plan.json', dict(entries=entries, reference_plan_sha256=digest(root / 'plan.json'),
        validation_sha256=digest(Path(entries[0]['cache']) / 'ood_val.pt'), synthetic=args.smoke,
        training_runs=0, test_loaded=False, random_draws=3,
        caveat='Frozen inference intervention, not retrained factorial. Edge counts matched; atoms/connectivity may differ across rankings. Random draws are not extra seeds.'))
    results = []
    for entry in entries:
        folder = out / f"seed{entry['seed']}"
        folder.mkdir()
        results.append(run(entry, args.device, folder))
        write(out / 'progress.json', results)
        print(entry['seed'], results[-1]['auc'], flush=True)
    summary = {mode: dict(mean=statistics.mean(r['auc'][mode] for r in results),
                         sample_std=statistics.stdev(r['auc'][mode] for r in results) if len(results)>1 else None)
               for mode in results[0]['auc']}
    write(out / 'summary.json', dict(validation_auc=summary, runs=results))
    write(out / 'complete.json', dict(complete=True, test_loaded=False, training_runs=0))
