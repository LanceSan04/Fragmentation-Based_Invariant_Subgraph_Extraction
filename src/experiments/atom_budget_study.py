"""Validation-only retained-atom budget study: three arms, five paired seeds."""
import argparse
import hashlib
import math
from pathlib import Path
import time
import numpy as np
import torch
from src.experiments import selection
from src.experiments.assay_followup import restore
from src.experiments.retention_audit import measure
from src.data.dataloader import load_processed_split
from src.experiments.atom_budget_recovery import parity_report, recover_reference

ARMS = {'reference': ('edges', 'learned'), 'atom_learned': ('atoms', 'learned'),
        'atom_random_rank': ('atoms', 'random')}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-dir')
    p.add_argument('--raw-path')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--replay-records', help='Reuse four recorded validation replays after verifying checkpoint/config identity')
    p.add_argument('--reuse-reference', help='Prior study results directory; validate and reuse completed seed8401 reference')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--pilot', action='store_true')
    a = p.parse_args(argv)
    if not a.smoke and not (a.processed_dir and a.raw_path):
        p.error('Real study requires processed-dir and raw-path')
    if a.replay_records and not a.reuse_reference: p.error('replay-records requires reuse-reference')
    if a.reuse_reference and (a.smoke or a.pilot): p.error('Reference recovery requires the full fixed study')
    torch.set_num_threads(1 if a.smoke else 4)
    out = Path(a.output_dir).resolve()
    if out.exists(): raise FileExistsError(out)
    cache = Path(a.processed_dir).resolve() if a.processed_dir else out/'processed'
    base = selection.base_config('assay', cache, a.device)
    base.update(processed_dir=str(cache), epochs=3 if a.smoke else 1 if a.pilot else 100,
                learning_rate=.0003, dropout=.5, weight_decay=0., num_views=5,
                edge_ratio=.5, fragment_sample_ratio=.7, lambda_contrast=.1,
                lambda_diversity=0., lambda_irm=0., contrast_scope='pooled',
                selection_mode='learned', scoring_mode='ordered', atom_retention_ratio=.65,
                inference_policy='independent', early_stopping=False,
                loader_split_names=['train','ood_val'], test_split=None, report_splits=[],
                log_gradient_influence=True)
    if a.raw_path: base['raw_path'] = str(Path(a.raw_path).resolve())
    if a.smoke:
        _, fixture = selection.make_plan(out, a.device, True)
        for key in ['dataset','raw_path','raw_format','splits','label_key','processed_dir',
                    'strict','batch_size','num_workers','pin_memory','hidden_dim','gnn_layers']:
            base[key] = fixture[key]
        base['split_names'] = ['train', 'ood_val']
        base['splits'] = {k: fixture['splits'][k] for k in base['split_names']}
    else:
        selection.verify_study_cache(cache, a.raw_path)
    seeds = [8401] if a.smoke or a.pilot else list(range(8401,8406))
    plan = dict(seeds=seeds, epochs=base['epochs'], training_runs=3*len(seeds),
                arms=ARMS, target_atom_fraction=.65, synthetic=a.smoke, resource_pilot=a.pilot,
                primary='independent/independent', evaluate_test=False,
                rule='Nearest nonempty ranked prefix; ties prefer fewer atoms then fewer edges; whole fragments.',
                success_screen='Mean paired validation gain >= .005 and positive in at least 4/5 seeds; not significance.',
                retention_warning='Learned/random mean atom fractions differing > .02 are not closely size-matched.',
                random_control='Random priorities only; learned scorer, message gates and pooled contrast remain active.',
                base_config=base)
    if not a.smoke:
        plan['data_sha256'] = {s:hashlib.sha256((cache/f'{s}.pt').read_bytes()).hexdigest() for s in ['train','ood_val']}
        plan['raw_sha256'] = hashlib.sha256(Path(a.raw_path).read_bytes()).hexdigest()
    plan['new_training_runs'] = plan['training_runs'] - int(bool(a.reuse_reference))
    plan['reuse_reference'] = str(Path(a.reuse_reference).resolve()) if a.reuse_reference else None
    out.mkdir(parents=True)
    selection.write(out/'plan.json', plan)
    if a.smoke: selection.run_preprocessing(selection.Config(base))
    records = {arm:[] for arm in ARMS}
    # Interleave arms within seed to keep resource drift from lining up with an arm.
    for seed in seeds:
        for arm, (budget, ranking) in ARMS.items():
            cfg = dict(base, selection_budget=budget, ranking_mode=ranking)
            folder = out/arm/f'seed{seed}'
            if a.device == 'cuda': torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            reused = bool(a.reuse_reference and arm=='reference' and seed==8401)
            result = recover_reference(a.reuse_reference, folder, cfg, plan, a.device, a.replay_records) if reused else selection.run_seed(cfg, folder, seed, policies=('independent',))
            selection.write(folder/'checkpoint_parity.json', parity_report(result))
            resource = dict(training_repeated=not reused, training_validation_seconds=None if reused else time.monotonic()-start,
                            recovery_seconds=time.monotonic()-start if reused else None,
                            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
                            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if a.device=='cuda' else None)
            selection.write(folder/'resources.json', resource)
            # Same 100 size-stratified validation molecules in every arm/seed.
            saved_config, views = restore(folder/'best_independent.pt', Path(base['processed_dir']), 'cpu')
            data = load_processed_split(Path(base['processed_dir'])/'ood_val.pt')['molecule_data']
            geometry = measure(views, saved_config, data, count=4 if a.smoke else 100)
            for row in geometry['rows']:
                row['target_atoms'] = math.ceil(.65*row['original_atoms']) if budget=='atoms' else None
                row['atom_budget_residual'] = len(row['atom_ids'])-row['target_atoms'] if budget=='atoms' else None
                row['edgeless_fallback'] = row['full_connections']==0
            selection.write(folder/'retention.json', geometry)
            records[arm].append(result)
            selection.write(out/'progress.json', {k:len(v) for k,v in records.items()})
    aucs = {arm:[r['results']['independent']['independent']['ood_val']['auc'] for r in rows] for arm,rows in records.items()}
    summary = {}
    for arm, values in aucs.items():
        delta = np.asarray(values)-np.asarray(aucs['reference'])
        summary[arm] = dict(aucs=values,mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else None,
                            paired_deltas=delta.tolist(),mean_delta=float(delta.mean()),positive_seeds=int((delta>0).sum()),
                            passes_screen=bool(len(values)==5 and delta.mean()>=.005 and (delta>0).sum()>=4))
    retention = {}
    for arm in ARMS:
        rows = []
        for seed in seeds:
            import json
            rows.extend(json.loads((out/arm/f'seed{seed}'/'retention.json').read_text())['rows'])
        active = [r for r in rows if not r['edgeless_fallback']]
        retention[arm] = dict(mean_atom_fraction=float(np.mean([r['atom_fraction'] for r in active])) if active else None,
                             fallback_rows=sum(r['edgeless_fallback'] for r in rows), rows=len(rows),
                             sample='Same size-stratified validation molecules, descriptive not population weighted')
    learned_fraction = retention['atom_learned']['mean_atom_fraction']
    random_fraction = retention['atom_random_rank']['mean_atom_fraction']
    gap = abs(learned_fraction-random_fraction) if learned_fraction is not None and random_fraction is not None else None
    selection.write(out/'retention_summary.json', dict(arms=retention,learned_random_fraction_gap=gap,
                                                      size_matching_warning=gap is None or gap>.02))
    selection.write(out/'summary.json', summary)
    selection.write(out/'complete.json', dict(complete=True,plan=plan))
    print(f'COMPLETE: {out}', flush=True)
