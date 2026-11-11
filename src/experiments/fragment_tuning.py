"""Prepared, independently launchable DrugOOD searches; validation only."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import statistics

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.analysis.selection_faithfulness import main as coherence_pilot
from src.evaluation.within_assay import within_assay_report
from src.experiments.checkpoints import digest, evaluate_saved, require_parity
from src.experiments.common import set_all_seeds
from src.experiments.cross_domain import dataset_config, prepare_cache, verify_cache
from src.experiments.fragment_dataset_search import diagnostic_summary
from src.experiments.plans import ROOT, make_plan
from src.experiments.selection import run_seed
from src.experiments.sugar_reference_search import write

DATASETS = tuple(f'{endpoint}_{split}' for endpoint in ('ec50', 'ki')
                 for split in ('assay', 'scaffold', 'size'))
FIXED = dict(num_views=3, view_fragmentations={'group_graph': 3},
             lambda_diversity=.005, diversity_delay_epochs=0, diversity_ramp_epochs=0,
             lambda_irm=0., fragment_feature_mode='static_pool',
             inference_policy='independent', ranking_mode='learned',
             selection_estimator='hard', message_gate_mode='learned')


def validate(spec, reference):
    if digest(reference) != spec['reference_plan_sha256']:
        raise ValueError('Reference plan changed')
    bounded = spec['schema'] == 2
    datasets = {'ki_assay', 'ki_scaffold', 'ki_size'} if bounded else set(DATASETS)
    seeds = [10701, 10702] if bounded else [10501, 10502]
    if (spec['schema'] not in [1, 2] or spec['epochs'] != 150 or spec['seeds'] != seeds
            or spec['fixed'] != FIXED or set(spec['datasets']) != datasets):
        raise ValueError('Prepared study scope changed')
    if spec['columns'] != ['learning_rate', 'edge_ratio', 'fragment_sample_ratio', 'lambda_contrast']:
        raise ValueError('Unexpected search dimensions')
    count = 6 if bounded else 12
    if len(spec['recipes']) != count or len({tuple(r) for r in spec['recipes']}) != count:
        raise ValueError(f'Expected {count} unique recipes')
    if bounded:
        parent_path = ROOT/'configs/fragment_tuning_ec50_ki_20261018.json'
        parent = json.loads(parent_path.read_text())
        ids = [1, 2, 3, 4, 5, 12]
        if (spec['parent_config_sha256'] != digest(parent_path) or spec['recipe_ids'] != ids
                or spec['recipes'] != [parent['recipes'][i-1] for i in ids]
                or any(spec['datasets'][d] != parent['datasets'][d] for d in datasets)):
            raise ValueError('Bounded Ki subset/source changed')
    for lr, edge, sample, contrast in spec['recipes']:
        if (lr not in [.0001, .0003, .001] or edge not in [.3, .5, .7, .9]
                or sample not in [.5, .7, .9] or contrast not in [0, .1, 2]):
            raise ValueError('Recipe outside prepared ranges')


def contained(project, relative):
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError('Study data paths must be relative to --project-root')
    path = (project / relative).resolve()
    path.relative_to(project.resolve())
    return path


def build_config(spec, reference, dataset, project, device):
    entry = spec['datasets'][dataset]
    base = deepcopy(reference['configs']['ec50_scaffold'])
    data = dataset_config(dataset, project / 'data/raw', project / 'data/processed', device)
    for key in ('dataset', 'raw_format', 'label_key', 'smiles_key', 'metadata_keys',
                'environment_key', 'molecule_id_key', 'split_names', 'loader_split_names'):
        base[key] = deepcopy(data[key])
    base.update(spec['fixed'])
    base.update(raw_path=str(contained(project, entry['raw_path'])),
                processed_dir=str(contained(project, entry['cache_path'])),
                device=device, epochs=spec['epochs'], compare_checkpoint_rules=False,
                report_within_assay=False, log_gradient_influence=False,
                selection_mode='learned', full_graph_gating=False, erm_baseline=False,
                early_stopping=False, selection_ma_window=1, selection_weight_avg=False,
                test_split=None, report_splits=[], log_test_curve=False,
                train_split='train', validation_split='ood_val')
    return base


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/fragment_tuning_ec50_ki_20261018.json'))
    p.add_argument('--reference-plan', required=True)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, required=True)
    p.add_argument('--stage', choices=['prepare', 'pilot', 'search'], default='search')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(argv)
    spec = json.loads(Path(a.config).read_text()); validate(spec, a.reference_plan)
    if len(set(a.datasets)) != len(a.datasets): p.error('Duplicate datasets')
    if not set(a.datasets).issubset(spec['datasets']): p.error('Dataset outside this configuration')
    if spec['schema']==2 and a.stage!='search': p.error('Bounded Ki configuration is search-only')
    reference = json.loads(Path(a.reference_plan).read_text())
    recipes = spec['recipes'] if a.stage == 'search' else [spec['recipes'][i] for i in (0, 7)]
    if a.smoke: recipes = recipes[:2]
    recipe_ids = spec['recipe_ids'][:len(recipes)] if spec['schema']==2 else list(range(1, len(recipes)+1))
    seeds = spec['seeds'] if a.stage == 'search' and not a.smoke else spec['seeds'][:1]
    epochs = 2 if a.smoke else 10 if a.stage == 'pilot' else spec['epochs']
    configs = {d: build_config(spec, reference, d, a.project_root.resolve(), a.device) for d in a.datasets}
    plan = dict(stage=a.stage, datasets=a.datasets, seeds=seeds, epochs=epochs,
                recipes=recipes, recipe_ids=recipe_ids, columns=spec['columns'], fixed=spec['fixed'], configs=configs,
                training_runs=0 if a.stage == 'prepare' else len(a.datasets)*len(recipes)*len(seeds),
                synthetic=a.smoke, test_evaluated=False, automatic_continuation=False,
                config_sha256=digest(a.config), reference_plan_sha256=digest(a.reference_plan),
                checkpoint_rule='best_raw_pooled_validation_auc',
                within_assay='Reported only, >=5 examples per class; never selects a recipe',
                note='Independent commands still obey the account-wide single-pipeline guard')
    if not a.execute:
        print(json.dumps(plan, indent=2)); return plan
    if not a.smoke:
        for dataset, base in configs.items():
            if digest(base['raw_path']) != spec['datasets'][dataset]['raw_sha256']:
                raise ValueError('Pinned raw source changed: '+dataset)
    out = Path(a.output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    write(out / 'plan.json', plan); rows = []
    for dataset, base in configs.items():
        if shutil.disk_usage(out).free < 20*1024**3: raise RuntimeError('Low disk; stopped')
        if a.smoke:
            from src.data.preprocess import run_preprocessing
            from src.training.pipeline import Config
            torch.set_num_threads(1)
            _, fixture = make_plan(out / dataset, a.device, True)
            base.update(fixture); base.update(spec['fixed'])
            base.update(epochs=epochs, split_names=['train', 'ood_val'],
                        loader_split_names=['train', 'ood_val'], early_stopping=False)
            base['splits'].pop('ood_test', None)
            run_preprocessing(Config(base))
        elif a.stage == 'prepare':
            path = Path(base['processed_dir'])
            cache = verify_cache(base) if path.exists() else prepare_cache(base)
            write(out / (dataset+'_cache.json'), cache)
        else:
            verify_cache(base)
        if a.stage == 'prepare': continue
        base['epochs'] = epochs
        for index, recipe in zip(recipe_ids, recipes):
            for seed in seeds:
                values = deepcopy(base); values.update(dict(zip(spec['columns'], recipe)))
                run = out / dataset / f'recipe{index:02d}' / f'seed{seed}'
                run.parent.mkdir(parents=True, exist_ok=True)
                row = dict(dataset=dataset, recipe=index, parameters=dict(zip(spec['columns'], recipe)),
                           seed=seed, status='running')
                rows.append(row); write(out / 'status.json', rows)
                try:
                    if shutil.disk_usage(out).free < 20*1024**3: raise RuntimeError('Low disk; stopped')
                    set_all_seeds(seed)
                    result = run_seed(values, run, seed, False, ('independent',))
                    checkpoint = run / 'best_independent.pt'
                    predictions = run / 'validation_predictions.npz'
                    replay = evaluate_saved(checkpoint, 'ood_val', Path(base['processed_dir']),
                                            a.device, 'independent', seed, prediction_path=predictions)
                    require_parity(replay['auc'], result['best']['independent']['auc'])
                    with np.load(predictions) as saved:
                        # Missing assay identifiers are excluded, not treated as one screen.
                        assays = [str(x) if str(x) else None for x in saved['assays']]
                        within = within_assay_report(saved['labels'], saved['probabilities'], assays)
                        balance = dict(positive=int((saved['labels']==1).sum()),
                                       negative=int((saved['labels']==0).sum()),
                                       positive_fraction=float(np.mean(saved['labels'])),
                                       predicted_positive_fraction_at_0p5=float(np.mean(saved['probabilities']>=.5)),
                                       balanced_accuracy_at_0p5=float(.5*(
                                           np.mean(saved['probabilities'][saved['labels']==1]>=.5)+
                                           np.mean(saved['probabilities'][saved['labels']==0]<.5))),
                                       average_precision_positive=float(average_precision_score(
                                           saved['labels'], saved['probabilities'])),
                                       average_precision_negative=float(average_precision_score(
                                           1-saved['labels'], 1-saved['probabilities'])))
                    write(run / 'within_assay_validation.json', within)
                    write(run / 'validation_class_balance.json', balance)
                    threads = torch.get_num_threads()
                    try:
                        report = coherence_pilot(['--checkpoint', str(checkpoint),
                            '--processed-dir', base['processed_dir'], '--output-dir', str(run/'coherence'),
                            '--device', a.device, '--molecules', '3' if a.smoke else '20',
                            '--random-draws', '12' if a.smoke else '64', '--controls', '2' if a.smoke else '8',
                            '--max-fragments', '2' if a.smoke else '3',
                            '--visual-molecules', '2' if a.smoke else '5'])
                    finally: torch.set_num_threads(threads)
                    row.update(status='completed', validation_auc=replay['auc'], within_assay=within,
                               checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                               diagnostics=diagnostic_summary(report))
                except Exception as error:
                    row.update(status='failed', error=str(error)); write(out/'status.json', rows); raise
                write(out / 'status.json', rows)
        candidates = []
        for index, recipe in zip(recipe_ids, recipes):
            subset = [r for r in rows if r['dataset']==dataset and r['recipe']==index]
            scores = [r['validation_auc'] for r in subset]
            candidates.append(dict(recipe=index, parameters=dict(zip(spec['columns'], recipe)),
                                   mean_validation_auc=statistics.mean(scores),
                                   sample_sd=statistics.stdev(scores) if len(scores)>1 else None,
                                   diagnostics=[r['diagnostics'] for r in subset]))
        write(out / dataset / 'candidates.json', sorted(candidates, key=lambda r:-r['mean_validation_auc']))
    write(out / 'completed.json', dict(stage=a.stage, training_runs=len(rows), synthetic=a.smoke,
                                       test_evaluated=False, automatic_continuation=False))
    return rows
