"""Frozen EC50 or Ki confirmation with validation-gated test evaluation."""
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
from src.experiments.cross_domain import verify_cache
from src.experiments.fragment_dataset_search import diagnostic_summary
from src.experiments.fragment_tuning import build_config, main as tuning_main, validate as validate_search
from src.experiments.plans import ROOT, make_plan
from src.experiments.selection import run_seed
from src.experiments.sugar_reference_search import write

CANDIDATES = [('ec50_assay', 4), ('ec50_scaffold', 4), ('ec50_scaffold', 7), ('ec50_size', 3)]
SEEDS = [10601, 10602, 10603, 10604, 10605]
KI_CANDIDATES = [('ki_assay', 1), ('ki_scaffold', 12), ('ki_size', 3)]
KI_SEEDS = [10801, 10802, 10803, 10804, 10805]


def validate(spec, search, reference, search_path):
    validate_search(search, reference)
    ki = spec['schema'] == 2
    if (spec['schema'] not in (1, 2) or (ki and search['schema'] != 2)
            or (not ki and search['schema'] != 1)
            or spec['epochs'] != 150 or spec['seeds'] != (KI_SEEDS if ki else SEEDS)
            or [(r['dataset'], r['recipe']) for r in spec['candidates']] != (KI_CANDIDATES if ki else CANDIDATES)
            or spec['search_config_sha256'] != digest(search_path)):
        raise ValueError('Frozen confirmation protocol changed')
    recipe_ids = search.get('recipe_ids', list(range(1, len(search['recipes'])+1)))
    for row in spec['candidates']:
        expected = dict(zip(search['columns'], search['recipes'][recipe_ids.index(row['recipe'])]))
        if row['parameters'] != expected: raise ValueError('Candidate recipe changed')


def require_validation_winner(records, dataset, search, selected_recipe):
    """Require the complete frozen search, then choose on mean validation only."""
    evidence = [r for r in records if r['dataset']==dataset]
    expected = {(rid, seed) for rid in search['recipe_ids'] for seed in search['seeds']}
    observed = [(r['recipe'], r['seed']) for r in evidence]
    if (len(observed)!=len(expected) or set(observed)!=expected
            or any(r['status']!='completed' or not np.isfinite(r['validation_auc']) for r in evidence)):
        raise ValueError('Incomplete validation search evidence: '+dataset)
    means = {rid: statistics.mean(r['validation_auc'] for r in evidence if r['recipe']==rid)
             for rid in search['recipe_ids']}
    if max(means, key=means.get) != selected_recipe:
        raise ValueError('Candidate is not the validation-selected winner: '+dataset)


def prediction_report(path):
    with np.load(path) as saved:
        y, p = saved['labels'], saved['probabilities']
        assays = [str(x) if str(x) else None for x in saved['assays']]
        within = within_assay_report(y, p, assays)
        balance = dict(positive=int((y==1).sum()), negative=int((y==0).sum()),
                       positive_fraction=float(np.mean(y)),
                       average_precision_positive=float(average_precision_score(y, p)),
                       average_precision_negative=float(average_precision_score(1-y, 1-p)))
    return dict(within_assay=within, class_balance=balance)


def stats(values):
    values = [float(value) for value in values]
    return dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values)>1 else None,
                n=len(values), values=values)


def summarize(entries, candidate_ids=None):
    candidates = []
    candidate_ids = CANDIDATES if candidate_ids is None else candidate_ids
    for dataset, recipe in candidate_ids:
        rows = [r for r in entries if r['dataset']==dataset and r['recipe']==recipe]
        candidates.append(dict(dataset=dataset, recipe=recipe, parameters=rows[0]['parameters'],
            per_seed=[dict(seed=r['seed'], selected_epoch=r['epoch'], metrics=r['metrics'],
                           diagnostics=r.get('diagnostics')) for r in rows],
            summary={split: stats([r['metrics'][split]['auc'] for r in rows])
                     for split in ['ood_val', 'ood_test']}))
    result = dict(candidates=candidates,
                test_evaluated=True, no_automatic_recipe_selection=True,
                limitation='Fresh training seeds, previously inspected benchmark test sets; no learned-vs-random or ERM comparison in this batch. Seed SD is not uncertainty across new datasets.')
    if ('ec50_scaffold', 4) not in candidate_ids or ('ec50_scaffold', 7) not in candidate_ids:
        return result
    shared_seeds = sorted({r['seed'] for r in entries if r['dataset']=='ec50_scaffold'})
    paired = {}
    for split in ['ood_val', 'ood_test']:
        delta = []
        for seed in shared_seeds:
            scores = {r['recipe']: r['metrics'][split]['auc'] for r in entries
                      if r['dataset']=='ec50_scaffold' and r['seed']==seed}
            delta.append(scores[4]-scores[7])
        paired[split] = dict(stats(delta), seeds=shared_seeds, positive_seeds=sum(int(v>0) for v in delta))
    result['scaffold_recipe04_minus_recipe07'] = paired
    return result


def evaluate_batch(entries, out, device, smoke=False, diagnostics=True, candidate_ids=None, report_name='ec50_confirmation.json'):
    """Every frozen checkpoint must pass validation before any test forward."""
    write(out/'frozen_checkpoints.json', entries)
    for entry in entries:
        if digest(entry['checkpoint']) != entry['checkpoint_sha256']:
            raise ValueError('Checkpoint changed before validation')
        run = Path(entry['run_dir'])
        predictions = run/'validation_predictions.npz'
        val = evaluate_saved(entry['checkpoint'], 'ood_val', entry['cache'], device,
                             'independent', entry['seed'], prediction_path=predictions)
        require_parity(val['auc'], entry['selected_auc'])
        val.update(prediction_report(predictions)); entry['metrics'] = dict(ood_val=val)
        if diagnostics:
            threads = torch.get_num_threads()
            try:
                report = coherence_pilot(['--checkpoint', entry['checkpoint'],
                    '--processed-dir', entry['cache'], '--output-dir', str(run/'coherence'),
                    '--device', device, '--molecules', '3' if smoke else '20',
                    '--random-draws', '12' if smoke else '64', '--controls', '2' if smoke else '8',
                    '--max-fragments', '2' if smoke else '3', '--visual-molecules', '2' if smoke else '5'])
            finally: torch.set_num_threads(threads)
            entry['diagnostics'] = diagnostic_summary(report)
        write(run/'validation_metrics.json', val)
        write(out/'validation_progress.json', entries)
    # A failure anywhere above prevents the first test evaluation.
    for entry in entries:
        if digest(entry['checkpoint']) != entry['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed; no test evaluation allowed')
    for entry in entries:
        if digest(entry['checkpoint']) != entry['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed before test')
        run = Path(entry['run_dir']); predictions = run/'test_predictions.npz'
        test = evaluate_saved(entry['checkpoint'], 'ood_test', entry['cache'], device,
                              'independent', entry['seed'], prediction_path=predictions)
        test.update(prediction_report(predictions)); entry['metrics']['ood_test'] = test
        write(run/'test_metrics.json', test); write(out/'test_progress.json', entries)
    result = summarize(entries, candidate_ids); write(out/report_name, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/fragment_confirmation_20261019.json'))
    p.add_argument('--search-config', default=str(ROOT/'configs/fragment_tuning_ec50_ki_20261018.json'))
    p.add_argument('--reference-plan', required=True)
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--then-ki-pilots', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(argv); project = a.project_root.resolve()
    spec = json.loads(Path(a.config).read_text()); search = json.loads(Path(a.search_config).read_text())
    reference = json.loads(Path(a.reference_plan).read_text()); validate(spec, search, a.reference_plan, a.search_config)
    ki = spec['schema'] == 2
    if ki and a.then_ki_pilots: p.error('Ki confirmation cannot launch additional pilots')
    candidate_ids = KI_CANDIDATES if ki else CANDIDATES
    family = 'ki' if ki else 'ec50'
    seeds = spec['seeds'][:1] if a.smoke else spec['seeds']; epochs = 2 if a.smoke else 150
    plan = dict(candidates=spec['candidates'], seeds=seeds, epochs=epochs,
                training_runs=len(spec['candidates'])*len(seeds),
                ec50_training_runs=0 if ki else len(spec['candidates'])*len(seeds),
                ki_training_runs=len(spec['candidates'])*len(seeds) if ki else 6 if a.then_ki_pilots else 0,
                ki_epochs=epochs if ki else 2 if a.smoke else 10, ki_full_search=False,
                fixed=search['fixed'], synthetic=a.smoke, test_evaluated_for=family.upper()+' only',
                test_gate='All confirmation training, frozen checkpoint identities and validation replays complete before test',
                config_sha256=digest(a.config), search_config_sha256=digest(a.search_config),
                reference_plan_sha256=digest(a.reference_plan), no_retry=True)
    if not a.execute: print(json.dumps(plan, indent=2)); return plan
    bases = {}
    for dataset, selected_recipe in candidate_ids:
        if dataset in bases: continue
        base = build_config(search, reference, dataset, project, a.device)
        if not a.smoke:
            source = project/spec['search_status_paths'][dataset]
            source.resolve().relative_to(project)
            if digest(source) != spec['search_status_sha256'][dataset]:
                raise ValueError('Source search evidence changed: '+dataset)
            if ki:
                require_validation_winner(json.loads(source.read_text()), dataset, search, selected_recipe)
            if digest(base['raw_path']) != search['datasets'][dataset]['raw_sha256']:
                raise ValueError('Raw source changed: '+dataset)
            verify_cache(base)
        bases[dataset] = base
    out = Path(a.output_dir).resolve(); out.mkdir(parents=True, exist_ok=False)
    write(out/'plan.json', plan); entries = []; stage = family+'_training'
    try:
        for dataset, base in bases.items():
            if a.smoke:
                from src.data.preprocess import run_preprocessing
                from src.training.pipeline import Config
                torch.set_num_threads(1); _, fixture = make_plan(out/'fixtures'/dataset, a.device, True)
                base.update(fixture); base.update(search['fixed'])
                base['loader_split_names'] = ['train', 'ood_val']
                run_preprocessing(Config(base))
            base['epochs'] = epochs
        for candidate in spec['candidates']:
            dataset, recipe = candidate['dataset'], candidate['recipe']
            for seed in seeds:
                if shutil.disk_usage(out).free < 20*1024**3: raise RuntimeError('Low disk; stopped')
                values = deepcopy(bases[dataset]); values.update(candidate['parameters'])
                run = out/family/dataset/f'recipe{recipe:02d}'/f'seed{seed}'
                run.parent.mkdir(parents=True, exist_ok=True)
                write(out/'pipeline_status.json', dict(stage=stage, current=dict(candidate, seed=seed), trained=len(entries)))
                result = run_seed(values, run, seed, False, ('independent',))
                best = result['best']['independent']; checkpoint = run/'best_independent.pt'
                entries.append(dict(dataset=dataset, recipe=recipe, parameters=candidate['parameters'],
                    seed=seed, epoch=best['epoch'], selected_auc=best['auc'], checkpoint=str(checkpoint),
                    checkpoint_sha256=digest(checkpoint), cache=values['processed_dir'], run_dir=str(run)))
                write(out/'training_progress.json', entries)
        stage = family+'_validation_and_test'
        write(out/'pipeline_status.json', dict(stage=stage, trained=len(entries)))
        evaluate_batch(entries, out, a.device, a.smoke, candidate_ids=candidate_ids,
                       report_name=family+'_confirmation.json')
        if a.then_ki_pilots:
            common = ['--config', a.search_config, '--reference-plan', a.reference_plan,
                      '--project-root', str(project), '--datasets', 'ki_assay', 'ki_scaffold', 'ki_size',
                      '--device', a.device, '--execute']
            if a.smoke: common += ['--smoke']
            for stage in ['prepare', 'pilot']:
                write(out/'pipeline_status.json', dict(stage='ki_'+stage, ec50_completed=True))
                tuning_main(common+['--stage', stage, '--output-dir', str(out/('ki_'+stage))])
        write(out/'completed.json', dict(training_runs=len(entries),
            ec50_training_runs=0 if ki else len(entries), ec50_test_evaluated=not ki,
            ki_training_runs=len(entries) if ki else 6 if a.then_ki_pilots else 0,
            ki_pilot_training_runs=6 if a.then_ki_pilots else 0, ki_test_evaluated=ki,
            ki_full_search=False, synthetic=a.smoke))
        write(out/'pipeline_status.json', dict(stage='completed'))
    except Exception as error:
        write(out/'pipeline_status.json', dict(stage=stage, status='failed', error=str(error)))
        raise
    return entries
