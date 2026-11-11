"""Frozen Assay confirmation: 20 trajectories, optional primary-only test report."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics

import yaml

from src.experiments.checkpoints import digest, evaluate_saved, require_parity
from src.experiments.cross_domain import dataset_config, prepare_cache, verify_cache
from src.experiments.selection import run_seed
from src.experiments.study_queue import write, view_diagnostics
from src.training.checkpoint import load_checkpoint


def verify_selection(spec, source):
    if digest(source) != spec['selection_source_sha256']:
        raise ValueError('Selection source hash changed')
    status = json.loads(Path(source).read_text())
    if status['status'] != 'tune_completed' or status['decisions'] != spec['selection_decisions']:
        raise ValueError('Selection source does not match frozen decisions')
    if len(status['jobs']) != 58 or any(j['status'] != 'completed' for j in status['jobs'].values()):
        raise ValueError('Expected two pilots and 56 completed selection jobs')


def evaluate_primary(entries, output, device, evaluator=evaluate_saved, validation_only=False):
    """Validate every primary checkpoint before opening any test split."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for entry in entries:
        if digest(entry['checkpoint']) != entry['checkpoint_sha256']:
            raise ValueError('Frozen primary checkpoint changed')
        payload = load_checkpoint(entry['checkpoint'])
        if (payload['epoch'] != entry['epoch'] or payload['config']['seed'] != entry['seed']
                or payload['config']['inference_policy'] != 'independent'
                or payload['config']['checkpoint_selection_rule'] != 'best_raw'):
            raise ValueError('Primary checkpoint identity differs')
        observed = evaluator(entry['checkpoint'], 'ood_val', entry['cache'], device,
                             'independent', entry['seed'])
        require_parity(observed['auc'], entry['selected_auc'])
        rows.append(dict(entry, ood_val=observed))
        write(output/'validation_replays.json', rows)
    # Check the complete frozen set again after all validation replays pass.
    for entry in entries:
        if digest(entry['checkpoint']) != entry['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during validation replay')
    if validation_only:
        return rows
    for row in rows:
        prediction_path = output/'predictions'/f"{row['arm']}_{row['seed']}.npz"
        row['ood_test'] = evaluator(row['checkpoint'], 'ood_test', row['cache'], device,
                                    'independent', row['seed'], prediction_path=prediction_path)
        row['prediction_file'] = str(prediction_path)
        write(output/'test_progress.json', rows)
    arms = list(dict.fromkeys(e['arm'] for e in entries))
    summary = {}
    for arm in arms:
        values = [float(r['ood_test']['auc']) for r in rows if r['arm']==arm]
        summary[arm] = dict(n=len(values), mean=statistics.mean(values),
                           sample_std=statistics.stdev(values) if len(values)>1 else None)
    paired = {}
    for first,second in [('internal_gnn','static_pool'),('static_pool','random_learned_gates'),
                         ('internal_gnn','random_learned_gates'),('static_pool','erm'),('internal_gnn','erm')]:
        a = {r['seed']:r for r in rows if r['arm']==first}
        b = {r['seed']:r for r in rows if r['arm']==second}
        seeds = sorted(set(a)&set(b))
        differences = [float(a[s]['ood_test']['auc']-b[s]['ood_test']['auc']) for s in seeds]
        if differences:
            mean = statistics.mean(differences)
            sd = statistics.stdev(differences) if len(differences)>1 else None
            radius = 2.7764451052*sd/math.sqrt(5) if len(differences)==5 else None
            paired[first+'_minus_'+second] = dict(seeds=seeds,differences=differences,mean=mean,
                sample_std=sd,positive_seeds=sum(d>0 for d in differences),
                seed_t95=[mean-radius,mean+radius] if radius is not None else None)
    result = dict(primary_rule='best_raw',inference_policy='independent',per_seed=rows,
                  summary=summary,paired=paired,test_selects=False,
                  caveats='Fixed controls are not equally tuned. Seed intervals omit assay/molecule sampling and adaptive benchmark reuse. Secondary checkpoints are not test-evaluated.')
    write(output/'comparison.json',result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='configs/mentor_assay_confirmation.yaml')
    p.add_argument('--selection-source',help='Completed original queue status.json')
    cache=p.add_mutually_exclusive_group(required=True)
    cache.add_argument('--processed-dir')
    cache.add_argument('--cache-root')
    p.add_argument('--datasets',nargs='+',choices=['ec50_assay','ec50_scaffold','ec50_size',
                   'ki_scaffold','ki_size','good_hiv_scaffold','good_hiv_size'],
                   default=['ec50_assay'])
    p.add_argument('--raw-root',default='data/raw')
    p.add_argument('--source-audit',default='configs/ki_good_sources_20261016.yaml')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--evaluate-primary-test',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a = p.parse_args(argv)
    if len(set(a.datasets))!=len(a.datasets) or (len(a.datasets)>1 and a.processed_dir):
        p.error('Unique datasets required; multiple datasets require --cache-root')
    spec = yaml.safe_load(Path(a.manifest).read_text())
    if (spec['schema']!=1 or spec['primary_rule']!='best_raw' or spec['inference_policy']!='independent'
            or spec['seeds']!=list(range(9521,9526)) or spec['epochs']!=100
            or set(spec['arms'])!={'static_pool','internal_gnn','random_learned_gates','erm'}):
        raise ValueError('Frozen confirmation scope changed')
    if not a.smoke:
        if not a.selection_source:
            p.error('--selection-source required')
        verify_selection(spec,a.selection_source)
    out = Path(a.output_dir).resolve()
    configs={}
    for dataset in a.datasets:
      cache_path=Path(a.processed_dir).resolve() if a.processed_dir else Path(a.cache_root).resolve()/dataset
      base = dataset_config(dataset,a.raw_root,cache_path.parent,a.device)
      base.update(processed_dir=str(cache_path),epochs=100,
                compare_checkpoint_rules=True,report_within_assay=not dataset.startswith('good_'),
                log_gradient_influence=True,diagnostic_molecules=spec['diagnostic_molecules'],
                lambda_irm=0.0,diversity_delay_epochs=0,diversity_ramp_epochs=0)
      seeds = spec['seeds']
      if a.smoke:
        import torch
        from src.experiments.plans import make_plan
        torch.set_num_threads(1)
        _,fixture=make_plan(out/dataset,a.device,True)
        base.update(fixture)
        base.update(epochs=5,compare_checkpoint_rules=True,report_within_assay=False,
                    num_views=2,view_fragmentations={'group_graph':2},diagnostic_molecules=4)
        seeds=[17]
      configs[dataset]=base
    plan=dict(study='mentor_frozen_confirmation',datasets=a.datasets,seeds=seeds,epochs=base['epochs'],
              arms=spec['arms'],training_runs=len(seeds)*4*len(a.datasets),primary_rule='best_raw',
              inference_policy='independent',synthetic=a.smoke,test_selects=False,
              evaluate_primary_test=a.evaluate_primary_test,
              selection_source_sha256=spec['selection_source_sha256'],
              configs=configs,manifest_sha256=digest(a.manifest),
              primary_comparison='internal_gnn_minus_static_pool',
              interpretation='Compare frozen tuned configurations, not an isolated feature-encoder effect. No threshold significance claim or automatic tuning.')
    if a.dry_run:
        print(json.dumps(plan,indent=2));return plan
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',plan)
    if not a.smoke and any(d.startswith(('ki_','good_')) for d in configs):
        from src.experiments.dataset_readiness import audit_dataset
        expected=yaml.safe_load(Path(a.source_audit).read_text())['datasets']
        audits={}
        for dataset in configs:
            if not dataset.startswith(('ki_','good_')):continue
            observed=audit_dataset(dataset,a.raw_root)
            if observed['raw_sha256']!=expected[dataset]['raw_sha256'] or observed['needs_review']:
                raise ValueError('Dataset source changed or overlap/label diagnostics require review')
            for split,counts in expected[dataset]['splits'].items():
                if any(observed['splits'][split][key]!=value for key,value in counts.items()):
                    raise ValueError('Dataset split/class counts differ from frozen audit')
            audits[dataset]=observed
        write(out/'dataset_audit.json',audits)
    for base in configs.values():
        if Path(base['processed_dir']).exists():
            verify_cache(base)
        else:
            prepare_cache(base)
    primary,all_rules,resources=[],[],[]
    for dataset,base in configs.items():
      for seed in seeds:
        for arm,settings in spec['arms'].items():
            config=deepcopy(base);config.update(settings)
            directory=out/'runs'/dataset/arm/f'seed{seed}'
            result=run_seed(config,directory,seed,False,('independent',))
            if arm!='erm':
                view_diagnostics(dict(directory=str(directory),config=config,seed=seed))
            resources.append(dict(dataset=dataset,arm=arm,seed=seed,**result['resources']))
            for rule,record in result['checkpoint_rules']['records']['independent'].items():
                entry=dict(dataset=dataset,arm=arm,seed=seed,rule=rule,checkpoint=record['checkpoint'],
                    checkpoint_sha256=digest(record['checkpoint']),epoch=record['epoch'],
                    cache=config['processed_dir'],selected_auc=record['validation_auc_at_selection'])
                all_rules.append(entry)
                if rule=='best_raw':primary.append(entry)
            write(out/'training_progress.json',all_rules)
            write(out/'resources.json',resources)
    write(out/'frozen_primary.json',primary)
    write(out/'frozen_all_rules.json',all_rules)
    if a.evaluate_primary_test:
        # Gate every dataset before any test loader opens. Per-dataset reports
        # replay again to retain the original evaluator's standalone safeguards.
        if len(configs)>1:
            evaluate_primary(primary,out/'all_primary_replays',a.device,validation_only=True)
        reports={}
        for dataset in configs:
            entries=[e for e in primary if e['dataset']==dataset]
            target=out/'evaluation'/dataset if len(configs)>1 else out/'evaluation'
            reports[dataset]=evaluate_primary(entries,target,a.device)
        if len(configs)>1:
            write(out/'comparison_all.json',dict(datasets=reports,test_selects=False))
    write(out/'completed.json',dict(training_runs=plan['training_runs'],primary_checkpoints=len(primary),
          saved_checkpoints=len(all_rules),test_evaluated=a.evaluate_primary_test,synthetic=a.smoke))
    return plan


if __name__=='__main__':
    main()
