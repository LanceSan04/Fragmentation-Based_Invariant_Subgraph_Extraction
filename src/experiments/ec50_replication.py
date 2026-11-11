"""Frozen EC50 seed-distribution study and paired scorer-and-classifier encoder comparison."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics

import numpy as np
import torch
from scipy.stats import t

from src.evaluation.within_assay import within_assay_report
from src.experiments.checkpoints import digest,evaluate_saved
from src.experiments.context_test import require_exact_predictions
from src.experiments.cross_domain import verify_cache
from src.experiments.fragment_tuning import contained
from src.experiments.original_tuning import (validate as validate_reference,require_complete,
    winners,retention,evaluate_single_arm,dataset_scope)
from src.experiments.plans import ROOT,make_plan
from src.experiments.selection import run_seed,validation_execution
from src.experiments.sugar_reference_search import write
from src.training.pipeline import Config

DATASETS=dataset_scope('ec50')
ENCODER=dict(fragment_feature_mode='molecular_gnn',fragment_encoder_layers=2,classifier_feature_mode='encoder')


def validate(spec):
    if (spec['schema']!=2 or spec['epochs']!=150 or spec['torch_threads']!=1
            or spec['static_seeds']!=list(range(11401,11431))
            or spec['shared_gnn_seeds']!=list(range(11401,11411))
            or spec['encoder']!=ENCODER or not spec['no_test_based_selection']
            or spec['fixed_recipes']!={'ec50_assay':1,'ec50_scaffold':2,'ec50_size':2}):
        raise ValueError('Frozen EC50 replication scope changed')


def seed_summary(values):
    values=[float(v) for v in values];n=len(values);mean=statistics.mean(values)
    sd=statistics.stdev(values) if n>1 else None
    half=float(t.ppf(.975,n-1))*sd/n**.5 if n>1 else None
    return dict(n=n,mean=mean,sample_sd=sd,median=statistics.median(values),
        minimum=min(values),maximum=max(values),
        mean_seed_ci95=[mean-half,mean+half] if half is not None else None,
        interpretation='Training-seed uncertainty conditional on fixed splits and recipe; not uncertainty across new molecules, assays or datasets.')


def study_scope(spec,study,smoke):
    seeds=spec['static_seeds'] if study=='static' else spec['shared_gnn_seeds']
    return dict(study=study,seeds=seeds[:2] if smoke else seeds,
        epochs=2 if smoke else spec['epochs'],datasets=DATASETS,
        training_runs=3*(2 if smoke else len(seeds)),
        feature_mode='static_pool' if study=='static' else 'molecular_gnn',
        classifier_feature_mode='raw' if study=='static' else 'encoder')


def compare_reports(static,gnn):
    if static['plan']['study']!='static' or gnn['plan']['study']!='shared-gnn':
        raise ValueError('Invalid paired arms')
    if (static['plan'].get('classifier_feature_mode')!='raw'
            or gnn['plan'].get('classifier_feature_mode')!='encoder'
            or gnn['plan'].get('feature_mode')!='molecular_gnn'):
        raise ValueError('Paired feature routing does not match the declared studies')
    a={(r['dataset'],r['seed']):r for r in static['test_evaluation']['runs']}
    summaries={}
    for dataset in DATASETS:
        rows=[r for r in gnn['test_evaluation']['runs'] if r['dataset']==dataset]
        deltas=[]
        for row in rows:
            key=(dataset,row['seed'])
            if key not in a:raise ValueError('Missing paired static seed')
            if row['recipe']!=a[key]['recipe']:raise ValueError('Unmatched recipes')
            deltas.append(dict(seed=row['seed'],static_auc=a[key]['metrics']['auc'],
                shared_gnn_auc=row['metrics']['auc'],difference=row['metrics']['auc']-a[key]['metrics']['auc']))
        summaries[dataset]=dict(pairs=deltas,**seed_summary(r['difference'] for r in deltas),
            positive_pairs=int(sum(r['difference']>0 for r in deltas)))
    return summaries


def check_membership(static_out,gnn_out,seeds):
    for dataset in DATASETS:
        for seed in seeds:
            a=static_out/'test_evaluation'/dataset/f'seed{seed}'/'test_predictions.npz'
            b=gnn_out/'test_evaluation'/dataset/f'seed{seed}'/'test_predictions.npz'
            with np.load(a) as first,np.load(b) as second:
                if any(not np.array_equal(first[k],second[k]) for k in ('labels','assays','row_index')):
                    raise ValueError('Paired test molecule membership differs')


def run_study(scope,spec,reference,project,out,device,smoke,deterministic_training=False):
    out.mkdir(parents=True,exist_ok=False)
    plan=dict(scope,protocol_sha256=digest(ROOT/'configs/ec50_replication_20261031.json'),
        fixed_recipes=spec['fixed_recipes'],synthetic=smoke,
        training_device=device,validation_device='cpu',test_planned=True,
        deterministic_training=deterministic_training,
        inference_policy='independent',checkpoint_rule='best_raw_pooled_cpu_validation_auc',
        no_test_based_selection=True)
    write(out/'plan.json',plan);rows=[];caches={};refs={}
    try:
        for entry in reference['datasets']:
            dataset=entry['dataset']
            if dataset not in scope['datasets']:continue
            base=deepcopy(reference['base']);rid=spec['fixed_recipes'][dataset]
            base.update(dict(zip(reference['columns'],reference['recipes'][rid-1])))
            base.update(dataset='drugood_lbap_core_'+dataset,device=device,epochs=scope['epochs'],
                raw_path=str(contained(project,entry['raw_path'])),processed_dir=str(contained(project,entry['cache_path'])),
                loader_split_names=['train','ood_val'],test_split=None,report_splits=[],log_test_curve=False)
            if deterministic_training:
                base['deterministic_training']=True
            if scope['study']=='shared-gnn':base.update(ENCODER)
            if (base['fragment_feature_mode']!=scope['feature_mode']
                    or base['classifier_feature_mode']!=scope['classifier_feature_mode']):
                raise ValueError('Runtime feature routing differs from the study plan')
            if smoke:
                from src.data.preprocess import run_preprocessing
                _,fixture=make_plan(out/dataset/'fixture',device,True)
                for key in ('dataset','raw_path','raw_format','splits','label_key','processed_dir',
                            'split_names','batch_size','num_workers','pin_memory'):
                    if key in fixture:base[key]=fixture[key]
                base.update(hidden_dim=16,gnn_layers=1,num_views=2,view_fragmentations={'group_graph':2},
                    num_workers=0,fragment_feature_mode=scope['feature_mode'],
                    classifier_feature_mode=scope['classifier_feature_mode'])
                run_preprocessing(Config(base))
            else:verify_cache(base)
            caches[dataset]=base['processed_dir']
            for seed in scope['seeds']:
                run=out/dataset/f'seed{seed}';run.parent.mkdir(parents=True,exist_ok=True)
                row=dict(dataset=dataset,recipe=rid,seed=seed,arm=scope['study'],status='running')
                rows.append(row);write(out/'status.json',rows)
                result=run_seed(base,run,seed,False,('independent',))
                checkpoint=run/'best_independent.pt';ref=run/'best_independent_validation_predictions.npz'
                meta=json.loads((run/'best_independent_validation_reference.json').read_text())
                if (meta['checkpoint_sha256']!=digest(checkpoint) or meta['prediction_sha256']!=digest(ref)
                        or meta['epoch']!=result['best']['independent']['epoch']
                        or meta['auc']!=result['best']['independent']['auc']):
                    raise ValueError('Selected checkpoint/reference pairing changed')
                with validation_execution('cpu'):
                    replay=evaluate_saved(checkpoint,'ood_val',Path(base['processed_dir']),'cpu','independent',seed,
                        prediction_path=run/'validation_replay.npz')
                require_exact_predictions(run/'validation_replay.npz',ref)
                row.update(status='completed',validation=replay,selected_epoch=meta['epoch'],
                    checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                    reference=str(ref),reference_sha256=digest(ref),resources=result['resources'],
                    retention=retention(checkpoint,base['processed_dir'],run/'retention.json'))
                refs[(dataset,scope['study'],seed)]=ref;write(out/'status.json',rows)
        tests=evaluate_single_arm(rows,caches,refs,out/'test_evaluation')
        report=dict(plan=plan,runs=rows,test_evaluated=True,test_evaluation=tests,
            summary={d:seed_summary(r['metrics']['auc'] for r in tests['runs'] if r['dataset']==d) for d in scope['datasets']})
        diagnostics=[]
        for row in tests['runs']:
            path=out/'test_evaluation'/row['dataset']/f"seed{row['seed']}"/'test_predictions.npz'
            with np.load(path) as arrays:
                diagnostics.append(dict(dataset=row['dataset'],seed=row['seed'],
                    within_assay=within_assay_report(arrays['labels'],arrays['probabilities'],arrays['assays'],5)))
        write(out/'within_assay_reported_only.json',diagnostics)
        write(out/'completed.json',report);return report
    except Exception as error:
        if rows and rows[-1]['status']=='running':rows[-1].update(status='failed',error=str(error))
        write(out/'status.json',rows);write(out/'failed.json',dict(error=str(error),runs=rows));raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/ec50_replication_20261031.json')
    p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--study',choices=['static','shared-gnn','both'],default='both')
    p.add_argument('--static-results',type=Path,help='Completed static-study directory, required for a standalone paired GNN study')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--deterministic-training',action='store_true',
                   help='Enable repeatable algorithms where available and record remaining warnings')
    a=p.parse_args(argv);spec=json.loads(a.config.read_text());validate(spec)
    # This study uses one fixed checked-in protocol; alternate files must be identical.
    if digest(a.config)!=digest(ROOT/'configs/ec50_replication_20261031.json'):p.error('Alternate protocol content changed')
    project=a.project_root.resolve();ref_path=ROOT/spec['reference_config']
    if digest(ref_path)!=spec['reference_config_sha256']:raise ValueError('Reference protocol changed')
    reference=json.loads(ref_path.read_text());validate_reference(reference)
    if not a.smoke:
        search_file=contained(project,spec['search_results'])/'completed.json'
        if digest(search_file)!=spec['search_completed_sha256']:raise ValueError('Frozen validation search changed')
        search=json.loads(search_file.read_text())
        require_complete(search,'search',digest(ref_path),False,project,DATASETS)
        if {d:w['recipe'] for d,w in winners(search).items()}!=spec['fixed_recipes']:
            raise ValueError('Frozen recipe winners changed')
        for entry in reference['datasets']:
            if entry['dataset'] in DATASETS and digest(contained(project,entry['raw_path']))!=entry['raw_sha256']:
                raise ValueError('Pinned raw source changed')
    if a.study=='shared-gnn' and not a.static_results:p.error('Paired GNN study requires --static-results')
    names=['static','shared-gnn'] if a.study=='both' else [a.study]
    scopes=[study_scope(spec,name,a.smoke) for name in names]
    plan=dict(protocol_sha256=digest(a.config),studies=scopes,training_runs=sum(s['training_runs'] for s in scopes),
        synthetic=a.smoke,fixed_recipes=spec['fixed_recipes'],test_planned=True,
        deterministic_training=a.deterministic_training,
        sequential=True,automatic_retry=False,stops_after_requested_studies=True)
    if not a.execute:print(json.dumps(plan,indent=2));return plan
    torch.set_num_threads(1);out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan)
    reports={};static_out=a.static_results.resolve() if a.static_results else out/'static'
    try:
        if a.study=='shared-gnn':
            static=json.loads((static_out/'completed.json').read_text())
            expected=study_scope(spec,'static',a.smoke)
            if (static['plan']['protocol_sha256']!=digest(a.config) or static['plan']['synthetic']!=a.smoke
                    or static['plan']['seeds']!=expected['seeds'] or len(static['runs'])!=expected['training_runs']
                    or not static['test_evaluated']):raise ValueError('Static paired reference is incomplete or incompatible')
            reports['static']=static
        for scope in scopes:
            reports[scope['study']]=run_study(scope,spec,reference,project,out/scope['study'],a.device,a.smoke,
                deterministic_training=a.deterministic_training)
        if 'shared-gnn' in reports:
            seeds=reports['shared-gnn']['plan']['seeds'];check_membership(static_out,out/'shared-gnn',seeds)
            paired=compare_reports(reports['static'],reports['shared-gnn']);write(out/'paired_comparison.json',paired)
        completed=dict(plan=plan,studies={k:dict(summary=v['summary']) for k,v in reports.items()},
            test_evaluated=True,all_requested_studies_completed=True)
        write(out/'completed.json',completed);return completed
    except Exception as error:
        write(out/'failed.json',dict(error=str(error),completed_studies=list(reports)));raise


if __name__=='__main__':main()
