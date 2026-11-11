"""Bounded static-feature tuning on pinned official EC50 and IC50 files."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.dataloader import build_split_dataloaders
from src.experiments.checkpoints import digest,evaluate_saved,restore
from src.experiments.context_test import require_exact_predictions
from src.experiments.cross_domain import prepare_cache,verify_cache
from src.experiments.fragment_tuning import contained
from src.experiments.plans import ROOT,make_plan
from src.experiments.selection import run_seed,validation_execution
from src.experiments.sugar_reference_search import write
from src.training.pipeline import Config,strip_non_tensor_batch_fields

DATASETS=[f'{family}_{split}' for family in ('ec50','ic50') for split in ('assay','scaffold','size')]
RECIPES=[[.0003,.5,.7,.1],[.0003,.9,.5,2],[.001,.7,.9,.1],
         [.0001,.3,.5,0],[.001,.5,.9,1],[.0003,.7,.7,4]]


def validate(spec):
    fixed=dict(fragment_feature_mode='static_pool',classifier_feature_mode='raw',
        validation_device='cpu',save_selected_validation_predictions=True,
        num_views=3,view_fragmentations={'group_graph':3},lambda_diversity=.005,
        lambda_irm=0.,selection_estimator='hard',ranking_mode='learned',
        message_gate_mode='learned',scorer_context='sampled',inference_policy='independent',
        erm_baseline=False,full_graph_gating=False,selection_mode='learned',selection_budget='edges',
        hidden_dim=128,gnn_layers=3,batch_size=64,dropout=.5,aggregation_mode='mean',
        contrast_scope='pooled',contrast_temperature=.1)
    if (spec['schema']!=1 or [r['dataset'] for r in spec['datasets']]!=DATASETS
            or spec['recipes']!=RECIPES or spec['columns']!=['learning_rate','edge_ratio','fragment_sample_ratio','lambda_contrast']
            or (spec['pilot_epochs'],spec['pilot_seed'])!=(5,11301)
            or spec['search_epochs']!=150 or spec['search_seeds']!=[11311,11312]
            or spec['confirmation_epochs']!=150 or spec['confirmation_seeds']!=[11321,11322,11323,11324,11325]
            or spec['pilot_recipe']!=1 or spec['budget']!=dict(pilot_runs=6,search_runs=72,confirmation_runs=30)
            or any(spec['base'].get(k)!=v for k,v in fixed.items())
            or any(not r['raw_path'].startswith('data/raw/drugood_official_chembl30_20261028/')
                   or Path(r['raw_path']).name!='lbap_core_'+r['dataset']+'.json' for r in spec['datasets'])):
        raise ValueError('Frozen original-model tuning protocol changed')


def dataset_scope(family):
    return [d for d in DATASETS if family=='all' or d.startswith(family+'_')]


def require_complete(report,stage,config_sha,synthetic,project,datasets=None):
    plan=report['plan']
    if plan['stage']!=stage or plan['config_sha256']!=config_sha or plan['synthetic']!=synthetic:
        raise ValueError('Prior-stage evidence does not match this protocol')
    expected_ids=[1] if stage=='prepare-pilot' else [1,2] if synthetic else list(range(1,7))
    expected_seeds=[11301] if stage=='prepare-pilot' else [11311] if synthetic else [11311,11312]
    expected_epochs=2 if synthetic else 5 if stage=='prepare-pilot' else 150
    if plan['recipe_ids']!=expected_ids or plan['seeds']!=expected_seeds or plan['epochs']!=expected_epochs:
        raise ValueError('Prior-stage budget/recipe set changed')
    datasets=DATASETS if datasets is None else datasets
    if plan.get('datasets',DATASETS)!=datasets:
        raise ValueError('Prior-stage dataset scope differs from requested scope')
    expected={(d,r,s) for d in datasets for r in plan['recipe_ids'] for s in plan['seeds']}
    rows=report['runs']
    if (len(rows)!=len(expected) or {(r['dataset'],r['recipe'],r['seed']) for r in rows}!=expected
            or any(r['status']!='completed' or not math.isfinite(r['validation']['auc']) for r in rows)):
        raise ValueError('Incomplete prior-stage evidence')
    for row in rows:
        for field in ('checkpoint','reference'):Path(row[field]).resolve().relative_to(project.resolve())
        if digest(row['checkpoint'])!=row['checkpoint_sha256'] or digest(row['reference'])!=row['reference_sha256']:
            raise ValueError('Prior-stage checkpoint or validation reference changed')
        with np.load(row['reference']) as ref:
            if roc_auc_score(ref['labels'],ref['probabilities'])!=row['validation']['auc']:
                raise ValueError('Prior-stage validation objective differs from its saved predictions')


def winners(report):
    selected={}
    for dataset in report.get('plan',{}).get('datasets',DATASETS):
        rows=[r for r in report['runs'] if r['dataset']==dataset]
        ids=sorted({r['recipe'] for r in rows})
        means={rid:statistics.mean(r['validation']['auc'] for r in rows if r['recipe']==rid) for rid in ids}
        selected[dataset]=dict(recipe=max(ids,key=lambda rid:means[rid]),validation_means=means)
    return selected


def retention(checkpoint,cache,out):
    cfg,views=restore(checkpoint,Path(cache),'cpu')
    values=dict(cfg.to_dict(),loader_split_names=['ood_val'],num_workers=0,pin_memory=False)
    loaders=build_split_dataloaders(cache,Config(values));strip_non_tensor_batch_fields(loaders)
    batch=next(iter(loaders['ood_val']));n=int(batch.num_graphs)
    atom_total=torch.bincount(batch.batch,minlength=n)
    bond_total=torch.bincount(batch.batch[batch.bond_index[0]],minlength=n)
    summaries=[]
    with validation_execution('cpu'),torch.no_grad():
        for i,view in enumerate(views):
            _,mid=view(batch,return_intermediates=True)
            atoms=torch.bincount(batch.batch[mid['selected_atom_ids']],minlength=n)
            bonds=torch.bincount(batch.batch[batch.bond_index[0,mid['selected_bond_ids']]],minlength=n)
            valid=bond_total>0
            summaries.append(dict(view=i,mean_atom_fraction=float((atoms/atom_total).mean()),
                mean_bond_fraction=float((bonds[valid]/bond_total[valid]).mean()) if valid.any() else None))
    report=dict(molecules=n,rows=list(range(n)),views=summaries,
        limitation='Fixed first validation batch, not a whole-split retention estimate; never selects recipes.')
    write(out,report);return report


def ki_transfer_inputs(project, reference_sha, smoke):
    """Map the frozen EC50 winners to the corresponding downloaded Ki splits."""
    frozen = json.loads((ROOT / "configs/ec50_replication_20261031.json").read_text())
    if frozen["reference_config_sha256"] != reference_sha:
        raise ValueError("Frozen EC50 recipes belong to another protocol")
    if not smoke:
        search = contained(project, frozen["search_results"]) / "completed.json"
        if digest(search) != frozen["search_completed_sha256"]:
            raise ValueError("Frozen EC50 validation search changed")
    hashes = {
        "assay": "42923fa7e92530ab949eaad1ddb26e47857dd06006cc30473ccda6d60c890059",
        "scaffold": "dcc4e300bec0ad1ef6b6d50b997d07b2a40c6971b4e1e12099e8abac751bfad9",
        "size": "446779027ee8814aba7ebded369d565bc9e85a057e657ec2c1be1bae86f7ffa9",
    }
    version = "drugood_official_chembl30_20261028"
    inputs = [dict(dataset=f"ki_{split}", raw_sha256=sha,
                   raw_path=f"data/raw/{version}/lbap_core_ki_{split}.json",
                   cache_path=f"data/processed/{version}/ki_{split}/views3")
              for split, sha in hashes.items()]
    selected = {f"ki_{split}": dict(recipe=frozen["fixed_recipes"][f"ec50_{split}"],
                                    source_dataset=f"ec50_{split}") for split in hashes}
    return inputs, selected


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/official_original_tuning_20261030.json'))
    p.add_argument('--project-root',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--stage',choices=['prepare-pilot','search','confirm','ki-transfer'],default='prepare-pilot')
    p.add_argument('--family',choices=['all','ec50','ic50'],default='all',
                   help='Restrict search/confirmation to one endpoint family; pilots cover all six datasets')
    p.add_argument('--pilot-results',type=Path)
    p.add_argument('--search-results',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--evaluate-test',action='store_true')
    p.add_argument('--deterministic-training',action='store_true',
                   help='Enable repeatable algorithms where available and record remaining warnings')
    a=p.parse_args(argv);spec=json.loads(Path(a.config).read_text());validate(spec)
    if a.stage=='prepare-pilot' and a.family!='all':p.error('Resource pilots must cover all six datasets')
    datasets=dataset_scope(a.family)
    if a.evaluate_test and a.stage not in ('confirm','ki-transfer'):
        p.error('Only confirmation or fixed Ki transfer can evaluate test')
    sha=digest(a.config);selected=None
    if a.stage=='search':
        if not a.pilot_results:p.error('Search requires --pilot-results')
        prior=json.loads((a.pilot_results/'completed.json').read_text());require_complete(prior,'prepare-pilot',sha,a.smoke,a.project_root)
    if a.stage=='confirm':
        if not a.search_results:p.error('Confirmation requires --search-results')
        prior=json.loads((a.search_results/'completed.json').read_text());require_complete(prior,'search',sha,a.smoke,a.project_root,datasets)
        selected=winners(prior)
    epochs=spec['pilot_epochs'] if a.stage=='prepare-pilot' else spec['search_epochs'] if a.stage=='search' else spec['confirmation_epochs']
    seeds=[spec['pilot_seed']] if a.stage=='prepare-pilot' else spec['search_seeds'] if a.stage=='search' else spec['confirmation_seeds']
    ids=[spec['pilot_recipe']] if a.stage=='prepare-pilot' else list(range(1,7)) if a.stage=='search' else None
    inputs=spec['datasets']
    if a.stage=='ki-transfer':
        if a.family!='all' or a.pilot_results or a.search_results:
            p.error('Ki transfer has fixed datasets and recipes')
        inputs,selected=ki_transfer_inputs(a.project_root.resolve(),sha,a.smoke)
        datasets=[entry['dataset'] for entry in inputs]
        seeds=[10801]
    if a.smoke:epochs=2;seeds=seeds[:1];ids=ids[:2] if ids else None
    plan=dict(stage=a.stage,config_sha256=sha,protocol=spec,synthetic=a.smoke,
        epochs=epochs,seeds=seeds,recipe_ids=ids or sorted({x['recipe'] for x in selected.values()}),
        selected=selected,datasets=datasets,training_runs=len(datasets)*len(seeds)*(len(ids) if ids else 1),
        training_device=a.device,validation_device='cpu',test_planned=a.evaluate_test,
        deterministic_training=a.deterministic_training,inputs=inputs,
        automatic_continuation=False,no_tuning_on_test=True)
    if not a.execute:print(json.dumps(plan,indent=2));return plan
    project=a.project_root.resolve();out=a.output_dir.resolve()
    if not a.smoke:
        for row in inputs:
            if row['dataset'] not in datasets:continue
            if digest(contained(project,row['raw_path']))!=row['raw_sha256']:raise ValueError('Raw source changed')
    out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan)
    torch.set_num_threads(1);rows=[];cache_map={};reference_map={}
    try:
        for entry in inputs:
            if entry['dataset'] not in datasets:continue
            dataset=entry['dataset'];base=deepcopy(spec['base'])
            if a.deterministic_training:
                base['deterministic_training']=True
            base.update(dataset='drugood_lbap_core_'+dataset,device=a.device,epochs=epochs,
                raw_path=str(contained(project,entry['raw_path'])),processed_dir=str(contained(project,entry['cache_path'])),
                loader_split_names=['train','ood_val'],test_split=None,report_splits=[],log_test_curve=False,
                early_stopping=False,selection_ma_window=1,selection_weight_avg=False,compare_checkpoint_rules=False)
            if a.smoke:
                from src.data.preprocess import run_preprocessing
                _,fixture=make_plan(out/dataset/'fixture',a.device,True);base.update(fixture)
                base.update(epochs=2,hidden_dim=16,gnn_layers=1,num_views=2,view_fragmentations={'group_graph':2},
                    num_workers=0,report_within_assay=True,fragment_feature_mode='static_pool',classifier_feature_mode='raw',
                    validation_device='cpu',save_selected_validation_predictions=True,
                    split_names=['train','ood_val','ood_test'],loader_split_names=['train','ood_val'],test_split=None)
                run_preprocessing(Config(base))
            elif a.stage in ('prepare-pilot','ki-transfer'):
                cache=Path(base['processed_dir']);manifest=verify_cache(base) if cache.exists() else prepare_cache(base)
                write(out/(dataset+'_cache.json'),manifest)
            else:verify_cache(base)
            cache_map[dataset]=base['processed_dir']
            for rid in ids if ids else [selected[dataset]['recipe']]:
                for seed in seeds:
                    values=deepcopy(base);values.update(dict(zip(spec['columns'],spec['recipes'][rid-1])))
                    run=out/dataset/f'recipe{rid:02d}'/f'seed{seed}';run.parent.mkdir(parents=True,exist_ok=True)
                    row=dict(dataset=dataset,recipe=rid,seed=seed,arm='original_static',status='running')
                    rows.append(row);write(out/'status.json',rows)
                    result=run_seed(values,run,seed,False,('independent',))
                    checkpoint=run/'best_independent.pt';reference=run/'best_independent_validation_predictions.npz'
                    metadata=json.loads((run/'best_independent_validation_reference.json').read_text())
                    if metadata['checkpoint_sha256']!=digest(checkpoint) or metadata['prediction_sha256']!=digest(reference):
                        raise ValueError('Selected checkpoint/reference pair changed')
                    if metadata['epoch']!=result['best']['independent']['epoch'] or metadata['auc']!=result['best']['independent']['auc']:
                        raise ValueError('Selected epoch/CPU validation record changed')
                    with validation_execution('cpu'):
                        replay=evaluate_saved(checkpoint,'ood_val',Path(base['processed_dir']),'cpu','independent',seed,
                            prediction_path=run/'validation_replay.npz')
                    require_exact_predictions(run/'validation_replay.npz',reference)
                    row.update(status='completed',validation=replay,selected_epoch=result['best']['independent']['epoch'],
                        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),reference=str(reference),
                        reference_sha256=digest(reference),parameters=dict(zip(spec['columns'],spec['recipes'][rid-1])),
                        resources=result['resources'],retention=retention(checkpoint,base['processed_dir'],run/'retention.json'))
                    reference_map[(dataset,'original_static',seed)]=reference
                    write(out/'status.json',rows)
        report=dict(plan=plan,runs=rows,test_evaluated=False)
        if a.stage=='search':report['winners']=winners(report)
        if a.evaluate_test:
            # A single original-model arm: use a dedicated all-checkpoints validation gate.
            report['test_evaluation']=evaluate_single_arm(rows,cache_map,reference_map,out/'test_evaluation')
            report['test_evaluated']=True
        write(out/'completed.json',report);return report
    except Exception as error:
        if rows and rows[-1]['status']=='running':rows[-1].update(status='failed',error=str(error))
        write(out/'status.json',rows);write(out/'failed.json',dict(error=str(error),runs=rows));raise


def evaluate_single_arm(rows,caches,references,out):
    """All validation references must match before any confirmation test evaluation."""
    out.mkdir(parents=True,exist_ok=False)
    for row in rows:
        if digest(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    for row in rows:
        run=out/row['dataset']/f"seed{row['seed']}";run.mkdir(parents=True)
        with validation_execution('cpu'):
            evaluate_saved(row['checkpoint'],'ood_val',Path(caches[row['dataset']]),'cpu','independent',row['seed'],
                prediction_path=run/'validation_predictions.npz')
        require_exact_predictions(run/'validation_predictions.npz',references[(row['dataset'],row.get('arm','original_static'),row['seed'])])
    for row in rows:
        if digest(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed; no test allowed')
    results=[]
    for row in rows:
        if digest(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed before tests')
        run=out/row['dataset']/f"seed{row['seed']}"
        with validation_execution('cpu'):
            metrics=evaluate_saved(row['checkpoint'],'ood_test',Path(caches[row['dataset']]),'cpu','independent',row['seed'],
                prediction_path=run/'test_predictions.npz')
        results.append(dict(dataset=row['dataset'],seed=row['seed'],recipe=row['recipe'],metrics=metrics))
        write(out/'progress.json',results)
    summary={dataset:dict(mean=statistics.mean(r['metrics']['auc'] for r in results if r['dataset']==dataset),
        sample_sd=statistics.stdev(r['metrics']['auc'] for r in results if r['dataset']==dataset) if
            sum(r['dataset']==dataset for r in results)>1 else None) for dataset in sorted({r['dataset'] for r in results})}
    report=dict(runs=results,summary=summary,no_test_based_selection=True);write(out/'completed.json',report);return report


if __name__=='__main__':main()
