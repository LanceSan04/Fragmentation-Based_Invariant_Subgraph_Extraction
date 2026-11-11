"""Bounded Scaffold/Size validation search with separate coherence diagnostics.

No test evaluation, automatic recipe promotion, new architecture or next study.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import statistics

import torch
from src.analysis.selection_faithfulness import main as pilot
from src.experiments.checkpoints import digest,evaluate_saved,require_parity
from src.experiments.common import set_all_seeds
from src.experiments.cross_domain import verify_cache
from src.experiments.plans import ROOT,make_plan
from src.experiments.selection import run_seed
from src.experiments.sugar_reference_search import write


def validate(spec,reference):
    if digest(reference)!=spec['reference_plan_sha256']:raise ValueError('Reference plan changed')
    if (spec['datasets']!=['ec50_scaffold','ec50_size'] or spec['seeds']!=[10501,10502]
        or spec['epochs']!=100 or spec['inference_policy']!='independent' or spec['feature_mode']!='static_pool'):
        raise ValueError('Frozen search scope changed')
    if spec['columns']!=['learning_rate','edge_ratio','fragment_sample_ratio','lambda_contrast']:
        raise ValueError('Unexpected search dimensions')
    if len(spec['recipes'])!=12 or len({tuple(r) for r in spec['recipes']})!=12:raise ValueError('Expected 12 unique recipes')
    expanded=spec.get('schema')==2
    if spec.get('schema') not in [1,2]:raise ValueError('Unknown search schema')
    if expanded and spec.get('fixed')!=dict(num_views=3,view_fragmentations={'group_graph':3},
        lambda_diversity=.005,diversity_delay_epochs=0,diversity_ramp_epochs=0):
        raise ValueError('Screen-selected settings changed')
    for lr,edge,sample,contrast in spec['recipes']:
        if (lr not in [.0001,.0003,.001] or edge not in ([.3,.5,.7,.9] if expanded else [.3,.5,.7])
            or sample not in ([.5,.7,.9] if expanded else [.7,.9]) or contrast not in [0,.1,2]):
            raise ValueError('Recipe outside frozen ranges')


def apply_screen_settings(base,spec,dataset,view_cache_root,smoke=False):
    """Use the verified three-view cache, preserving the legacy search protocol."""
    if spec.get('schema')!=2:return
    if not smoke and not view_cache_root:raise ValueError('Three-view search requires --view-cache-root')
    base.update(deepcopy(spec['fixed']))
    base['lambda_irm']=0.
    if not smoke:
        base['processed_dir']=str(Path(view_cache_root).resolve()/dataset/'views3')


def diagnostic_summary(report):
    rows=report['records'];groups={}
    atom_overlap=[];bond_overlap=[]
    from itertools import combinations
    for index in {r['index'] for r in rows}:
        for first,second in combinations([r for r in rows if r['index']==index],2):
            for key,values in [('atom_ids',atom_overlap),('bond_atom_pairs',bond_overlap)]:
                aa={tuple(x) if isinstance(x,list) else x for x in first['selected'][key]}
                bb={tuple(x) if isinstance(x,list) else x for x in second['selected'][key]}
                if aa|bb:values.append(len(aa&bb)/len(aa|bb))
    for name in ('ring','amide','carbonyl','hydroxyl','nitrile'):
        values=[r['selected']['chemical_groups'].get(name) for r in rows if r['selected']['chemical_groups'] is not None]
        values=[v for v in values if v and v['touched']]
        groups[name]=dict(molecule_views_with_touched_groups=len(values),
            mean_cut_fraction=None if not values else statistics.mean(v['cut']/v['touched'] for v in values))
    return dict(coverage=report['summary'],groups=groups,
        mean_pairwise_atom_jaccard=statistics.mean(atom_overlap) if atom_overlap else None,
        mean_pairwise_bond_jaccard=statistics.mean(bond_overlap) if bond_overlap else None,
        mean_atom_fraction=statistics.mean(r['selected']['atoms']/r['original_atoms'] for r in rows),
        mean_bond_fraction=statistics.mean(r['selected']['bonds']/r['original_bonds'] for r in rows if r['original_bonds']),
        mean_components=statistics.mean(r['selected']['components'] for r in rows),
        caveat='Descriptive own-view diagnostics; whole-fragment preservation is not proof of faithfulness')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/fragment_scaffold_size_search_20261018.json'))
    p.add_argument('--reference-plan',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--view-cache-root',help='Existing screen caches; no preprocessing is performed')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args(argv);spec=json.loads(Path(a.config).read_text());validate(spec,a.reference_plan)
    if spec.get('schema')==2 and not a.smoke and not a.view_cache_root:
        p.error('Three-view search requires --view-cache-root')
    reference=json.loads(Path(a.reference_plan).read_text())
    seeds=spec['seeds'][:1] if a.smoke else spec['seeds'];recipes=spec['recipes'][:2] if a.smoke else spec['recipes']
    plan=dict(spec,seeds=seeds,recipes=recipes,training_runs=len(seeds)*len(recipes)*2,
        epochs=2 if a.smoke else 100,synthetic=a.smoke,test_evaluated=False,
        selection='Report mean validation AUC and separate coherence evidence; no automatic test or combined-score selection')
    if not a.execute:print(json.dumps(plan,indent=2));return plan
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan)
    rows=[]
    for dataset in spec['datasets']:
        base=deepcopy(reference['configs'][dataset]);base.update(device=a.device,epochs=plan['epochs'],
            compare_checkpoint_rules=False,report_within_assay=False,log_gradient_influence=False,
            fragment_feature_mode='static_pool',lambda_diversity=0.,lambda_irm=0.,
            diversity_delay_epochs=0,diversity_ramp_epochs=0,inference_policy='independent',
            ranking_mode='learned',selection_mode='learned',selection_estimator='hard',
            message_gate_mode='learned',full_graph_gating=False,erm_baseline=False,
            test_split=None,report_splits=[],log_test_curve=False)
        if a.smoke:
            from src.training.pipeline import Config
            from src.data.preprocess import run_preprocessing
            torch.set_num_threads(1);_,fixture=make_plan(out/dataset,a.device,True)
            base.update(fixture);base.update(epochs=2,num_views=2,view_fragmentations={'group_graph':2},
                loader_split_names=['train','ood_val'],split_names=['train','ood_val'])
            apply_screen_settings(base,spec,dataset,a.view_cache_root,True)
            base['splits'].pop('ood_test',None);run_preprocessing(Config(base))
        else:
            apply_screen_settings(base,spec,dataset,a.view_cache_root)
            verify_cache(base)
        for i,recipe in enumerate(recipes):
            for seed in seeds:
                values=deepcopy(base);values.update(dict(zip(spec['columns'],recipe)))
                run=out/dataset/f'recipe{i+1:02d}'/f'seed{seed}';run.parent.mkdir(parents=True,exist_ok=True)
                row=dict(dataset=dataset,recipe=i+1,seed=seed,parameters=dict(zip(spec['columns'],recipe)),status='running')
                rows.append(row);write(out/'status.json',rows)
                if not a.smoke and shutil.disk_usage(out).free<20*1024**3:raise RuntimeError('Low disk; stopped')
                try:
                    set_all_seeds(seed);result=run_seed(values,run,seed,False,('independent',))
                    checkpoint=run/'best_independent.pt';selected=result['best']['independent']['auc']
                    replay=evaluate_saved(checkpoint,'ood_val',Path(base['processed_dir']),a.device,
                        'independent',seed,prediction_path=run/'validation_predictions.npz')
                    require_parity(replay['auc'],selected)
                    threads=torch.get_num_threads()
                    try:
                        report=pilot(['--checkpoint',str(checkpoint),'--processed-dir',base['processed_dir'],
                            '--output-dir',str(run/'coherence'),'--device',a.device,
                            '--molecules','3' if a.smoke else '20','--random-draws','12' if a.smoke else '64',
                            '--controls','2' if a.smoke else '8','--max-fragments','2' if a.smoke else '3',
                            '--visual-molecules','2' if a.smoke else '5'])
                    finally:torch.set_num_threads(threads)
                    row.update(status='completed',validation_auc=replay['auc'],checkpoint=str(checkpoint),
                        checkpoint_sha256=digest(checkpoint),diagnostics=diagnostic_summary(report))
                except Exception as error:
                    row.update(status='failed',error=str(error));write(out/'status.json',rows);raise
                write(out/'status.json',rows)
        candidates=[]
        for i in range(len(recipes)):
            matching=[r for r in rows if r['dataset']==dataset and r['recipe']==i+1]
            scores=[r['validation_auc'] for r in matching]
            candidates.append(dict(recipe=i+1,parameters=matching[0]['parameters'],
                mean_validation_auc=statistics.mean(scores),sample_sd=statistics.stdev(scores) if len(scores)>1 else None,
                diagnostics=[r['diagnostics'] for r in matching]))
        write(out/dataset/'candidates.json',sorted(candidates,key=lambda r:-r['mean_validation_auc']))
    write(out/'completed.json',dict(training_runs=len(rows),synthetic=a.smoke,test_evaluated=False,
        note='Review validation/coherence tradeoffs before freezing recipes; no automatic continuation'))
    return rows
