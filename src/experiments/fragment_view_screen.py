"""Native GroupGraph view-count/diversity screen; validation only, no retries."""
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
from src.experiments.cross_domain import prepare_cache,verify_cache
from src.experiments.fragment_dataset_search import diagnostic_summary
from src.experiments.plans import ROOT,make_plan
from src.experiments.selection import run_seed
from src.experiments.sugar_reference_search import write


def validate(spec,reference):
    if digest(reference)!=spec['reference_plan_sha256']:raise ValueError('Reference plan changed')
    if (spec['schema']!=1 or spec['datasets']!=['ec50_scaffold','ec50_size'] or
        spec['seeds']!=[10501,10502] or spec['views']!=[3,5] or spec['epochs']!=100):
        raise ValueError('Screen scope changed')
    expected={'off':dict(lambda_diversity=0,diversity_delay_epochs=0,diversity_ramp_epochs=0),
        'constant':dict(lambda_diversity=.005,diversity_delay_epochs=0,diversity_ramp_epochs=0),
        'ramp':dict(lambda_diversity=.005,diversity_delay_epochs=10,diversity_ramp_epochs=20)}
    if spec['diversity']!=expected:raise ValueError('Diversity schedules changed')
    fixed=dict(learning_rate=.0003,edge_ratio=.5,fragment_sample_ratio=.7,lambda_contrast=2,
        lambda_irm=0,fragment_feature_mode='static_pool',inference_policy='independent',
        ranking_mode='learned',selection_estimator='hard')
    if spec['fixed']!=fixed:raise ValueError('Fixed recipe changed')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/fragment_view_diversity_screen_20261018.json'))
    p.add_argument('--reference-plan',required=True);p.add_argument('--view-cache-root',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args(argv);spec=json.loads(Path(a.config).read_text());validate(spec,a.reference_plan)
    reference=json.loads(Path(a.reference_plan).read_text());seeds=spec['seeds'][:1] if a.smoke else spec['seeds']
    plan=dict(spec,seeds=seeds,epochs=5 if a.smoke else 100,training_runs=len(seeds)*2*3*2,
        synthetic=a.smoke,test_evaluated=False,automatic_continuation=False,
        note='Compare complete view-count recipes; loss normalization is unchanged')
    if not a.execute:print(json.dumps(plan,indent=2));return plan
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan)
    rows=[]
    for dataset in spec['datasets']:
        if not a.smoke:verify_cache(reference['configs'][dataset])
        for nviews in spec['views']:
            base=deepcopy(reference['configs'][dataset]);base.update(spec['fixed'])
            base.update(device=a.device,epochs=plan['epochs'],num_views=nviews,
                view_fragmentations={'group_graph':nviews},compare_checkpoint_rules=False,
                report_within_assay=False,log_gradient_influence=False,selection_mode='learned',
                message_gate_mode='learned',full_graph_gating=False,erm_baseline=False,
                test_split=None,report_splits=[],log_test_curve=False)
            if a.smoke:
                from src.data.preprocess import run_preprocessing
                from src.training.pipeline import Config
                torch.set_num_threads(1);_,fixture=make_plan(out/dataset/f'views{nviews}',a.device,True)
                base.update(fixture);base.update(epochs=5,num_views=nviews,view_fragmentations={'group_graph':nviews},
                    loader_split_names=['train','ood_val'],split_names=['train','ood_val'])
                base.update(spec['fixed'])
                base['splits'].pop('ood_test',None);run_preprocessing(Config(base))
            elif nviews==5:verify_cache(base)
            else:
                base['processed_dir']=str(Path(a.view_cache_root).resolve()/dataset/'views3')
                if Path(base['processed_dir']).exists():verify_cache(base)
                else:prepare_cache(base)
            for name,settings in spec['diversity'].items():
                for seed in seeds:
                    values=deepcopy(base);values.update(settings)
                    if a.smoke and name=='ramp':values.update(diversity_delay_epochs=1,diversity_ramp_epochs=2)
                    run=out/dataset/f'views{nviews}_{name}'/f'seed{seed}';run.parent.mkdir(parents=True,exist_ok=True)
                    row=dict(dataset=dataset,views=nviews,diversity=name,seed=seed,status='running')
                    rows.append(row);write(out/'status.json',rows)
                    try:
                        if not a.smoke and shutil.disk_usage(out).free<20*1024**3:raise RuntimeError('Low disk')
                        set_all_seeds(seed);result=run_seed(values,run,seed,False,('independent',))
                        checkpoint=run/'best_independent.pt'
                        replay=evaluate_saved(checkpoint,'ood_val',Path(base['processed_dir']),a.device,
                            'independent',seed,prediction_path=run/'validation_predictions.npz')
                        require_parity(replay['auc'],result['best']['independent']['auc'])
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
        for nviews in spec['views']:
            for name in spec['diversity']:
                subset=[r for r in rows if r['dataset']==dataset and r['views']==nviews and r['diversity']==name]
                scores=[r['validation_auc'] for r in subset]
                candidates.append(dict(views=nviews,diversity=name,mean_validation_auc=statistics.mean(scores),
                    sample_sd=statistics.stdev(scores) if len(scores)>1 else None,
                    diagnostics=[r['diagnostics'] for r in subset]))
        write(out/dataset/'candidates.json',sorted(candidates,key=lambda r:-r['mean_validation_auc']))
    write(out/'completed.json',dict(training_runs=len(rows),synthetic=a.smoke,test_evaluated=False,
        note='Review validation and coherence before second-stage tuning; no automatic selection or continuation'))
    return rows
