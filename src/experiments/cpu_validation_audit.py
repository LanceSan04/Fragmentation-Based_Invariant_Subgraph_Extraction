"""Two deterministic CPU validation traces for each of 30 frozen checkpoints."""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from src.data.dataloader import build_split_dataloaders
from src.experiments.checkpoints import digest
from src.experiments.common import set_all_seeds
from src.experiments.fragment_tuning import contained
from src.experiments.plans import ROOT
from src.experiments.reload_diagnostic import state_hash,compare
from src.experiments.sugar_reference_search import write
from src.training.checkpoint import load_checkpoint,load_view_state_dicts
from src.training.pipeline import Config,build_views_from_feature_dims,strip_non_tensor_batch_fields,_evaluate_ensemble


def validate(spec):
    expected={(d,a,s) for d in ('ec50_assay','ec50_scaffold','ec50_size')
              for a in ('scorer_only','scorer_and_classifier') for s in range(11211,11216)}
    if (spec['schema']!=1 or spec['passes_per_checkpoint']!=2 or spec['batch_size']!=64
            or spec['torch_threads']!=1 or len(spec['rows'])!=30
            or {(r['dataset'],r['arm'],r['seed']) for r in spec['rows']}!=expected):
        raise ValueError('CPU validation audit scope changed')


def trace(payload,cfg,loader,seed):
    set_all_seeds(seed);views=build_views_from_feature_dims(cfg,payload['feature_dims'])
    load_view_state_dicts(views,payload)
    for v in views:v.cpu().eval()
    before=state_hash(views);ys=[];ps=[];edge_rows=[];cursor=0
    masks=[[] for _ in views];scores=[[] for _ in views];full=[]
    class Collector:
        def eval(self):
            for v in views:v.eval()
        def __call__(self,batch,**kwargs):
            nonlocal cursor
            outputs=[]
            for i,v in enumerate(views):
                pred,mid=v(batch,return_intermediates=True);outputs.append(pred)
                masks[i].append(mid['selected_fragment_mask'].numpy())
                scores[i].append(mid['fragment_scores'].numpy())
                full.append(bool(mid['fragment_edge_eligible'].all()))
                if i==0:edge_rows.append(mid['fragment_edge_batch'].numpy()+cursor)
            pred=torch.logsumexp(torch.stack(outputs,0),0)-math.log(len(views))
            ys.append(batch.y.numpy().reshape(-1));ps.append(pred.exp()[:,1].numpy())
            cursor+=int(batch.y.numel());return pred
    with torch.no_grad():loss,acc,auc=_evaluate_ensemble(Collector(),loader,'cpu',True)
    if before!=state_hash(views) or not all(full):
        raise ValueError('Evaluation changed model state or used sampled edges')
    arrays=dict(labels=np.concatenate(ys),probabilities=np.concatenate(ps),edge_row=np.concatenate(edge_rows))
    counts=np.bincount(arrays['edge_row'],minlength=len(arrays['labels']))
    arrays['edge_column']=np.arange(arrays['edge_row'].size)-np.repeat(np.r_[0,np.cumsum(counts)[:-1]],counts)
    for i in range(len(views)):
        arrays['mask'+str(i)]=np.concatenate(masks[i]);arrays['scores'+str(i)]=np.concatenate(scores[i])
    return dict(loss=loss,acc=acc,auc=float(auc),model_state_sha256=before),arrays


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/cpu_validation_audit_20261030.json'))
    p.add_argument('--project-root',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--device',choices=['cpu'],default='cpu')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv);spec=json.loads(Path(a.config).read_text());validate(spec)
    plan=dict(protocol=spec,validation_passes=60,training_runs=0,test_evaluated=False,
        tolerance_modified=False,automatic_continuation=False,device='cpu',config_sha256=digest(a.config))
    if not a.execute:print(json.dumps(plan,indent=2));return plan
    project=a.project_root.resolve();checked={}
    # Integrity preflight for every frozen input before the first validation forward.
    for row in spec['rows']:
        for name,sha in [(row['checkpoint'],row['checkpoint_sha256']),
                         (row['cache']+'/ood_val.pt',row['validation_cache_sha256']),
                         (row['reference_predictions'],row['reference_predictions_sha256'])]:
            if name not in checked:checked[name]=digest(contained(project,name))
            if checked[name]!=sha:raise ValueError('Frozen input changed: '+name)
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    loaders={};results=[]
    try:
        for row in spec['rows']:
            started=time.perf_counter();checkpoint=contained(project,row['checkpoint'])
            payload=load_checkpoint(checkpoint,map_location='cpu')
            if payload['epoch']!=row['checkpoint_epoch']:raise ValueError('Selected epoch changed')
            cfg=Config(dict(payload['config'],processed_dir=str(contained(project,row['cache'])),
                device='cpu',loader_split_names=['ood_val'],num_workers=0,pin_memory=False,
                batch_size=64,test_split=None,report_splits=[]))
            if row['dataset'] not in loaders:
                loader=build_split_dataloaders(cfg.processed_dir,cfg)['ood_val']
                strip_non_tensor_batch_fields({'ood_val':loader});loaders[row['dataset']]=loader
            with np.load(contained(project,row['reference_predictions'])) as ref:
                y_ref,p_ref=ref['labels'].copy(),ref['probabilities'].copy()
            run=out/row['dataset']/row['arm']/('seed'+str(row['seed']));run.mkdir(parents=True)
            metrics=[];arrays=[]
            for repeat in range(2):
                m,t=trace(payload,cfg,loaders[row['dataset']],row['seed'])
                if not np.array_equal(t['labels'],y_ref):raise ValueError('Validation membership/order changed')
                np.savez_compressed(run/('repeat'+str(repeat)+'.npz'),**t)
                metrics.append(m);arrays.append(t)
            comparison=compare(arrays[0],arrays[1],len(payload['view_state_dicts']))
            exact=all(np.array_equal(arrays[0][k],arrays[1][k]) for k in arrays[0])
            result=dict(dataset=row['dataset'],arm=row['arm'],seed=row['seed'],
                checkpoint=str(checkpoint),checkpoint_sha256=row['checkpoint_sha256'],
                selected_epoch=row['checkpoint_epoch'],passes=metrics,bit_identical=exact,
                model_states_identical=metrics[0]['model_state_sha256']==metrics[1]['model_state_sha256'],
                comparison=comparison,difference_from_epoch_auc=metrics[0]['auc']-row['epoch_auc'],
                max_probability_difference_from_saved_gpu=float(abs(arrays[0]['probabilities']-p_ref).max()),
                seconds=time.perf_counter()-started)
            if digest(checkpoint)!=row['checkpoint_sha256']:raise ValueError('Checkpoint file changed')
            write(run/'diagnostic.json',result);results.append(result);write(out/'progress.json',results)
            print(json.dumps({k:result[k] for k in ['dataset','arm','seed','bit_identical','difference_from_epoch_auc','seconds']}),flush=True)
        report=dict(plan=plan,runs=results,all_repeats_bit_identical=all(r['bit_identical'] for r in results),
            all_states_identical=all(r['model_states_identical'] for r in results),
            max_absolute_epoch_auc_difference=max(abs(r['difference_from_epoch_auc']) for r in results),
            checkpoints_exceeding_original_gpu_auc_gate=sum(abs(r['difference_from_epoch_auc'])>1e-4 for r in results),
            test_evaluated=False,tolerance_modified=False,
            note='Diagnostic only: original GPU checkpoint-selection records are preserved; no gate waiver or test acceptance is implied.')
        write(out/'completed.json',report);return report
    except Exception as error:
        write(out/'failed.json',dict(error=str(error),completed_checkpoints=results,test_evaluated=False));raise


if __name__=='__main__':main()
