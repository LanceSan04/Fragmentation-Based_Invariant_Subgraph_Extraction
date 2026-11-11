"""Three frozen validation-only reloads with prediction and edge-selection traces."""
import argparse
import hashlib
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
from src.experiments.sugar_reference_search import write
from src.training.checkpoint import load_checkpoint,load_view_state_dicts
from src.training.pipeline import Config,build_views_from_feature_dims,strip_non_tensor_batch_fields,_evaluate_ensemble


def state_hash(views):
    h=hashlib.sha256()
    for view in views:
        for name,value in view.state_dict().items():
            h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def compare(a,b,num_views):
    if not np.array_equal(a['labels'],b['labels']) or not np.array_equal(a['edge_row'],b['edge_row']):
        raise ValueError('Prediction/edge membership differs')
    result=dict(max_probability_difference=float(abs(a['probabilities']-b['probabilities']).max()),
        changed_probabilities=int((a['probabilities']!=b['probabilities']).sum()),
        class_decisions_changed=int(((a['probabilities']>=.5)!=(b['probabilities']>=.5)).sum()),views=[])
    for i in range(num_views):
        changed=a['mask'+str(i)]!=b['mask'+str(i)]
        result['views'].append(dict(view=i,changed_selection_bits=int(changed.sum()),
            changed_molecule_rows=np.unique(a['edge_row'][changed]).astype(int).tolist(),
            max_score_difference=float(abs(a['scores'+str(i)]-b['scores'+str(i)]).max())))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/reload_gpu_diagnostic_20261030.json'))
    p.add_argument('--project-root',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv);spec=json.loads(Path(a.config).read_text())
    if (spec['schema']!=1 or spec['repeats']!=3 or spec['seed']!=11215
            or spec['batch_size']!=64 or spec['expected_selected_epoch']!=141):
        raise ValueError('Diagnostic scope changed')
    plan=dict(protocol=spec,device=a.device,validation_passes=3,training_runs=0,
        test_evaluated=False,tolerance_modified=False,config_sha256=digest(a.config))
    if not a.execute:print(json.dumps(plan,indent=2));return plan
    project=a.project_root.resolve();checkpoint=contained(project,spec['checkpoint'])
    cache=contained(project,spec['cache']);reference=contained(project,spec['reference_predictions'])
    for path,sha in [(checkpoint,spec['checkpoint_sha256']),
                     (cache/'ood_val.pt',spec['validation_cache_sha256']),
                     (reference,spec['reference_predictions_sha256'])]:
        if digest(path)!=sha:raise ValueError('Pinned diagnostic input changed: '+str(path))
    payload=load_checkpoint(checkpoint,map_location='cpu')
    if payload['epoch']!=141:raise ValueError('Wrong selected checkpoint')
    cfg=Config(dict(payload['config'],processed_dir=str(cache),device=a.device,
        batch_size=64,loader_split_names=['ood_val'],num_workers=0,pin_memory=False,
        test_split=None,report_splits=[]))
    torch.set_num_threads(spec['torch_threads'])
    loaders=build_split_dataloaders(cache,cfg)
    with np.load(reference) as saved:
        y_ref,p_ref=saved['labels'].copy(),saved['probabilities'].copy()
    labels=np.asarray([int(d.y.reshape(-1)[0]) for d in loaders['ood_val'].dataset])
    if not np.array_equal(labels,y_ref):raise ValueError('Validation labels/order changed')
    strip_non_tensor_batch_fields(loaders)
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan)
    results=[]
    try:
        for repeat in range(3):
            started=time.perf_counter();set_all_seeds(spec['seed'])
            views=build_views_from_feature_dims(cfg,payload['feature_dims']);load_view_state_dicts(views,payload)
            for view in views:view.to(a.device).eval()
            before=state_hash(views);ys=[];ps=[];edge_rows=[]
            masks=[[] for _ in views];scores=[[] for _ in views];eligibility=[];cursor=0
            class Collector:
                def eval(self):
                    for view in views:view.eval()
                def __call__(self,batch,**kwargs):
                    nonlocal cursor
                    outputs=[]
                    for i,view in enumerate(views):
                        pred,mid=view(batch,return_intermediates=True);outputs.append(pred)
                        masks[i].append(mid['selected_fragment_mask'].cpu().numpy())
                        scores[i].append(mid['fragment_scores'].cpu().numpy())
                        eligibility.append(bool(mid['fragment_edge_eligible'].all()))
                        if i==0:edge_rows.append(mid['fragment_edge_batch'].cpu().numpy()+cursor)
                    pred=torch.logsumexp(torch.stack(outputs,0),0)-math.log(len(views))
                    ys.append(batch.y.cpu().numpy().reshape(-1));ps.append(pred.exp()[:,1].cpu().numpy())
                    cursor+=int(batch.y.numel());return pred
            with torch.no_grad():loss,acc,auc=_evaluate_ensemble(Collector(),loaders['ood_val'],a.device,True)
            if before!=state_hash(views):raise ValueError('Evaluation mutated parameters/buffers')
            trace=dict(labels=np.concatenate(ys),probabilities=np.concatenate(ps),edge_row=np.concatenate(edge_rows))
            counts=np.bincount(trace['edge_row'],minlength=len(trace['labels']))
            offsets=np.r_[0,np.cumsum(counts)[:-1]]
            trace['edge_column']=np.arange(trace['edge_row'].size)-np.repeat(offsets,counts)
            if not np.array_equal(trace['labels'],y_ref) or not all(eligibility):
                raise ValueError('Validation ordering or evaluation sampler changed')
            for i in range(len(views)):
                trace['mask'+str(i)]=np.concatenate(masks[i]);trace['scores'+str(i)]=np.concatenate(scores[i])
            np.savez_compressed(out/('repeat'+str(repeat)+'.npz'),**trace)
            row=dict(repeat=repeat,auc=float(auc),loss=loss,acc=acc,seconds=time.perf_counter()-started,
                difference_from_epoch_auc=float(auc-spec['expected_epoch_auc']),
                max_probability_difference_from_saved_gpu=float(abs(trace['probabilities']-p_ref).max()),
                model_state_unchanged=True)
            results.append(row);write(out/'progress.json',results);print(json.dumps(row),flush=True)
        pairs=[]
        for i in range(3):
            for j in range(i+1,3):
                with np.load(out/('repeat'+str(i)+'.npz')) as x,np.load(out/('repeat'+str(j)+'.npz')) as y:
                    pairs.append(dict(repeats=[i,j],**compare(x,y,len(views))))
        if digest(checkpoint)!=spec['checkpoint_sha256']:raise ValueError('Checkpoint file changed')
        report=dict(plan=plan,runs=results,pairwise=pairs,checkpoint_unchanged=True,
            test_evaluated=False,tolerance_modified=False,automatic_continuation=False,
            cuda_version=torch.version.cuda,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            limitation='Instrumented inference copies traces to CPU between batches. Repeatability here does not prove all earlier execution contexts were identical.')
        write(out/'completed.json',report);return report
    except Exception as error:
        write(out/'failed.json',dict(error=str(error),completed_repeats=results,test_evaluated=False));raise


if __name__=='__main__':main()
