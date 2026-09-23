"""Fixed-trajectory dual-policy checkpoint selection; Assay first, no early stopping."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.experiments.plans import base_config
from src.experiments.plans import make_plan
from src.experiments.common import set_all_seeds
from train_sugar import Config,build_views_from_feature_dims,build_trainer,infer_feature_dimensions,strip_non_tensor_batch_fields,_evaluate_ensemble
from src.data.dataloader import build_split_dataloaders
from src.data.preprocess import run_preprocessing
from src.models.sugar_ensemble import SuGAREnsemble
from src.training.checkpoint import save_training_checkpoint,load_checkpoint,load_view_state_dicts

POLICIES=('shared','independent')

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))

@contextmanager
def preserve_rng():
    py=random.getstate();np_state=np.random.get_state()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        random.setstate(py);np.random.set_state(np_state)

def run_seed(base,out,seed,evaluate_test=False):
    out.mkdir(parents=True,exist_ok=False)
    values=dict(base,seed=seed,sampler_seed=seed,early_stopping=False,selection_ma_window=1,
                selection_weight_avg=False,greedy_model_selection=False,loss_log_path=None,
                test_split=None,report_splits=[],loader_split_names=['train','ood_val'])
    policies=('shared',) if values.get('erm_baseline',False) else POLICIES
    if values.get('erm_baseline',False):
        values.update(lambda_contrast=0.,lambda_diversity=0.,lambda_irm=0.)
    cfg=Config(values);write(out/'config.json',values)
    set_all_seeds(seed)
    loaders=build_split_dataloaders(cfg.processed_dir,cfg);strip_non_tensor_batch_fields(loaders)
    dims=infer_feature_dimensions(loaders['train'])
    views=build_views_from_feature_dims(cfg,dims);trainer=build_trainer(cfg,views,cfg.device)
    ensembles={p:SuGAREnsemble(views,aggregation_mode=cfg.aggregation_mode,inference_policy=p) for p in policies}
    history=[];best={p:dict(auc=-1.,epoch=None) for p in policies}
    for epoch in range(1,cfg.epochs+1):
        loss,acc,components=trainer.train_epoch(loaders['train'],return_loss_components=True)
        row=dict(epoch=epoch,train_loss=loss,train_accuracy=acc,train_components=components,validation={})
        with preserve_rng():
            for policy in policies:
                vl,va,auc=_evaluate_ensemble(ensembles[policy],loaders['ood_val'],cfg.device,True)
                if auc is None or not math.isfinite(auc):raise ValueError('Finite validation AUC required')
                row['validation'][policy]=dict(loss=vl,acc=va,auc=auc)
                if auc>best[policy]['auc']:
                    best[policy]=dict(auc=auc,epoch=epoch)
                    saved=Config(dict(values,inference_policy=policy))
                    save_training_checkpoint(out/f'best_{policy}.pt',trainer,saved,dims,
                        epoch=epoch,history={'selected_checkpoint_epoch':epoch},include_optimizers=False)
        history.append(row);write(out/'history.json',history);write(out/'best.json',best)
        print(f'Seed {seed} epoch {epoch}/{cfg.epochs}: '+str(row['validation']),flush=True)
    # Test data is first loaded only after all checkpoint-selection decisions for this seed.
    if evaluate_test:
        test_cfg=Config(dict(values,loader_split_names=['ood_test']))
        test_loaders=build_split_dataloaders(cfg.processed_dir,test_cfg)
        strip_non_tensor_batch_fields(test_loaders);loaders.update(test_loaders)
    results={}
    for selected_by in policies:
        load_view_state_dicts(views,load_checkpoint(out/f'best_{selected_by}.pt'))
        results[selected_by]={}
        for policy in policies:
            results[selected_by][policy]={}
            for split in ['ood_val']+(['ood_test'] if evaluate_test else []):
                loss,acc,auc=_evaluate_ensemble(ensembles[policy],loaders[split],cfg.device,True)
                if auc is None or not math.isfinite(auc):raise ValueError('Finite final AUC required')
                results[selected_by][policy][split]=dict(loss=loss,acc=acc,auc=auc)
    result=dict(seed=seed,best=best,results=results,
        layout='results[checkpoint selected by policy][evaluation policy][split]',
        note='Fixed training budget; no early stopping. Test never selects checkpoints. Ties keep earliest epoch.')
    write(out/'results.json',result)
    return result

def main(argv=None):
    from src.experiments.plans import workflow_defaults
    defaults=workflow_defaults()["selection"]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['fragment','erm'],default='fragment')
    p.add_argument('--dataset',choices=['assay','size'],default='assay')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--seeds',type=int,nargs='+',default=defaults["seeds"])
    p.add_argument('--epochs',type=int,default=defaults["epochs"])
    p.add_argument('--lambda-contrast',type=float,default=.1,help='Fragment contrastive coefficient; ERM always uses zero')
    p.add_argument('--processed-dir')
    p.add_argument('--output-dir')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--evaluate-test',action='store_true',help='Final evaluation only, after selection; both checkpoints x both policies')
    a=p.parse_args(argv);os.chdir(ROOT)
    if not math.isfinite(a.lambda_contrast) or a.lambda_contrast<0:p.error('lambda-contrast must be finite and nonnegative')
    if a.epochs<1 or not a.seeds or len(set(a.seeds))!=len(a.seeds):p.error('Positive epochs and unique seeds required')
    out=Path(a.output_dir or f'outputs/{a.dataset}/select/{datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
    cache=Path(a.processed_dir or ROOT/'data/processed/ec50_followup_v1'/a.dataset).resolve()
    base=base_config(a.dataset,cache,a.device)
    if a.processed_dir:base['processed_dir']=str(cache)
    if a.dataset=='size':base.update(defaults['size_overrides'])
    base.update(epochs=a.epochs,inference_policy='shared',lambda_diversity=0.,lambda_irm=0.,lambda_contrast=a.lambda_contrast,
                diversity_delay_epochs=0,diversity_ramp_epochs=0)
    base['erm_baseline']=a.model=='erm'
    if a.model=='erm':base.update(lambda_contrast=0.,lambda_diversity=0.,lambda_irm=0.)
    policies=('shared',) if a.model=='erm' else POLICIES
    seeds=a.seeds
    if a.smoke:
        _,fixture=make_plan(out,a.device,True)
        for k in ['dataset','raw_path','raw_format','splits','label_key','processed_dir','strict','batch_size','num_workers','pin_memory','hidden_dim','gnn_layers']:base[k]=fixture[k]
        base['epochs']=3;seeds=[5001];torch.set_num_threads(1)
    print(f'{a.dataset}: {len(seeds)} trajectories x {base["epochs"]} epochs; model={a.model}, {len(policies)} validation policy/policies; final test={a.evaluate_test}; output={out}',flush=True)
    if a.dry_run:return
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    out.mkdir(parents=True,exist_ok=False)
    write(out/'provenance.json',dict(synthetic=a.smoke,python=sys.version,torch=torch.__version__,
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [*ROOT.glob('*.py'),*(ROOT/'src').rglob('*.py'),*(ROOT/'scripts').glob('*.py')]}))
    if a.smoke:run_preprocessing(Config(base))
    records=[run_seed(base,out/f'seed{seed}',seed,a.evaluate_test) for seed in seeds]
    from src.experiments.common import _summarize
    summary={s:{p:{split:_summarize([r['results'][s][p][split]['auc'] for r in records])
        for split in records[0]['results'][s][p]} for p in policies} for s in policies}
    write(out/'summary.json',dict(runs=records,summary=summary,layout='summary[selection policy][inference policy][split]'))
    print(f'COMPLETE: {out/"summary.json"}',flush=True)

if __name__=='__main__':main()
