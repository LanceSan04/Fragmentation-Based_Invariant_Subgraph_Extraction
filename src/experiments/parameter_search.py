"""Fresh validation-only Optuna search; first trial repeats the reference settings."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
import yaml
from src.experiments.search import run_search
from src.experiments.plans import make_plan
from src.data.preprocess import run_preprocessing
from train_sugar import Config


def plan(output,device,smoke=False,dataset='assay',cache_dir=None):
    base=yaml.safe_load((ROOT/'configs/diversity_schedule_off.yaml').read_text())
    if dataset == 'size':
        from src.experiments.plans import base_config
        base=base_config('size',Path(cache_dir or ROOT/'data/processed/ec50_followup_v1/size').resolve(),device)
    elif dataset != 'assay':
        raise ValueError('Unsupported dataset')
    elif cache_dir:
        base['processed_dir']=str(Path(cache_dir).resolve())
    base.update(output_root='',lambda_contrast=.1,lambda_diversity=0.,lambda_irm=0.,
                diversity_delay_epochs=0,diversity_ramp_epochs=0,device=device)
    from src.experiments.plans import workflow_defaults
    defaults=workflow_defaults()["parameter_search"]
    grid=defaults["grid"]
    reference={k:base[k] for k in grid}
    spec=dict(output_dir=str(output),study_name='ec50_parameters_v1',sampler='tpe',
              grid=grid,enqueue_trials=[reference],n_trials=defaults["n_trials"],search_repeats=defaults["search_repeats"],
              base_seed=defaults["assay_seed"],sampler_seed=defaults["sampler_seed"],common_search_seeds=True,auc_ma_window=1,
              maximize=True,overrides={})
    if dataset == 'size':
        spec.update(study_name='ec50_size_parameters_v1',base_seed=defaults["size_seed"])
    if smoke:
        _,fixture=make_plan(output,device,True)
        base.update({k:fixture[k] for k in ['dataset','raw_path','raw_format','splits','label_key',
                                         'processed_dir','strict','batch_size','num_workers',
                                         'pin_memory','hidden_dim','gnn_layers']})
        base['splits'].pop('ood_test',None)
        base.update(split_names=['train','ood_val'],epochs=2,early_stopping_min_epochs=0)
        spec['n_trials']=2
    base.update(test_split=None,report_splits=[],log_test_curve=False,
                loader_split_names=[base['train_split'],base['validation_split']])
    return spec,base


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--output-dir')
    parser.add_argument('--dataset',choices=['assay','size'],default='assay')
    parser.add_argument('--inference-policy',choices=['shared','independent'],default='shared')
    parser.add_argument('--cache-dir',help='Existing processed cache; Size defaults to its follow-up cache')
    args=parser.parse_args(argv);os.chdir(ROOT)
    prefix='outputs/smoke/tune' if args.smoke else f'outputs/{args.dataset}/tune'
    output=Path(args.output_dir or f'{prefix}/{datetime.now():%Y%m%d_%H%M%S}').resolve()
    spec,base=plan(output,args.device,args.smoke,args.dataset,args.cache_dir)
    base['inference_policy']=args.inference_policy
    print('Inference policy:',args.inference_policy)
    print(f"{spec['n_trials']} trials including reference x {spec['search_repeats']} seeds; max {base['epochs']} epochs/run.",flush=True)
    print('Search space:',spec['grid']);print('Output:',output)
    if args.dry_run:return
    import optuna
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if not args.smoke:
        if args.dataset == 'size':
            ready=json.loads((Path(base['processed_dir'])/'ready.json').read_text())
            if ready['dataset'] != base['dataset'] or ready['raw_sha256'] != hashlib.sha256(Path(base['raw_path']).read_bytes()).hexdigest():
                raise ValueError('Size cache source mismatch; use the matching prepared cache')
        for split in base['loader_split_names']:
            if not (Path(base['processed_dir'])/f'{split}.pt').is_file():
                raise FileNotFoundError(f"Missing cached {split} data in {base['processed_dir']}")
    output.mkdir(parents=True,exist_ok=False)
    sources=list(ROOT.glob('*.py'))+list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'configs').glob('*.yaml'))
    (output/'provenance.json').write_text(json.dumps(dict(synthetic=args.smoke,python=sys.version,
        torch=torch.__version__,optuna=optuna.__version__,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
    (output/'base_config.yaml').write_text(yaml.safe_dump(base))
    spec['_base_config_path']=str(output/'base_config.yaml')
    (output/'search_spec.json').write_text(json.dumps(spec,indent=2))
    if args.smoke:
        torch.set_num_threads(1);run_preprocessing(Config(base))
    run_search(spec,base,SimpleNamespace(epochs=None,device=args.device,dry_run=False,trials=None,repeats=None))
    print('Search returned. Inspect leaderboard states; no test evaluation was launched.',flush=True)


if __name__=='__main__':main()
