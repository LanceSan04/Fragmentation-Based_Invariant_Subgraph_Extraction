"""Bounded, validation-only ERM tuning with a fixed four-configuration grid."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from src.experiments.plans import base_config
from src.experiments.selection import run_seed, write, verify_study_cache


GRID = (
    dict(learning_rate=0.0003, dropout=0.5, weight_decay=0.0),
    dict(learning_rate=0.0003, dropout=0.1, weight_decay=0.0001),
    dict(learning_rate=0.001, dropout=0.3, weight_decay=0.0001),
    dict(learning_rate=0.001, dropout=0.0, weight_decay=0.0001),
)


def rank_conditions(rows):
    """Select highest mean validation AUC; ties keep the first grid entry."""
    return max(rows, key=lambda row: (row['mean_ood_val_auc'], -row['index']))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['assay'],default='assay')
    p.add_argument('--processed-dir',required=True)
    p.add_argument('--raw-path',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--seeds',type=int,nargs='+',default=[8201,8202])
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--output-dir')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if a.epochs<1 or not a.seeds or len(set(a.seeds))!=len(a.seeds):
        p.error('Positive epochs and unique seeds required')
    cache=Path(a.processed_dir).resolve()
    raw=Path(a.raw_path).resolve()
    if not raw.is_file():raise FileNotFoundError(raw)
    verify_study_cache(cache,raw)
    for split in ('train','ood_val'):
        if not (cache/f'{split}.pt').is_file():raise FileNotFoundError(cache/f'{split}.pt')
    out=Path(a.output_dir or f'outputs/assay/erm_tune/{datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
    print(f'{len(GRID)} ERM conditions x {len(a.seeds)} seeds x {a.epochs} epochs; validation only; output={out}',flush=True)
    if a.dry_run:
        print(json.dumps(dict(grid=GRID,seeds=a.seeds),indent=2))
        return
    out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',dict(grid=GRID,seeds=a.seeds,epochs=a.epochs,cache=str(cache),raw_path=str(raw)))
    base=base_config(a.dataset,cache,a.device)
    base.update(raw_path=str(raw),processed_dir=str(cache),epochs=a.epochs,erm_baseline=True,
                selection_mode='learned',lambda_contrast=0.,lambda_diversity=0.,lambda_irm=0.)
    rows=[]
    for index,parameters in enumerate(GRID):
        condition=dict(base,**parameters)
        records=[run_seed(condition,out/f'condition{index}'/f'seed{seed}',seed,False)
                 for seed in a.seeds]
        aucs=[record['best']['shared']['auc'] for record in records]
        row=dict(index=index,parameters=parameters,seeds=a.seeds,
                 validation_aucs=aucs,mean_ood_val_auc=sum(aucs)/len(aucs))
        rows.append(row)
        write(out/'leaderboard.json',dict(conditions=rows,complete=False))
    winner=rank_conditions(rows)
    write(out/'leaderboard.json',dict(conditions=rows,winner=winner,complete=True))
    print(f'ERM validation winner: condition{winner["index"]} {winner["parameters"]}',flush=True)


if __name__=='__main__':main()
