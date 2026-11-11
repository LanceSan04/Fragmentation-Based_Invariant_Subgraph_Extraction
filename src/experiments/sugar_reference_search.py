"""Fixed-budget SuGAr reference search; validation only, sequential, no retries."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def validate(spec):
    if (spec['schema']!=1 or spec['dataset']!='ec50_assay' or spec['objective']!='all_mean'
        or spec['contrastive_indexing']!='released' or spec['epochs']!=40
        or spec['views']!=10 or spec['initialization']!=1):
        raise ValueError('Frozen reference search scope changed')
    trials=spec['trials']
    if len(trials)!=18 or len({tuple(row) for row in trials})!=18:raise ValueError('Expected 18 unique trials')
    for contrast,penalty,sample in trials:
        if contrast not in [.5,1,2,4,8,16,32] or penalty not in [.01,.1,1,2,4,8] or sample not in [85,90,95]:
            raise ValueError('Trial outside documented paper ranges')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/sugar_overnight_20261017.json'))
    p.add_argument('--raw-path',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(argv);spec=json.loads(Path(a.config).read_text());validate(spec)
    trials=spec['trials'][:2] if a.smoke else spec['trials']
    plan=dict(spec,trials=trials,training_runs=len(trials),epochs=2 if a.smoke else spec['epochs'],
              views=2 if a.smoke else spec['views'],synthetic=a.smoke,no_test=True,no_retry=True,
              selection='Highest all_mean OOD validation AUC; tie resolved by earlier trial ID')
    if not a.execute:print(json.dumps(plan,indent=2));return plan
    raw=Path(a.raw_path).resolve()
    if not a.smoke and hashlib.sha256(raw.read_bytes()).hexdigest()!=spec['raw_sha256']:
        raise ValueError('Raw dataset hash differs from frozen plan')
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'logs').mkdir();(out/'trials').mkdir();write(out/'plan.json',plan)
    rows=[];start=time.perf_counter()
    for i,(contrast,penalty,sample) in enumerate(trials):
        if not a.smoke and shutil.disk_usage(out).free<10*1024**3:
            raise RuntimeError('Low disk; no further trials launched')
        trial=out/'trials'/f'{i+1:02d}';trial.mkdir()
        row=dict(trial=i+1,contrast=contrast,diversity_penalty=penalty,sample_rate=sample,status='running')
        rows.append(row);write(out/'status.json',rows)
        train=[sys.executable,'-u',str(ROOT/'baselines/sugar/adapters/run.py'),
            '--raw-path',str(raw),'--output-dir',str(trial/'train'),'--device',a.device,
            '--epochs','2' if a.smoke else str(spec['epochs']),
            '--views','2' if a.smoke else str(spec['views']),'--seed',str(spec['initialization']),
            '--contrastive-indexing','released','--contrast',str(contrast),
            '--diversity-penalty',str(penalty),'--sample-rate',str(sample)]
        if a.smoke:train+=['--smoke']
        evaluate=[sys.executable,'-u',str(ROOT/'baselines/sugar/adapters/evaluate_reference.py'),
            '--checkpoint',str(trial/'train/best_views.pt'),'--raw-path',str(raw),
            '--output-dir',str(trial/'validation'),'--device',a.device]
        for stage,command in [('training',train),('validation',evaluate)]:
            row.update(stage=stage,command=command);write(out/'status.json',rows)
            print(f'Trial {i+1}/{len(trials)}: {stage}',flush=True)
            with (out/'logs'/f'{i+1:02d}_{stage}.log').open('x') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
            if result.returncode:
                row.update(status='failed',exit_code=result.returncode);write(out/'status.json',rows)
                raise RuntimeError(f'Trial {i+1} {stage} failed; batch stopped without retry')
        evaluation=json.loads((trial/'validation/completed.json').read_text())
        if evaluation['test_evaluated'] or evaluation['synthetic']!=a.smoke:
            raise RuntimeError('Unexpected evaluation scope')
        auc=evaluation['variants']['all_mean']['auc']
        if not math.isfinite(auc):raise RuntimeError('Nonfinite objective')
        row.update(status='completed',validation_auc=auc,checkpoint=str(trial/'train/best_views.pt'))
        write(out/'status.json',rows)
        write(out/'leaderboard.json',sorted(rows,key=lambda r:(-r['validation_auc'],r['trial'])))
    winner=max(rows,key=lambda r:r['validation_auc'])
    write(out/'completed.json',dict(training_runs=len(rows),synthetic=a.smoke,test_evaluated=False,
        wall_seconds=time.perf_counter()-start,best=winner,objective='all_mean',
        note='Single-initialization validation selection; no final five-seed or test benchmark'))
    return winner
