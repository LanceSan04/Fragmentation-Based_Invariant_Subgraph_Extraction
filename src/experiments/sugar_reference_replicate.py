"""Five frozen-recipe replications, followed by primary-only frozen test scoring."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from src.experiments.sugar_reference_search import write

ROOT=Path(__file__).resolve().parents[2]


def validate(spec,selection_bytes):
    if hashlib.sha256(selection_bytes).hexdigest()!=spec['selection_sha256']:
        raise ValueError('Search selection source hash changed')
    selection=json.loads(selection_bytes);winner=selection['best']
    if selection['training_runs']!=18 or selection['test_evaluated'] or selection['objective']!='all_mean':
        raise ValueError('Expected completed validation-only search')
    if spec['initializations']!=[2,3,4,5,6] or spec['epochs']!=40 or spec['views']!=10:
        raise ValueError('Replication budget changed')
    if spec['primary']!='all_mean' or spec['contrastive_indexing']!='released':
        raise ValueError('Reference policy changed')
    for key in ('contrast','diversity_penalty','sample_rate'):
        if spec[key]!=winner[key]:raise ValueError('Recipe differs from selected winner')


def summarize(rows):
    if any(row['status']!='completed' for row in rows):raise ValueError('Incomplete replication')
    result={'runs':rows,'primary':'all_mean','std_convention':'Sample standard deviation (ddof=1)'}
    for metric in ('validation_auc','test_auc'):
        values=[r[metric] for r in rows]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite AUC')
        result[metric]=dict(mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else None,n=len(values))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/sugar_winner_replication_20261017.json'))
    p.add_argument('--selection-source',required=True)
    p.add_argument('--raw-path',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args(argv);spec=json.loads(Path(a.config).read_text())
    validate(spec,Path(a.selection_source).read_bytes())
    seeds=spec['initializations'][:2] if a.smoke else spec['initializations']
    plan=dict(spec,initializations=seeds,training_runs=len(seeds),synthetic=a.smoke,
        epochs=2 if a.smoke else spec['epochs'],views=2 if a.smoke else spec['views'],
        test_scope='all_mean only, after every training and validation stage passes',
        retries=0,hyperparameter_search=False)
    if not a.execute:print(json.dumps(plan,indent=2));return plan
    raw=Path(a.raw_path).resolve()
    if not a.smoke and hashlib.sha256(raw.read_bytes()).hexdigest()!=spec['raw_sha256']:
        raise ValueError('Raw dataset changed')
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'logs').mkdir();(out/'runs').mkdir();write(out/'plan.json',plan)
    rows=[]
    def execute(command,row,stage):
        if not a.smoke and shutil.disk_usage(out).free<10*1024**3:raise RuntimeError('Low disk')
        row.update(stage=stage,command=command);write(out/'status.json',rows)
        print(f"Initialization {row['initialization']}: {stage}",flush=True)
        with (out/'logs'/f"init{row['initialization']}_{stage}.log").open('x') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
        if result.returncode:
            row.update(status='failed',exit_code=result.returncode);write(out/'status.json',rows)
            raise RuntimeError('Replication stage failed; no retry or further stages')
    for seed in seeds:
        run=out/'runs'/f'init{seed}';run.mkdir()
        row=dict(initialization=seed,status='running');rows.append(row)
        train=[sys.executable,'-u',str(ROOT/'baselines/sugar/adapters/run.py'),
            '--raw-path',str(raw),'--output-dir',str(run/'train'),'--device',a.device,
            '--epochs',str(plan['epochs']),'--views',str(plan['views']),'--seed',str(seed),
            '--contrastive-indexing','released','--contrast',str(spec['contrast']),
            '--diversity-penalty',str(spec['diversity_penalty']),'--sample-rate',str(spec['sample_rate'])]
        if a.smoke:train+=['--smoke']
        execute(train,row,'training')
        evaluation=[sys.executable,'-u',str(ROOT/'baselines/sugar/adapters/evaluate_reference.py'),
            '--checkpoint',str(run/'train/best_views.pt'),'--raw-path',str(raw),
            '--output-dir',str(run/'validation'),'--device',a.device]
        execute(evaluation,row,'validation')
        valpath=run/'validation/completed.json';valbytes=valpath.read_bytes();val=json.loads(valbytes)
        if val['test_evaluated'] or val['synthetic']!=a.smoke:raise ValueError('Unexpected validation scope')
        protocol=dict(primary='all_mean',checkpoint_sha256=val['checkpoint_sha256'],
            validation_sha256=hashlib.sha256(valbytes).hexdigest(),
            raw_sha256=val['provenance']['source'].get('raw_sha256'),
            variants={'all_mean':val['variants']['all_mean']['views']})
        write(run/'frozen_test_protocol.json',protocol)
        row.update(status='validation_passed',validation_auc=val['variants']['all_mean']['auc'])
        write(out/'status.json',rows)
    # No test scoring can start until all five training/validation stages passed.
    for row in rows:
        run=out/'runs'/f"init{row['initialization']}"
        command=[sys.executable,'-u',str(ROOT/'baselines/sugar/adapters/evaluate_reference.py'),
            '--checkpoint',str(run/'train/best_views.pt'),'--raw-path',str(raw),
            '--output-dir',str(run/'test'),'--device',a.device,
            '--test-protocol',str(run/'frozen_test_protocol.json'),
            '--frozen-validation',str(run/'validation/completed.json')]
        execute(command,row,'test')
        result=json.loads((run/'test/completed.json').read_text())
        if not result['test_evaluated'] or set(result['variants'])!={'all_mean'}:
            raise ValueError('Unexpected test scope')
        row.update(status='completed',test_auc=result['variants']['all_mean']['auc'])
        write(out/'status.json',rows)
    report=summarize(rows);report.update(synthetic=a.smoke,training_runs=len(rows))
    write(out/'completed.json',report)
    return report
