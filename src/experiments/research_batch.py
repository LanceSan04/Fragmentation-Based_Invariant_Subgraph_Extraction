"""Approved-scope sequential package: upstream pilot, Ki, then GOOD.

Preparation default is dry-run. Execution requires --execute and the usual
remote exact-command approval. No retries, hidden search or full upstream sweep.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from src.experiments.study_queue import write

ROOT=Path(__file__).resolve().parents[2]


def commands(args):
    output=Path(args.output_dir).resolve()
    overlay=Path(args.reference_env).resolve() if args.reference_env else None
    reference_python=(overlay/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')) if overlay else Path(sys.executable)
    steps=[]
    if not args.smoke:
        if overlay is None:raise ValueError('--reference-env required for environment isolation')
        steps.append(dict(name='reference_environment',command=[sys.executable,
            str(ROOT/'baselines/sugar/adapters/prepare_environment.py'),'--venv',str(overlay),'--install-reference-deps']))
    upstream=[str(reference_python),'-u',str(ROOT/'baselines/sugar/adapters/run.py'),
              '--raw-path',str(Path(args.raw_root).resolve()/'drugood/lbap_core_ec50_assay.json'),
              '--epochs','2','--views','2' if args.smoke else '10','--device',args.device,
              '--output-dir',str(output/'steps/upstream_pilot')]
    if args.smoke:upstream+=['--smoke']
    steps.append(dict(name='upstream_pilot',depends_on=None if args.smoke else 'reference_environment',command=upstream))
    for group in ('ki','good'):
        datasets=['ki_scaffold','ki_size'] if group=='ki' else ['good_hiv_scaffold','good_hiv_size']
        command=[sys.executable,'-u',str(ROOT/'main.py'),'experiment','mentor-confirm',
                 '--datasets',*datasets,'--raw-root',str(Path(args.raw_root).resolve()),
                 '--cache-root',str(Path(args.cache_root).resolve()),
                 '--selection-source',str(Path(args.selection_source).resolve()),
                 '--source-audit',str(ROOT/'configs/ki_good_sources_20261016.yaml'),
                 '--device',args.device,'--output-dir',str(output/('steps/'+group)),
                 '--evaluate-primary-test']
        if args.smoke:command+=['--smoke']
        steps.append(dict(name=group,command=command))
    return steps


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--raw-root',default='data/raw')
    p.add_argument('--cache-root',default='data/processed/ki_good_frozen_20261016_v1')
    p.add_argument('--selection-source',required=True)
    p.add_argument('--reference-env')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(argv);steps=commands(a)
    spec=dict(native_training_runs=16 if a.smoke else 80,upstream_pilot_runs=1,
              native_epochs=5 if a.smoke else 100,upstream_epochs=2,
              steps=steps,synthetic=a.smoke,no_retry=True,full_upstream_benchmark=False)
    if not a.execute:
        print(json.dumps(spec,indent=2));return spec
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'steps').mkdir();(out/'logs').mkdir()
    write(out/'plan.json',spec)
    status={}
    for step in steps:
        name=step['name'];dependency=step.get('depends_on')
        if dependency and status[dependency]['status']!='succeeded':
            status[name]=dict(status='blocked',reason='Environment prerequisite failed')
            write(out/'status.json',status);continue
        if not a.smoke and shutil.disk_usage(out).free<30*1024**3:
            raise RuntimeError('Low disk; no further stages launched')
        status[name]=dict(status='running',command=step['command']);write(out/'status.json',status)
        print('START '+name,flush=True)
        log=out/'logs'/(name+'.log')
        with log.open('x',encoding='utf-8') as stream:
            result=subprocess.run(step['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
        status[name].update(status='succeeded' if result.returncode==0 else 'failed',exit_code=result.returncode)
        write(out/'status.json',status)
        if result.returncode:
            tail=log.read_text(encoding='utf-8',errors='replace')[-20000:].lower()
            if any(message in tail for message in ('out of memory','no space left','cuda error','device-side assert')):
                raise RuntimeError('Resource/device failure; entire package stopped')
            print(name+' failed; retained logs; continuing independent stages without retry',flush=True)
    if any(v['status']!='succeeded' for v in status.values()):
        raise RuntimeError('Some package stages failed/blocked; inspect status.json; no retry')
    write(out/'completed.json',dict(**spec,stages=status))
    return status
