"""Sequential controlled contrastive sweep through the existing selection workflow."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['assay','size'],default='size')
    p.add_argument('--lambdas',type=float,nargs='+',default=[0,.01,.03,.3])
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(7001,7011)))
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--output-dir');p.add_argument('--processed-dir')
    p.add_argument('--smoke',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--evaluate-test',action='store_true',help='Opt in to exploratory test reporting after each seed selection')
    a=p.parse_args(argv)
    if len(set(a.lambdas))!=len(a.lambdas) or any(not math.isfinite(x) or x<0 for x in a.lambdas):p.error('Unique finite nonnegative lambdas required')
    if a.epochs<1 or len(set(a.seeds))!=len(a.seeds):p.error('Positive epochs and unique seeds required')
    dataset='smoke' if a.smoke else a.dataset
    out=Path(a.output_dir or ROOT/'outputs'/dataset/'contrastive_sweep'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    commands=[]
    for i,coefficient in enumerate(a.lambdas):
        name=f'condition{i}_lambda{coefficient:g}'
        cmd=[sys.executable,'-u',str(ROOT/'main.py'),'experiment','select','--dataset',a.dataset,
             '--model','fragment','--lambda-contrast',str(coefficient),'--device',a.device,
             '--epochs',str(a.epochs),'--seeds',*[str(s) for s in a.seeds],'--output-dir',str(out/name)]
        if a.processed_dir:cmd.extend(['--processed-dir',a.processed_dir])
        if a.smoke:cmd.append('--smoke')
        if a.evaluate_test:cmd.append('--evaluate-test')
        commands.append((name,cmd))
    print(f'{len(commands)} conditions; {1 if a.smoke else len(a.seeds)} seeds each; {3 if a.smoke else a.epochs} epochs. Output: {out}',flush=True)
    if a.dry_run:
        for _,cmd in commands:print(' '.join(cmd))
        return
    out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(dict(arguments=vars(a),commands=commands),indent=2))
    for name,cmd in commands:
        log=out/(name+'.log')
        print(f'START {name}; live output also saved to {log}',flush=True)
        with log.open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
            for line in child.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
            status=child.wait()
        if status:raise RuntimeError(f'{name} failed (exit {status}). See {log}; completed results preserved. No automatic resume.')
    (out/'complete.json').write_text(json.dumps({'conditions':[n for n,_ in commands]}))
    print(f'ALL CONDITIONS COMPLETE: {out}',flush=True)
