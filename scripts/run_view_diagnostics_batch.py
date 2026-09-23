"""Run or resume validation-only view diagnostics for all matched Assay controls."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def jobs(output):
    for condition in ('contrast01', 'contrast0', 'contrast001'):
        base = ROOT / ('outputs/assay/select/seeds7001_7010' if condition == 'contrast01'
                       else f'outputs/assay/controls/seeds7001_7010/{condition}')
        for seed in range(7001, 7011):
            for policy in ('shared', 'independent'):
                yield base/f'seed{seed}'/f'best_{policy}.pt', output/condition/f'seed{seed}'/policy

def complete(folder, checkpoint):
    try:
        report = json.loads((folder/'comparison.json').read_text())
        return (Path(report['checkpoint']).resolve() == checkpoint.resolve()
                and report.get('partial') is False and report.get('n', 0) > 0
                and bool(report.get('metrics')) and bool(report.get('pairs')))
    except (OSError, ValueError, KeyError, TypeError):
        return False

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--output-dir', default='outputs/assay/view_diagnostics/matched_7001_7010')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    output = Path(a.output_dir)
    if not output.is_absolute(): output = ROOT/output
    plan = list(jobs(output))
    missing = [str(c) for c,_ in plan if not c.is_file()]
    if missing:
        p.error('Missing checkpoints (upload/finish these first):\n'+'\n'.join(missing))
    if a.dry_run:
        print(f'All {len(plan)} checkpoints exist. Output: {output}')
        return
    # Linux advisory lock prevents two launchers using the same output directory.
    if sys.platform != 'linux':
        p.error('Run this launcher on the Linux server; --dry-run also works locally.')
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'batch.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('This diagnostics batch is already running.')
    check = 'import torch; ' + ("assert torch.cuda.is_available(), 'CUDA unavailable'" if a.device == 'cuda' else "print('CPU selected')")
    subprocess.run([sys.executable,'-c',check],cwd=ROOT,check=True)
    for index,(checkpoint, destination) in enumerate(plan,1):
        if complete(destination,checkpoint):
            print(f'[{index}/60] SKIP completed: {destination}',flush=True)
            continue
        # Preserve incomplete attempts. The worker requires a fresh output directory.
        attempt = destination
        retry = 0
        while attempt.exists():
            retry += 1
            attempt = destination.with_name(destination.name+f'_attempt{retry}')
            if complete(attempt,checkpoint): break
        if complete(attempt,checkpoint):
            print(f'[{index}/60] SKIP completed: {attempt}',flush=True)
            continue
        attempt.parent.mkdir(parents=True,exist_ok=True)
        log = attempt.with_name(attempt.name+'.log')
        print(f'[{index}/60] START {checkpoint}\nLog: {log}',flush=True)
        command = [sys.executable,'-u',str(ROOT/'main.py'),'experiment','compare-views',
                   str(checkpoint),'--device',a.device,'--output-dir',str(attempt)]
        with log.open('w') as stream:
            result = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode or not complete(attempt,checkpoint):
            print(f'FAILED. See {log}. Run this same command again after fixing the error.',flush=True)
            sys.exit(result.returncode or 1)
        print(f'[{index}/60] DONE',flush=True)
    print(f'ALL 60 COMPLETE: {output}',flush=True)

if __name__ == '__main__': main()
