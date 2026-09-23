"""Six validation-only runs: off/constant/ramp diversity on seeds 101 and 102."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for schedule in ['off','constant','ramp']:
        for seed in [101,102]:
            command=[sys.executable,'-u',str(ROOT/'main.py'),'train','--config',
                str(ROOT/'configs'/f'diversity_schedule_{schedule}.yaml'),'--device',args.device,
                '--seed',str(seed),'--sampler-seed',str(seed),
                '--run-name',f'{stamp}_{schedule}_seed{seed}']
            print(' '.join(command),flush=True)
            if not args.dry_run: subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
