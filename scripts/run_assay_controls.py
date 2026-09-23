"""Sequential matched Assay controls. Run inside tmux; existing output refuses restart."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--output-dir', default='outputs/assay/controls/seeds7001_7010')
    a = p.parse_args()
    out = Path(a.output_dir)
    if not out.is_absolute(): out = ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    for name, model, coefficient in [('contrast0','fragment','0'),('contrast001','fragment','0.01'),('erm','erm','0')]:
        cmd = [sys.executable,'-u',str(ROOT/'main.py'),'experiment','select',
               '--dataset','assay','--model',model,'--lambda-contrast',coefficient,
               '--seeds',*[str(x) for x in range(7001,7011)],'--epochs','100',
               '--evaluate-test','--device',a.device,'--output-dir',str(out/name)]
        if a.smoke: cmd.append('--smoke')
        print('START:',name,flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True)
        print('FINISHED:',out/name/'summary.json',flush=True)
    print('ALL THREE CONDITIONS COMPLETE',flush=True)

if __name__=='__main__': main()
