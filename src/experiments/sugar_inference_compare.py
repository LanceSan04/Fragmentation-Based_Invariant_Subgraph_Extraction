"""Run the isolated, no-training SuGAr inference comparison."""
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,'-u',str(root/'baselines/sugar/adapters/compare_inference.py'),
        *(sys.argv[1:] if argv is None else argv)],cwd=root,check=False)
    if result.returncode:raise RuntimeError('Inference comparison failed; no automatic retry')
