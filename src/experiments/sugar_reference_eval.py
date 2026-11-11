"""Isolated shared-subgraph ENS evaluator; test requires a frozen protocol."""
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,'-u',str(root/'baselines/sugar/adapters/evaluate_reference.py'),
        *(sys.argv[1:] if argv is None else argv)],cwd=root,check=False)
    if result.returncode:raise RuntimeError('SuGAr reference evaluation failed; no automatic retry')
