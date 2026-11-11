"""Run the isolated upstream SuGAr validation-only adapter from main.py."""
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,'-u',str(root/'baselines/sugar/adapters/run.py'),
                           *(sys.argv[1:] if argv is None else argv)],cwd=root,check=False)
    if result.returncode:raise RuntimeError('Upstream validation-only adapter failed; no automatic retry')
