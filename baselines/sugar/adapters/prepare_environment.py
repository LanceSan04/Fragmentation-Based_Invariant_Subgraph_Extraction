"""Create a project-local reference overlay; never modifies the training venv.

The new environment reads existing Torch/PyG packages via a narrowly scoped .pth
file. New dependencies install into the overlay, avoiding another Torch/CUDA
download. Record this reuse explicitly; it is not the original 2021 environment.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import venv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--venv',required=True)
    p.add_argument('--install-reference-deps',action='store_true')
    a=p.parse_args()
    target=Path(a.venv).expanduser().resolve()
    if target.exists():raise FileExistsError('Fresh overlay required; do not overwrite an existing environment')
    venv.EnvBuilder(with_pip=True).create(target)
    interpreter=target/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    site=Path(subprocess.check_output([str(interpreter),'-c','import sysconfig;print(sysconfig.get_path("purelib"))'],text=True).strip())
    source=Path(sysconfig.get_path('purelib')).resolve()
    (site/'project_runtime.pth').write_text(str(source)+'\n',encoding='utf-8')
    packages=['ogb==1.3.6','dgl==1.1.3','setuptools<81']
    if a.install_reference_deps:
        subprocess.run([str(interpreter),'-m','pip','install','--disable-pip-version-check',*packages],check=True)
    (target/'reference_environment.json').write_text(json.dumps(dict(base_python=sys.executable,
        base_site_packages=str(source),reference_packages=packages,installed=a.install_reference_deps,
        note='Local overlay reuses project Torch/PyG; original training environment not modified.'),indent=2))
    print(interpreter)


if __name__=='__main__':main()
