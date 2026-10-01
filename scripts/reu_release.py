"""Build/verify source-only release bundles; no network or credentials involved."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DIRECTORIES=('src','scripts','tests','configs')
SUFFIXES={'.py','.yaml','.yml','.json','.sh','.md'}
ROOT_FILES=('main.py','train_sugar.py','requirements.txt','README.md','GROUP_GRAPH_LICENSE.md','RELMOLE_LICENSE.md')

def files(root):
    result={}
    for directory in DIRECTORIES:
        for p in (root/directory).rglob('*'):
            if '__pycache__' in p.parts or not p.is_file() or p.suffix not in SUFFIXES:continue
            if p.name == 'reu_ssh.py' or 'codex' in p.name.lower():continue
            if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise ValueError(f'Unsafe release path: {p}')
            result[p.relative_to(root).as_posix()]=p
    for name in ROOT_FILES:
        p=root/name
        if p.is_file():
            if p.is_symlink():raise ValueError(f'Symlink: {name}')
            result[name]=p
    for name in ['REMOTE_WORKING_AGREEMENT.md','REMOTE_OPERATIONS.md','BASELINE_STUDY.md',
                 'CROSS_DOMAIN_STUDY_2026_09_28.md']:
        p=root/'docs'/name
        if p.is_file():
            if p.is_symlink():raise ValueError(f'Symlink: {p}')
            result['docs/'+name]=p
    return result

def digest(data):return hashlib.sha256(data).hexdigest()

def approval_path(root):
    """Keep the approved digest outside the source snapshot it protects."""
    root=Path(root).resolve()
    return root.parent/'.approved-releases'/f'{root.name}.sha256'

def approved_hash(root,provided=None):
    """Use a one-time release pin, retaining explicit hashes for old releases."""
    path=approval_path(root)
    if path.is_symlink() or path.parent.is_symlink():raise ValueError(f'Unsafe release pin: {path}')
    if path.exists():
        stored=path.read_text(encoding='ascii').strip()
        if not re.fullmatch(r'[0-9a-f]{64}',stored):raise ValueError(f'Invalid release pin: {path}')
        if provided is not None and provided!=stored:raise ValueError('Explicit hash differs from approved release pin')
        return stored
    if provided is not None:return provided
    raise FileNotFoundError(f'No approved release pin at {path}; run reu_release.py verify --sha256 HASH --pin once')

def pin_verified_release(root,expected):
    """Persist a successfully verified digest once; never replace a different pin."""
    verify(root,Path(root)/'release-manifest.json',expected)
    path=approval_path(root)
    if path.parent.is_symlink() or path.is_symlink():raise ValueError(f'Unsafe release pin: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        approved_hash(root,expected)
    else:
        with path.open('x',encoding='ascii') as stream:stream.write(expected+'\n')
    return path

def verify(root,manifest_path,expected):
    raw=manifest_path.read_bytes()
    if digest(raw)!=expected:raise ValueError('Manifest hash does not match approved release')
    document=json.loads(raw);actual=files(root)
    wanted=document['files']
    if set(actual)!=set(wanted):raise ValueError('Release file set differs (missing or extra source files)')
    for name,p in actual.items():
        if digest(p.read_bytes())!=wanted[name]:raise ValueError(f'Release hash mismatch: {name}')
    return document

def build(root,output):
    selected=files(root);data={name:p.read_bytes() for name,p in sorted(selected.items())}
    try:head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):head=None
    document={'schema':1,'git_head':head,'note':'Exact content hashes, not Git HEAD alone, identify this potentially dirty source snapshot. No data or secrets bundled.',
              'files':{name:digest(content) for name,content in data.items()}}
    raw=json.dumps(document,indent=2,sort_keys=True).encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
        for name,content in data.items():z.writestr(name,content)
        z.writestr('release-manifest.json',raw)
    return digest(raw)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    b=sub.add_parser('build');b.add_argument('--output',type=Path,required=True)
    v=sub.add_parser('verify');v.add_argument('--manifest',type=Path,default=ROOT/'release-manifest.json');v.add_argument('--sha256',required=True)
    v.add_argument('--pin',action='store_true',help='After verification, save the approved digest outside this release for future launches')
    a=p.parse_args()
    if a.action=='build':print('Manifest SHA256:',build(ROOT,a.output))
    else:
        verify(ROOT,a.manifest,a.sha256)
        if a.pin:
            if a.manifest.resolve()!=ROOT/'release-manifest.json':p.error('--pin requires this release\'s default manifest')
            print(f'Release verified and pinned: {pin_verified_release(ROOT,a.sha256)}')
        else:print('Release verified')

if __name__=='__main__':main()
