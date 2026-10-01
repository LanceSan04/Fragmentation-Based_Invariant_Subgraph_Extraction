"""Linux launcher for an approved main.py experiment; use inside tmux/scheduler."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .reu_release import approved_hash,verify
else:
    from reu_release import approved_hash,verify
ROOT=Path(__file__).resolve().parents[1]

def within_home(path,home):
    result=Path(path).expanduser().resolve()
    result.relative_to(home.resolve())
    return result

def acquire_lock(path):
    import fcntl
    if path.is_symlink():raise ValueError('Lock file must not be a symlink')
    fd=os.open(str(path),os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:
        os.close(fd);raise
    return fd

def other_project_jobs(root):
    found=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if int(p.parent.name)==os.getpid() or p.stat().st_uid!=os.getuid():continue
            words=p.read_bytes().split(b'\0')
            if not words or b'python' not in Path(os.fsdecode(words[0])).name.encode():continue
            cwd=(p.parent/'cwd').resolve()
            if cwd==root or root in cwd.parents or any(str(root).encode() in x for x in words):found.append(int(p.parent.name))
        except (OSError,ValueError):continue
    return found

def require_tmux_session():
    pane=os.environ.get('TMUX_PANE')
    if not os.environ.get('TMUX') or not pane:
        raise RuntimeError('Attach to the existing sugar_run tmux session before launching a pipeline')
    session=subprocess.check_output(['tmux','display-message','-p','-t',pane,'#{session_name}'],text=True).strip()
    if session!='sugar_run':
        raise RuntimeError(f'Remote pipelines must run in sugar_run, not {session!r}')
    return session

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-sha256',help='Optional legacy override; a pinned verified release needs no hash argument')
    p.add_argument('--record-dir',type=Path,required=True,help='Fresh directory under your home; contains results and launch records')
    p.add_argument('--gpu',required=True,help='cpu or one physical GPU index approved for this job')
    p.add_argument('arguments',nargs=argparse.REMAINDER)
    a=p.parse_args();args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    if sys.platform!='linux':p.error('This launcher runs on Linux inside tmux or a scheduler allocation')
    if not args or args[0] not in ['selection-gating-probe','cross-domain','select','atom-budget-study','fragment-ratio-sweep','connectivity-probe','assay-followup','contrastive-study','scorer-study','tune','erm-tune','baseline-eval','contrastive-sweep','benchmark','compare-views','compare-inference','diagnose','overnight','visualize-views']:
        p.error('Supply a supported experiment workflow after --')
    if any(x.split('=')[0] in ['--output-dir','--device'] for x in args):p.error('Launcher supplies --device and --output-dir; omit these from workflow arguments')
    if a.gpu not in ['cpu','allocated'] and not a.gpu.isdecimal():p.error('--gpu must be cpu, allocated, or one nonnegative physical index')
    if a.gpu=='allocated' and (not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']):p.error('allocated requires exactly one scheduler-provided CUDA_VISIBLE_DEVICES entry')
    session=require_tmux_session()
    home=Path.home().resolve();root=within_home(ROOT,home);record=within_home(a.record_dir,home)
    # One fixed per-account lock coordinates different checkouts and agent sessions.
    state=within_home(home/'.local/state/reu',home);state.mkdir(parents=True,exist_ok=True)
    try:lock=acquire_lock(state/'pipeline.lock')
    except BlockingIOError:p.error('Another guarded REU pipeline is still running; no job started')
    try:
        active=other_project_jobs(root)
        if active:p.error('Project Python jobs already running (PIDs): '+','.join(map(str,active)))
        manifest_sha256=approved_hash(root,a.manifest_sha256)
        document=verify(root,root/'release-manifest.json',manifest_sha256)
        record.mkdir(parents=True,exist_ok=False)
        command=[sys.executable,'-u',str(root/'main.py'),'experiment',*args,
                 '--device','cpu' if a.gpu=='cpu' else 'cuda','--output-dir',str(record/'results')]
        env=os.environ.copy()
        if a.gpu!='allocated':env['CUDA_VISIBLE_DEVICES']='' if a.gpu=='cpu' else a.gpu
        metadata={'command':command,'cwd':str(root),'gpu':a.gpu,'manifest_sha256':manifest_sha256,'cuda_visible_devices':env['CUDA_VISIBLE_DEVICES'],
                  'started_at':datetime.now(timezone.utc).isoformat(),'status':'starting','tmux_session':session,
                  'note':'Approval is a human workflow requirement, not enforced by this program. No automatic time limit/retry.'}
        (record/'source-manifest.json').write_text(json.dumps(document,indent=2))
        def save(): (record/'launch.json').write_text(json.dumps(metadata,indent=2))
        save()
        with (record/'console.log').open('x') as log:
            # Inherited lock stays held if the launcher dies while the pipeline survives.
            child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock,))
            metadata.update(pid=child.pid,status='running');save()
            print(f'PID {child.pid}; log: {record/"console.log"}; results: {record/"results"}',flush=True)
            code=child.wait()
        metadata.update(status='succeeded' if code==0 else 'failed',exit_code=code,ended_at=datetime.now(timezone.utc).isoformat());save()
        return code
    finally:os.close(lock)

if __name__=='__main__':sys.exit(main())
