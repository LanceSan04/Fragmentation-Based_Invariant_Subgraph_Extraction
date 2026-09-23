"""Dry-run-first Linux workspace migration. Stop training jobs before --apply."""
import argparse,hashlib,json,sys
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def plan(root):
    pairs=[]
    cache=root/'outputs_overnight_losses_20260913_102501/processed'
    if cache.exists():pairs.append((cache,root/'data/processed/ec50_followup_v1/assay'))
    known={'outputs_diversity_schedule':'outputs/assay/diversity_schedule',
           'outputs_policy_comparison':'outputs/assay/policy_comparison',
           'outputs_multiview_ablation':'outputs/assay/multiview_ablation',
           'outputs_preliminary_test':'outputs/assay/preliminary_test',
           'outputs_parameter_winner_test':'outputs/assay/parameter_winner_test',
           'logs':'outputs/logs','final_testing_outputs':'outputs/legacy/final_testing_outputs',
           'search_best_individual_auc_v1':'outputs/search/best_individual_auc_v1'}
    for source in sorted(root.iterdir()):
        if source.name in known or source.name.startswith('outputs_'):
            dest=root/known.get(source.name,'outputs/legacy/'+source.name)
            pairs.append((source,dest))
    for source,dest in pairs:
        source.resolve().relative_to(root.resolve());dest.resolve().relative_to(root.resolve())
        if source.is_symlink() or any(x.is_symlink() for x in source.rglob('*')):raise ValueError(f'Symlink requires manual review: {source}')
        if dest.exists():raise FileExistsError(f'Destination exists; no merging or overwriting: {dest}')
    return pairs

def inventory(path):
    files=[path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
    result={}
    for p in files:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        result[str(p.relative_to(path)) if p!=path else '.']={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');a=parser.parse_args()
    pairs=plan(ROOT)
    for source,dest in pairs:print(f'{source} -> {dest}',flush=True)
    if not a.apply:
        print(f'DRY RUN: {len(pairs)} moves. Stop all project jobs, then rerun with --apply.');return
    if sys.platform!='linux':parser.error('--apply is for the Linux server only')
    # Refuse when another project Python job is visible in /proc.
    import os
    active=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if int(p.parent.name)==os.getpid():continue
            args=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
            cwd=(p.parent/'cwd').resolve()
            if 'python' in args and (str(ROOT) in args or cwd==ROOT or ROOT in cwd.parents):active.append(p.parent.name)
        except (OSError,ValueError):pass
    if active:parser.error('Project Python processes still running: '+', '.join(active))
    if not pairs:print('Already migrated.');return
    record=ROOT/'outputs/migrations'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');record.mkdir(parents=True,exist_ok=False)
    entries=[dict(source=str(s),destination=str(d),status='pending') for s,d in pairs]
    manifest=record/'moves.json';manifest.write_text(json.dumps(entries,indent=2))
    for i,(source,dest) in enumerate(pairs):
        before=inventory(source);(record/f'inventory_{i}.json').write_text(json.dumps(before,indent=2))
        dest.parent.mkdir(parents=True,exist_ok=True);source.rename(dest)
        entries[i]['status']='moved';manifest.write_text(json.dumps(entries,indent=2))
        if before!=inventory(dest):raise RuntimeError(f'Post-move verification failed: {dest}')
        entries[i]['status']='verified';manifest.write_text(json.dumps(entries,indent=2))
    print(f'COMPLETE. Recovery map: {manifest}',flush=True)

if __name__=='__main__':main()
