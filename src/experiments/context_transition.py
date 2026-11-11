"""Explicit sequential historical-test / official-data EC50 comparison. No tuning."""
import argparse
import json
import math
from pathlib import Path

from src.experiments.checkpoints import digest
from src.experiments.context_features import main as feature_main, validate as validate_feature
from src.experiments.context_test import evaluate_frozen
from src.experiments.cross_domain import verify_cache
from src.experiments.fragment_tuning import contained
from src.experiments.plans import ROOT
from src.experiments.sugar_reference_search import write


def historical_entries(spec,project):
    record=contained(project,spec['historical_record'])
    completed=record/'results/completed.json'
    if digest(completed)!=spec['historical_completed_sha256']:
        raise ValueError('Historical completion record changed')
    launch=json.loads((record/'launch.json').read_text())
    if launch['status']!='succeeded' or launch['manifest_sha256']!=spec['historical_manifest_sha256']:
        raise ValueError('Historical source/launch differs from frozen study')
    saved=json.loads(completed.read_text());rows=saved['runs']
    expected={(d,a,s) for d in spec['datasets'] for a in spec['arms'] for s in spec['historical_seeds']}
    if len(rows)!=30 or len(rows)!=len(expected) or {(r['dataset'],r['arm'],r['seed']) for r in rows}!=expected:
        raise ValueError('Historical membership changed')
    protocol=saved['plan']['protocol']
    if protocol['epochs']!=150 or protocol['datasets']!=spec['historical_data']:
        raise ValueError('Historical recipe/data changed')
    caches={}
    for row in spec['historical_data']:
        raw=contained(project,row['raw_path']);cache=contained(project,row['cache_path'])
        if digest(raw)!=row['raw_sha256']: raise ValueError('Historical raw source changed')
        cfg=dict(protocol['base'],dataset='drugood_lbap_core_'+row['dataset'],
                 raw_path=str(raw),processed_dir=str(cache))
        verify_cache(cfg);caches[row['dataset']]=str(cache)
    for row in rows:
        Path(row['checkpoint']).resolve().relative_to((record/'results').resolve())
        if row['status']!='completed' or digest(row['checkpoint'])!=row['checkpoint_sha256']:
            raise ValueError('Historical checkpoint changed/incomplete')
    return rows,caches


def require_pilot(report,out):
    if len(report['runs'])!=6 or report['test_evaluated']:
        raise ValueError('Pilot scope/test status changed')
    for row in report['runs']:
        if row['status']!='completed' or not math.isfinite(row['validation']['auc']):
            raise ValueError('Pilot failed')
        path=out/row['dataset']/row['arm']/('seed'+str(row['seed']))/'history.json'
        history=json.loads(path.read_text())
        if [h['epoch'] for h in history]!=list(range(1,6)) or any(not math.isfinite(h['train_loss']) for h in history):
            raise ValueError('Pilot has missing epochs or nonfinite losses')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/ec50_data_transition_20261029.json'))
    p.add_argument('--project-root',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--stage',choices=['historical-test','prepare-pilot','full','all'],default='historical-test')
    p.add_argument('--pilot-results',type=Path,help='Required successful official pilot directory for standalone full stage')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv);spec=json.loads(Path(a.config).read_text())
    if (spec['schema']!=1 or spec['datasets']!=['ec50_assay','ec50_scaffold','ec50_size']
        or spec['arms']!=['scorer_only','scorer_and_classifier']
        or spec['historical_seeds']!=[11011,11012,11013,11014,11015]
        or spec['inference_policy']!='independent'):
        raise ValueError('Transition scope changed')
    pilot_path=contained(ROOT,spec['pilot_config']);full_path=contained(ROOT,spec['full_config'])
    pilot=json.loads(pilot_path.read_text());full=json.loads(full_path.read_text())
    validate_feature(pilot);validate_feature(full)
    if pilot['datasets']!=full['datasets'] or pilot['base']!=full['base']:
        raise ValueError('Pilot and full recipes/data differ')
    if pilot['seeds']!=[11201] or full['seeds']!=[11211,11212,11213,11214,11215]:
        raise ValueError('Unexpected official-data seeds')
    plan=dict(stage=a.stage,config_sha256=digest(a.config),
        historical_test_checkpoints=30 if a.stage in ('historical-test','all') else 0,
        new_training_runs=(6 if a.stage in ('prepare-pilot','all') else 0)+(30 if a.stage in ('full','all') else 0),
        official_test_checkpoints=30 if a.stage in ('full','all') else 0,
        pilot_epochs=5,full_epochs=150,automatic_retry=False,no_tuning=True,
        pilot_gate='Six completed finite five-epoch runs and successful validation reloads; not an AUC-performance gate',
        protocol=spec,pilot_config_sha256=digest(pilot_path),full_config_sha256=digest(full_path))
    if not a.execute: print(json.dumps(plan,indent=2));return plan
    project=a.project_root.resolve();out=a.output_dir.resolve()
    if a.stage in ('full','prepare-pilot','all'):
        for row in pilot['datasets']:
            if digest(contained(project,row['raw_path']))!=row['raw_sha256']:
                raise ValueError('Official raw source changed')
    old_rows,old_caches=historical_entries(spec,project) if a.stage in ('historical-test','all') else (None,None)
    if a.stage=='full':
        if not a.pilot_results: p.error('Standalone full requires --pilot-results')
        report=json.loads((a.pilot_results/'completed.json').read_text())
        if report['plan']['config_sha256']!=digest(pilot_path): raise ValueError('Pilot protocol differs')
        require_pilot(report,a.pilot_results)
    out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan);progress=[]
    def features(config,subdir,extra):
        return feature_main(['--config',str(config),'--project-root',str(project),
            '--output-dir',str(out/subdir),'--device',a.device,'--execute',*extra])
    try:
        if old_rows is not None:
            evaluate_frozen(old_rows,old_caches,out/'historical_test',a.device)
            progress.append('historical_test');write(out/'progress.json',progress)
        if a.stage in ('prepare-pilot','all'):
            features(pilot_path,'official_prepare',['--stage','prepare'])
            progress.append('official_prepare');write(out/'progress.json',progress)
            report=features(pilot_path,'official_pilot',[])
            require_pilot(report,out/'official_pilot')
            progress.append('official_pilot');write(out/'progress.json',progress)
        if a.stage in ('full','all'):
            features(full_path,'official_full',['--evaluate-test'])
            progress.append('official_full_and_test');write(out/'progress.json',progress)
        report=dict(plan=plan,completed_stages=progress,no_additional_runs=True)
        write(out/'completed.json',report);return report
    except Exception as error:
        write(out/'failed.json',dict(error=str(error),completed_stages=progress))
        raise


if __name__=='__main__': main()
