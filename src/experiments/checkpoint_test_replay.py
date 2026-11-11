"""Fresh deterministic CPU test forwards for frozen original and rerun checkpoints."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.experiments.checkpoints import digest,evaluate_saved
from src.experiments.fragment_tuning import contained
from src.experiments.plans import ROOT
from src.experiments.selection import validation_execution
from src.experiments.sugar_reference_search import write

SEEDS={'ec50_assay':[11422,11420,11409,11415,11408],
       'ec50_scaffold':[11413,11403,11409,11414,11401],
       'ec50_size':[11401,11413,11430,11427,11428]}


def validate(spec):
    expected={(c,d,s) for c in ('original','first_rerun') for d,seeds in SEEDS.items() for s in seeds}
    if (spec['schema']!=1 or spec['batch_size']!=64 or spec['torch_threads']!=1
            or spec['inference_policy']!='independent' or spec['passes_per_checkpoint']!=2
            or spec['checkpoint_count']!=30 or spec['training_runs']!=0 or len(spec['rows'])!=30
            or {(r['cohort'],r['dataset'],r['seed']) for r in spec['rows']}!=expected
            or {d['dataset'] for d in spec['datasets']}!=set(SEEDS)):
        raise ValueError('Frozen checkpoint replay scope changed')


def compare_predictions(actual,reference):
    with np.load(actual) as a,np.load(reference) as b:
        fields=('labels','probabilities','assays','row_index')
        equal={k:bool(a[k].dtype==b[k].dtype and np.array_equal(a[k],b[k])) for k in fields}
        same_shape=a['probabilities'].shape==b['probabilities'].shape
        difference=(float(np.max(np.abs(a['probabilities'].astype(float)-b['probabilities'].astype(float))))
                    if same_shape and a['probabilities'].size else None)
        return dict(exact=all(equal.values()),fields_equal=equal,max_probability_difference=difference,
            auc=float(roc_auc_score(a['labels'],a['probabilities'])),
            reference_auc=float(roc_auc_score(b['labels'],b['probabilities'])))


def replay(rows,out,passes=2):
    """All validation checks precede test passes; every forward restores a fresh model."""
    out.mkdir(parents=True,exist_ok=False);validation=[];tests=[]
    try:
        for row in rows:
            if digest(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            run=out/row['cohort']/row['dataset']/f"seed{row['seed']}";run.mkdir(parents=True)
            with validation_execution('cpu'):
                metrics=evaluate_saved(row['checkpoint'],'ood_val',Path(row['cache']),'cpu','independent',row['seed'],
                    prediction_path=run/'validation_predictions.npz')
            comparison=compare_predictions(run/'validation_predictions.npz',row['validation_reference'])
            if metrics['auc']!=row['validation_auc'] or not comparison['exact']:
                raise ValueError('Validation replay changed; no test forwards allowed')
            validation.append(dict(cohort=row['cohort'],dataset=row['dataset'],seed=row['seed'],**comparison))
            write(out/'validation_progress.json',validation)
        for row in rows:
            run=out/row['cohort']/row['dataset']/f"seed{row['seed']}"
            for iteration in range(1,passes+1):
                if digest(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed before test')
                path=run/f'test_repeat{iteration}.npz'
                with validation_execution('cpu'):
                    metrics=evaluate_saved(row['checkpoint'],'ood_test',Path(row['cache']),'cpu','independent',row['seed'],prediction_path=path)
                comparison=compare_predictions(path,row['test_reference'])
                record=dict(cohort=row['cohort'],dataset=row['dataset'],seed=row['seed'],iteration=iteration,
                    expected_auc=row['test_auc'],metrics=metrics,**comparison)
                record['auc_matches_saved']=bool(metrics['auc']==row['test_auc']==comparison['auc']==comparison['reference_auc'])
                tests.append(record);write(out/'test_progress.json',tests)
                print(f"{row['cohort']} {row['dataset']} seed{row['seed']} repeat{iteration}: exact={comparison['exact']}",flush=True)
        report=dict(checkpoints=len(rows),validation_evaluations=len(validation),test_evaluations=len(tests),
            validation=validation,tests=tests,all_test_predictions_exact=all(r['exact'] and r['auc_matches_saved'] for r in tests),
            no_training=True,no_checkpoint_reselection=True,device='cpu',passes_per_checkpoint=passes)
        write(out/'audit_results.json',report)
        if not report['all_test_predictions_exact']:raise ValueError('Fresh test probabilities differ; inspect audit_results before further training')
        write(out/'completed.json',report);return report
    except Exception as error:
        write(out/'failed.json',dict(error=str(error),validation=validation,tests=tests));raise


def verify_inputs(spec,project):
    checked={};source_checked=set()
    def check(name,sha):
        if name not in checked:checked[name]=digest(contained(project,name))
        if checked[name]!=sha:raise ValueError('Frozen replay input changed: '+name)
    for d in spec['datasets']:
        for name,sha in [(d['raw_path'],d['raw_sha256']),
                (d['cache_path']+'/ood_val.pt',d['validation_cache_sha256']),
                (d['cache_path']+'/ood_test.pt',d['test_cache_sha256'])]:check(name,sha)
    for row in spec['rows']:
        for name in ('checkpoint','validation_reference','test_reference','source_manifest'):
            check(row[name],row[name+'_sha256'])
        if row['source_manifest'] not in source_checked:
            source=json.loads(contained(project,row['source_manifest']).read_text())['files']
            for name,sha in source.items():
                if name.startswith('src/models/') or name in ('src/training/pipeline.py','src/data/dataloader.py',
                        'src/data/config_hash.py','src/training/checkpoint.py','src/experiments/checkpoints.py'):
                    if digest(ROOT/name)!=sha:raise ValueError('Model/evaluation implementation changed: '+name)
            source_checked.add(row['source_manifest'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',choices=['cpu'],default='cpu');p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv);path=ROOT/'configs/checkpoint_test_replay_20261103.json';spec=json.loads(path.read_text());validate(spec)
    project=a.project_root.resolve();verify_inputs(spec,project)
    plan=dict(protocol_sha256=digest(path),checkpoint_count=30,validation_evaluations=30,
        test_evaluations=60,training_runs=0,device='cpu',threads=1,batch_size=64,
        no_checkpoint_reselection=True,no_tolerance_waiver=True)
    if not a.execute:print(json.dumps(plan,indent=2));return plan
    torch.set_num_threads(1);rows=[]
    for row in spec['rows']:
        entry=dict(row)
        for key in ('checkpoint','validation_reference','test_reference'):entry[key]=str(contained(project,row[key]))
        entry['cache']=str(contained(project,next(d['cache_path'] for d in spec['datasets'] if d['dataset']==row['dataset'])))
        rows.append(entry)
    result=replay(rows,a.output_dir.resolve());verify_inputs(spec,project)
    write(a.output_dir/'plan.json',plan);return result


if __name__=='__main__':main()
