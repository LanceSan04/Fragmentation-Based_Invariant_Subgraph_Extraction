"""Official EC50 frozen tests with an explicitly amended CPU prediction reference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.experiments.checkpoints import digest
from src.experiments.context_test import evaluate_frozen
from src.experiments.fragment_tuning import contained
from src.experiments.plans import ROOT
from src.experiments.sugar_reference_search import write


def validate(spec):
    expected={(d,a,s) for d in ('ec50_assay','ec50_scaffold','ec50_size')
              for a in ('scorer_only','scorer_and_classifier') for s in range(11211,11216)}
    if (spec['schema']!=1 or spec['torch_threads']!=1 or spec['batch_size']!=64
            or spec['verification_policy']!='exact_cpu_prediction_reference'
            or len(spec['rows'])!=30
            or {(r['dataset'],r['arm'],r['seed']) for r in spec['rows']}!=expected):
        raise ValueError('Frozen CPU test scope changed')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/cpu_frozen_test_20261030.json'))
    p.add_argument('--project-root',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--device',choices=['cpu'],default='cpu')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv);spec=json.loads(Path(a.config).read_text());validate(spec)
    plan=dict(protocol=spec,config_sha256=digest(a.config),training_runs=0,
        validation_evaluations=30,test_evaluations=30,checkpoint_reselection=False,
        numeric_tolerance_increased=False,evaluation_policy_amended=True,
        reference='Audited deterministic CPU predictions; original GPU selection records retained')
    if not a.execute:print(json.dumps(plan,indent=2));return plan
    project=a.project_root.resolve();audit_dir=contained(project,spec['audit_record'])
    audit_file=audit_dir/'results/completed.json'
    if digest(audit_file)!=spec['audit_completed_sha256']:raise ValueError('CPU audit record changed')
    audit=json.loads(audit_file.read_text());launch=json.loads((audit_dir/'launch.json').read_text())
    if (launch['status']!='succeeded' or launch['manifest_sha256']!=spec['audit_manifest_sha256']
            or not audit['all_repeats_bit_identical'] or not audit['all_states_identical']):
        raise ValueError('CPU audit is not the frozen successful reference')
    source=json.loads((audit_dir/'source-manifest.json').read_text())['files']
    for name,sha in source.items():
        if name.startswith('src/models/') or name in ('src/training/pipeline.py',
                'src/data/dataloader.py','src/training/checkpoint.py','src/data/config_hash.py',
                'src/experiments/checkpoints.py','src/experiments/common.py'):
            if digest(ROOT/name)!=sha:raise ValueError('Model/evaluation data source changed: '+name)
    checked={};rows=[];caches={};references={}
    for row in spec['rows']:
        audit_row=next(x for x in audit['runs'] if (x['dataset'],x['arm'],x['seed'])==
                       (row['dataset'],row['arm'],row['seed']))
        if (audit_row['checkpoint_sha256']!=row['checkpoint_sha256']
                or audit_row['selected_epoch']!=row['checkpoint_epoch']
                or audit_row['passes'][0]['auc']!=row['cpu_reference_auc']):
            raise ValueError('Audit/checkpoint/reference identity changed')
        inputs=[(row['checkpoint'],row['checkpoint_sha256']),
                (row['cache']+'/ood_val.pt',row['validation_cache_sha256']),
                (row['cache']+'/ood_test.pt',row['test_cache_sha256']),
                (row['raw_path'],row['raw_sha256']),
                (row['reference_predictions'],row['reference_predictions_sha256']),
                (row['cpu_reference'],row['cpu_reference_sha256']),
                (row['cpu_repeat'],row['cpu_repeat_sha256'])]
        for name,sha in inputs:
            if name not in checked:checked[name]=digest(contained(project,name))
            if checked[name]!=sha:raise ValueError('Frozen input changed: '+name)
        reference=contained(project,row['cpu_reference'])
        with np.load(reference) as x,np.load(contained(project,row['cpu_repeat'])) as y:
            if set(x.files)!=set(y.files) or any(not np.array_equal(x[k],y[k]) for k in x.files):
                raise ValueError('Audited CPU repeats no longer agree')
            if roc_auc_score(x['labels'],x['probabilities'])!=row['cpu_reference_auc']:
                raise ValueError('CPU reference AUC does not match its frozen predictions')
        key=(row['dataset'],row['arm'],row['seed']);references[key]=reference
        caches[row['dataset']]=str(contained(project,row['cache']))
        rows.append(dict(dataset=row['dataset'],arm=row['arm'],seed=row['seed'],status='completed',
            checkpoint=str(contained(project,row['checkpoint'])),checkpoint_sha256=row['checkpoint_sha256'],
            selected_epoch=row['checkpoint_epoch'],original_gpu_selection_auc=row['epoch_auc'],
            cpu_reference_auc=row['cpu_reference_auc'],
            cpu_minus_original_gpu_auc=row['cpu_reference_auc']-row['epoch_auc'],
            validation=dict(auc=row['cpu_reference_auc'])))
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'plan.json',plan);write(out/'evaluation_policy_amendment.json',dict(
        policy='Exact deterministic CPU prediction reference for all30 checkpoints',
        original_gpu_selection_preserved=True,selected_checkpoints_unchanged=True,
        old_gpu_differences=[dict(dataset=r['dataset'],arm=r['arm'],seed=r['seed'],
            difference=r['cpu_minus_original_gpu_auc']) for r in rows],
        rationale='Three fixed GPU reloads varied; two CPU passes per checkpoint were bit-identical.'))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    try:
        report=evaluate_frozen(rows,caches,out/'evaluation','cpu',validation_references=references)
        report.update(plan=plan,original_gpu_selection_records_retained=True,
            cpu_validation_reference_used_uniformly=True,checkpoint_reselection=False)
        write(out/'completed.json',report);return report
    except Exception as error:
        write(out/'failed.json',dict(error=str(error),test_status='See evaluation/failed.json'));raise


if __name__=='__main__':main()
