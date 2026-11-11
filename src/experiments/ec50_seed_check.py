"""Requested EC50 initializations and explicitly test-selected reproducibility reruns."""
import argparse
import json
from pathlib import Path
import torch

from src.experiments.checkpoints import digest
from src.experiments.ec50_replication import run_study,DATASETS
from src.experiments.fragment_tuning import contained
from src.experiments.original_tuning import validate as validate_reference
from src.experiments.plans import ROOT
from src.experiments.sugar_reference_search import write


def top_five(report):
    return {d:[dict(seed=r['seed'],original_test_auc=r['metrics']['auc']) for r in sorted(
        [r for r in report['test_evaluation']['runs'] if r['dataset']==d],
        key=lambda r:(-r['metrics']['auc'],r['seed']))[:5]] for d in DATASETS}


def validate(spec):
    expected={'ec50_assay':[11422,11420,11409,11415,11408],
              'ec50_scaffold':[11413,11403,11409,11414,11401],
              'ec50_size':[11401,11413,11430,11427,11428]}
    if (spec['schema']!=1 or spec['epochs']!=150 or spec['requested_seeds']!=[1,2,3,4,5]
            or spec['training_runs']!=30 or not spec['selected_reruns_are_reproducibility_only']
            or spec['fixed_recipes']!={'ec50_assay':1,'ec50_scaffold':2,'ec50_size':2}
            or {d:[r['seed'] for r in spec['top_by_dataset'][d]] for d in DATASETS}!=expected):
        raise ValueError('Requested seed-check scope changed')


def scopes(spec,smoke,selected_only=False):
    def group(name,datasets,seeds):
        return dict(study=name,datasets=datasets,seeds=seeds[:1] if smoke else seeds,
            epochs=2 if smoke else spec['epochs'],training_runs=len(datasets)*(1 if smoke else len(seeds)),
            feature_mode='static_pool',classifier_feature_mode='raw')
    groups=[group('seeds_1_to_5',DATASETS,spec['requested_seeds'])]+[
        group('selected_'+d,[d],[r['seed'] for r in spec['top_by_dataset'][d]]) for d in DATASETS]
    return groups[1:] if selected_only else groups


def previous_reruns(project,spec):
    config_path=ROOT/'configs/ec50_selected_repeat_20261103.json'
    repeat=json.loads(config_path.read_text())
    if (repeat['schema']!=1 or repeat['training_runs']!=15 or repeat['epochs']!=150
            or not repeat['selected_seed_set_unchanged'] or not repeat['no_seeds_1_to_5']):
        raise ValueError('Selected-only repeat scope changed')
    path=contained(project,repeat['previous_comparison'])
    if digest(path)!=repeat['previous_comparison_sha256']:raise ValueError('First rerun comparison changed')
    rows=json.loads(path.read_text())['runs']
    expected={(d,r['seed']):r['original_test_auc'] for d in DATASETS for r in spec['top_by_dataset'][d]}
    if (len(rows)!=15 or {(r['dataset'],r['seed']):r['original_test_auc'] for r in rows}!=expected):
        raise ValueError('First rerun has a different selected seed set')
    return {(r['dataset'],r['seed']):r['rerun_test_auc'] for r in rows},digest(config_path)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--selected-only',action='store_true',help='Repeat the same 15 selected seeds; exclude seeds 1–5 and compare with first reruns')
    p.add_argument('--execute',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args(argv);spec_path=ROOT/'configs/ec50_seed_check_20261102.json'
    spec=json.loads(spec_path.read_text());validate(spec)
    ref_path=ROOT/spec['reference_protocol']
    if digest(ref_path)!=spec['reference_protocol_sha256']:raise ValueError('Reference recipe protocol changed')
    reference=json.loads(ref_path.read_text());validate_reference(reference);project=a.project_root.resolve()
    if not a.smoke:
        source=contained(project,spec['source_results'])/'completed.json'
        if digest(source)!=spec['source_completed_sha256']:raise ValueError('Frozen 30-seed result source changed')
        report=json.loads(source.read_text())
        if len(report['runs'])!=90 or not report['test_evaluated'] or top_five(report)!=spec['top_by_dataset']:
            raise ValueError('Top-seed selection does not match original results')
        for entry in reference['datasets']:
            if entry['dataset'] in DATASETS and digest(contained(project,entry['raw_path']))!=entry['raw_sha256']:
                raise ValueError('Downloaded source changed')
    previous,repeat_sha=previous_reruns(project,spec) if a.selected_only and not a.smoke else ({},None)
    groups=scopes(spec,a.smoke,a.selected_only)
    plan=dict(protocol=spec,protocol_sha256=digest(spec_path),groups=groups,
        training_runs=sum(g['training_runs'] for g in groups),synthetic=a.smoke,
        independent_initialization_runs=True,views_per_run=3 if not a.smoke else 2,
        selected_only=a.selected_only,repeat_protocol_sha256=repeat_sha,
        test_planned=True,selected_seed_results_not_unbiased_method_evidence=True)
    if not a.execute:print(json.dumps(plan,indent=2));return plan
    torch.set_num_threads(1);out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan)
    reports={}
    try:
        for group in groups:
            reports[group['study']]=run_study(group,spec,reference,project,out/group['study'],a.device,a.smoke)
        deltas=[]
        for dataset in DATASETS:
            results=reports['selected_'+dataset]['test_evaluation']['runs']
            for row in results:
                old=next(r['original_test_auc'] for r in spec['top_by_dataset'][dataset] if r['seed']==row['seed'])
                deltas.append(dict(dataset=dataset,seed=row['seed'],original_test_auc=old,
                    rerun_test_auc=row['metrics']['auc'],difference=float(row['metrics']['auc']-old)))
                if previous:
                    first=previous[(dataset,row['seed'])]
                    deltas[-1].update(first_rerun_test_auc=first,
                        difference_from_first_rerun=float(row['metrics']['auc']-first))
        write(out/'selected_seed_reproducibility.json',dict(runs=deltas,
            interpretation='Seeds chosen using prior test AUC. Descriptive reproducibility only; do not pool with independent-seed performance. Synthetic differences have no scientific interpretation.'))
        completed=dict(plan=plan,groups={k:dict(summary=v['summary']) for k,v in reports.items()},
            all_requested_runs_completed=True,test_evaluated=True)
        write(out/'completed.json',completed);return completed
    except Exception as error:
        write(out/'failed.json',dict(error=str(error),completed_groups=list(reports)));raise


if __name__=='__main__':main()
