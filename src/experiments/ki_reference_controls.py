"""Fixed Ki Scaffold/Size references and ranking controls; no tuning or retries."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from src.experiments.checkpoints import digest, evaluate_saved, require_parity, restore
from src.experiments.cross_domain import prepare_cache, verify_cache
from src.experiments.fragment_confirmation import prediction_report
from src.experiments.fragment_tuning import contained
from src.experiments.plans import ROOT, make_plan
from src.experiments.selection import run_seed, write

ARMS=('fragment_learned','fragment_random','erm','sugar_reference')
DATASETS=('ki_scaffold','ki_size')
SPLITS=('train','ood_val','ood_test')


def validate(spec):
    if (spec['schema']!=1 or spec['datasets']!=list(DATASETS) or spec['arms']!=list(ARMS)
            or spec['seeds']!=list(range(10901,10906)) or spec['pilot_seed']!=10900
            or spec['pilot_epochs']!=3 or spec['fragment_epochs']!=150 or spec['reference_epochs']!=40):
        raise ValueError('Frozen comparison budget changed')
    reference=spec['reference']
    expected=dict(views=10,contrast=32,diversity_penalty=.01,sample_rate=90,
                  contrastive_indexing='released',primary='all_mean')
    if any(reference.get(k)!=v for k,v in expected.items()):
        raise ValueError('Frozen reference recipe changed')
    base=spec['fragment_base']
    expected=dict(num_views=3,view_fragmentations={'group_graph':3},dropout=.5,
        hidden_dim=128,gnn_layers=3,batch_size=64,weight_decay=0.,lambda_diversity=.005,
        lambda_irm=0.,contrast_scope='pooled',contrast_temperature=.1,selection_estimator='hard',
        message_gate_mode='learned',fragment_feature_mode='static_pool',strict=True)
    if any(base.get(k)!=v for k,v in expected.items()):
        raise ValueError('Frozen fragment settings changed')
    if spec['parameters']!={
        'ki_scaffold':dict(learning_rate=.001,edge_ratio=.9,fragment_sample_ratio=.7,lambda_contrast=0),
        'ki_size':dict(learning_rate=.001,edge_ratio=.7,fragment_sample_ratio=.9,lambda_contrast=.1)}:
        raise ValueError('Frozen Ki parameters changed')
    sources=json.loads((ROOT/'configs/ki_bounded_search_20261019.json').read_text())['datasets']
    if spec['data']!={d:sources[d] for d in DATASETS} or spec['retention_samples']!=60:
        raise ValueError('Frozen sources or retention budget changed')


def arm_config(base, arm):
    if arm not in ARMS[:3]:
        raise ValueError('Project model arm required')
    values=deepcopy(base)
    values.update(selection_mode='learned',ranking_mode='random' if arm=='fragment_random' else 'learned',
        erm_baseline=arm=='erm',full_graph_gating=False,selection_estimator='hard',
        message_gate_mode='learned',inference_policy='independent',scorer_context='sampled')
    if arm=='erm':
        values.update(lambda_contrast=0.,lambda_diversity=0.,lambda_irm=0.,erm_contrast_control=False)
    return values


def audit_source(raw, expected_hash):
    """Inspect fixed files without relabeling, filtering, or selecting a recipe."""
    from rdkit import Chem
    if digest(raw)!=expected_hash:
        raise ValueError('Pinned raw source changed: '+str(raw))
    payload=json.loads(Path(raw).read_text())
    records=payload['split'];summary={};members={}
    for split in SPLITS:
        rows=records[split];labels=Counter();members[split]=set();ordered=[]
        assay_labels={};conflicts={}
        for row in rows:
            label=row['cls_label']
            if type(label) is not int or label not in (0,1):
                raise ValueError('Finite integer binary labels required')
            mol=Chem.MolFromSmiles(row['smiles'])
            if mol is None or mol.GetNumBonds()==0:
                raise ValueError('Invalid/edgeless molecule: released top-k cannot silently skip it')
            canonical=Chem.MolToSmiles(mol,isomericSmiles=True)
            members[split].add(canonical);labels[label]+=1
            conflicts.setdefault(canonical,set()).add(label)
            assay=str(row.get('assay_id',''))
            assay_labels.setdefault(assay,Counter())[label]+=1
            ordered.append((row['smiles'],label,str(row.get('domain_id','')),assay))
        if set(labels)!={0,1}:
            raise ValueError('Both classes required in every split')
        summary[split]=dict(molecules=len(rows),positive=labels[1],negative=labels[0],
            positive_fraction=labels[1]/len(rows),canonical_unique=len(members[split]),
            repeated_canonical_rows=len(rows)-len(members[split]),
            conflicting_label_structures=sum(len(v)>1 for v in conflicts.values()),
            assay_groups=len(assay_labels),eligible_assays_ge5_each=sum(v[0]>=5 and v[1]>=5 for v in assay_labels.values()),
            ordered_membership_sha256=hashlib.sha256(json.dumps(ordered,separators=(',',':')).encode()).hexdigest())
    return dict(raw_sha256=expected_hash,splits=summary,source_cfg=payload.get('cfg'),
        threshold_for_cls=payload.get('statistics',{}).get('thr_for_cls'),
        canonical_overlap={a+'__'+b:len(members[a]&members[b])
                           for a,b in [('train','ood_val'),('train','ood_test'),('ood_val','ood_test')]},
        label_handling='Read cls_label unchanged; no new threshold, deduplication, filtering or re-splitting',
        limitation='Internal provenance check, not proof of identity with author-used Ki files. Repeated chemistry across assays is not by itself leakage.')


def verify_inputs(spec, project, cache=True):
    audits={}
    for dataset in DATASETS:
        source=spec['data'][dataset]
        audits[dataset]=audit_source(contained(project,source['raw_path']),source['raw_sha256'])
        if cache:
            verify_cache(base_config(spec,project,dataset,'cpu',150))
    return audits


def base_config(spec,project,dataset,device,epochs):
    values=deepcopy(spec['fragment_base']);values.update(spec['parameters'][dataset])
    source=spec['data'][dataset]
    values.update(dataset='drugood_lbap_core_'+dataset,raw_path=str(contained(project,source['raw_path'])),
        processed_dir=str(contained(project,source['cache_path'])),device=device,epochs=epochs,
        early_stopping=False,compare_checkpoint_rules=False,report_within_assay=False,
        loader_split_names=['train','ood_val'],test_split=None,report_splits=[],
        log_test_curve=False,selection_ma_window=1,selection_weight_avg=False)
    return values


def ensure_identities(entries, spec, project, synthetic):
    """Check the entire frozen batch before the first test forward, and again per run."""
    for row in entries:
        if digest(row['checkpoint'])!=row['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed; test blocked')
    if not synthetic:
        for dataset in DATASETS:
            source=spec['data'][dataset]
            if digest(contained(project,source['raw_path']))!=source['raw_sha256']:
                raise ValueError('Frozen raw source changed; test blocked')


def project_retention(entry,count):
    if entry['arm']=='erm':
        return dict(atom_fraction=1.,bond_fraction=1.,scope='Full-graph model by construction; no rationale/scorer interpretation')
    from src.analysis.retention import measure
    from src.data.dataloader import load_processed_split, _expected_metadata
    config,views=restore(entry['checkpoint'],entry['cache'],'cpu')
    data=load_processed_split(Path(entry['cache'])/'ood_val.pt',_expected_metadata(config))['molecule_data']
    report=measure(views,config,data,edge_ratio=config.edge_ratio,count=count)
    write(Path(entry['run_dir'])/'validation_retention.json',report)
    return dict(summary=report['summary'],overlap=report['overlap'],indices=report['indices'],
                sample_rule=report['sample_rule'],scope='Validation-only evaluation graphs')


def execute(command, log):
    """Subprocess isolation avoids upstream top-level module name collisions."""
    print('Running '+str(log),flush=True)
    with Path(log).open('x') as stream:
        completed=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
    if completed.returncode:
        raise RuntimeError('Stage failed; no retry. Inspect '+str(log))


def reference_preflight(interpreter,device):
    """Check the actual reference interpreter and imports before any training."""
    code=r'''
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
sys.path.insert(0,str(root/'baselines/sugar/adapters'))
sys.path.insert(0,str(root/'baselines/sugar/upstream'))
import torch,torch_geometric,ogb,dgl,rdkit,sklearn
from compat import verify_sources,feature_module,install_masks
from models import ciga
from ogb.graphproppred import Evaluator
verify_sources();install_masks(ciga)
graph=feature_module().smile2graph('CCO')
assert graph.ndata['x'].shape[1]==39 and graph.edata['x'].shape[1]==10
assert sys.argv[2]!='cuda' or torch.cuda.is_available(), 'Reference CUDA unavailable'
print(json.dumps(dict(python=sys.executable,prefix=sys.prefix,torch=torch.__version__,
    pyg=torch_geometric.__version__,ogb=ogb.__version__,dgl=dgl.__version__,
    rdkit=rdkit.__version__,sklearn=sklearn.__version__,
    feature_dimensions=[39,10],device_requested=sys.argv[2],training=False)))
'''
    completed=subprocess.run([str(interpreter),'-B','-c',code,str(ROOT),device],
        cwd=ROOT,capture_output=True,text=True,check=False)
    if completed.returncode:
        raise RuntimeError('Reference dependency preflight failed before training: '+completed.stderr[-4000:])
    report=json.loads(completed.stdout.strip().splitlines()[-1])
    report['warnings']=completed.stderr[-4000:]
    return report


def reference_command(raw,run,device,seed,epochs,spec,smoke,interpreter=None):
    reference=spec['reference']
    command=[str(interpreter or sys.executable),'-u',str(ROOT/'baselines/sugar/adapters/run.py'),
        '--dataset','drugood_'+raw.stem,'--raw-path',str(raw),'--output-dir',str(run/'train'),
        '--device',device,'--seed',str(seed),'--epochs',str(epochs),
        '--views',str(2 if smoke else reference['views']),
        '--contrast',str(reference['contrast']),'--diversity-penalty',str(reference['diversity_penalty']),
        '--sample-rate',str(reference['sample_rate']),'--contrastive-indexing',reference['contrastive_indexing']]
    return command+(['--smoke'] if smoke else [])


def reference_validation(entry,device,count):
    run=Path(entry['run_dir'])
    command=[entry.get('reference_python',sys.executable),'-u',str(ROOT/'baselines/sugar/adapters/evaluate_reference.py'),
        '--checkpoint',entry['checkpoint'],'--raw-path',entry['raw_path'],
        '--output-dir',str(run/'validation'),'--device',device,'--retention-samples',str(count)]
    execute(command,run/'validation.log')
    path=run/'validation/completed.json';validation=json.loads(path.read_text())
    if validation['test_evaluated'] or validation['checkpoint_sha256']!=entry['checkpoint_sha256']:
        raise ValueError('Invalid reference validation record')
    protocol=dict(primary='all_mean',checkpoint_sha256=entry['checkpoint_sha256'],
        validation_sha256=digest(path),raw_sha256=validation['provenance']['source'].get('raw_sha256'),
        variants={'all_mean':validation['variants']['all_mean']['views']})
    write(run/'frozen_test_protocol.json',protocol)
    entry['metrics']={'ood_val':dict(auc=validation['variants']['all_mean']['auc'])}
    entry['retention']=validation['validation_retention']


def reference_predictions(entry,split,synthetic):
    """Normalize saved reference predictions for identical diagnostic definitions."""
    run=Path(entry['run_dir'])
    stage='validation' if split=='ood_val' else 'test'
    with np.load(run/stage/'all_mean.npz',allow_pickle=False) as saved:
        labels=saved['y_true'];probabilities=saved['y_pred']
    if synthetic:
        assays=['']*len(labels)
    else:
        raw=json.loads(Path(entry['raw_path']).read_text())['split'][split]
        if not np.array_equal(labels,np.asarray([r['cls_label'] for r in raw])):
            raise ValueError('Reference prediction membership/labels differ from source')
        assays=[str(r.get('assay_id','')) for r in raw]
    path=run/('validation_predictions.npz' if split=='ood_val' else 'test_predictions.npz')
    np.savez_compressed(path,labels=labels,probabilities=probabilities,
                        assays=np.asarray(assays),row_index=np.arange(len(labels)))
    entry['metrics'][split].update(prediction_report(path))


def evaluate_batch(entries,out,spec,project,device,smoke=False,test=False):
    seeds=[17] if smoke else spec['seeds'] if test else [spec['pilot_seed']]
    identities=[(r['dataset'],r['arm'],r['seed']) for r in entries]
    expected={(d,arm,seed) for d in DATASETS for arm in ARMS for seed in seeds}
    if len(identities)!=len(expected) or set(identities)!=expected:
        raise ValueError('Incomplete or duplicate frozen training batch; test blocked')
    ensure_identities(entries,spec,project,smoke)
    write(out/'frozen_checkpoints.json',entries)
    for row in entries:
        run=Path(row['run_dir'])
        if row['arm']=='sugar_reference':
            reference_validation(row,device,3 if smoke else spec['retention_samples'])
            reference_predictions(row,'ood_val',smoke)
        else:
            predictions=run/'validation_predictions.npz'
            metric=evaluate_saved(row['checkpoint'],'ood_val',row['cache'],device,
                                  'independent',row['seed'],prediction_path=predictions)
            require_parity(metric['auc'],row['selected_auc'])
            metric.update(prediction_report(predictions));row['metrics']={'ood_val':metric}
            row['retention']=project_retention(row,3 if smoke else spec['retention_samples'])
        write(out/'validation_progress.json',entries)
    # No test is scored until every arm/seed/split has passed validation replay.
    ensure_identities(entries,spec,project,smoke)
    write(out/'validation_passed.json',entries)
    if not test:
        return
    for row in entries:
        ensure_identities(entries,spec,project,smoke)
        run=Path(row['run_dir'])
        if row['arm']=='sugar_reference':
            command=[row.get('reference_python',sys.executable),'-u',str(ROOT/'baselines/sugar/adapters/evaluate_reference.py'),
                '--checkpoint',row['checkpoint'],'--raw-path',row['raw_path'],
                '--output-dir',str(run/'test'),'--device',device,
                '--test-protocol',str(run/'frozen_test_protocol.json'),
                '--frozen-validation',str(run/'validation/completed.json')]
            execute(command,run/'test.log')
            result=json.loads((run/'test/completed.json').read_text())
            if not result['test_evaluated'] or set(result['variants'])!={'all_mean'}:
                raise ValueError('Unexpected reference test scope')
            row['metrics']['ood_test']=dict(auc=result['variants']['all_mean']['auc'])
            reference_predictions(row,'ood_test',smoke)
        else:
            predictions=run/'test_predictions.npz'
            metric=evaluate_saved(row['checkpoint'],'ood_test',row['cache'],device,
                'independent',row['seed'],prediction_path=predictions)
            metric.update(prediction_report(predictions));row['metrics']['ood_test']=metric
        write(out/'test_progress.json',entries)


def summarize(entries,test):
    def stats(values):
        values=[float(value) for value in values]
        if not values or not all(math.isfinite(v) for v in values):
            raise ValueError('Finite complete results required')
        return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values) if len(values)>1 else None,
                    n=len(values),values=values)
    report={}
    for dataset in DATASETS:
        rows=[r for r in entries if r['dataset']==dataset]
        report[dataset]={}
        for split in ['ood_val']+(['ood_test'] if test else []):
            scores={arm:[r['metrics'][split]['auc'] for r in rows if r['arm']==arm] for arm in ARMS}
            paired=[]
            for seed in sorted({r['seed'] for r in rows}):
                values={r['arm']:r['metrics'][split]['auc'] for r in rows if r['seed']==seed}
                if set(values)!=set(ARMS):raise ValueError('Incomplete paired results')
                paired.append(float(values['fragment_learned']-values['fragment_random']))
            report[dataset][split]=dict(arms={arm:stats(values) for arm,values in scores.items()},
                learned_minus_random=dict(stats(paired),positive_seeds=sum(v>0 for v in paired)))
    return report


def downloaded_reference(args, spec, project, out, interpreter):
    """Repeat the frozen SuGAr transfer recipe once per official Ki split."""
    expected = {
        "ki_assay": "42923fa7e92530ab949eaad1ddb26e47857dd06006cc30473ccda6d60c890059",
        "ki_scaffold": "dcc4e300bec0ad1ef6b6d50b997d07b2a40c6971b4e1e12099e8abac751bfad9",
        "ki_size": "446779027ee8814aba7ebded369d565bc9e85a057e657ec2c1be1bae86f7ffa9",
    }
    seed, epochs = spec["seeds"][0], 100
    plan = dict(datasets=list(expected), seed=seed, epochs=epochs, training_runs=3,
                reference_recipe=spec["reference"], raw_sha256=expected, test_planned=True)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return plan
    raw_dir = Path("data/raw/drugood_official_chembl30_20261028")
    raw_files = {d: contained(project, raw_dir / f"lbap_core_{d}.json") for d in expected}
    for dataset, raw in raw_files.items():
        if digest(raw) != expected[dataset]:
            raise ValueError(f"Downloaded Ki source changed: {dataset}")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    write(out / "reference_environment.json", reference_preflight(interpreter, args.device))
    entries = []
    for dataset, raw in raw_files.items():
        run = out / dataset / f"seed{seed}"
        run.mkdir(parents=True)
        print(f"Training {dataset}, seed {seed}", flush=True)
        execute(reference_command(raw, run, args.device, seed, epochs, spec, False, interpreter),
                run / "training.log")
        checkpoint = run / "train/best_views.pt"
        row = dict(dataset=dataset, seed=seed, arm="sugar_reference", run_dir=str(run),
                   raw_path=str(raw), checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                   reference_python=interpreter,
                   resources=json.loads((run / "train/completed.json").read_text()))
        reference_validation(row, args.device, spec["retention_samples"])
        reference_predictions(row, "ood_val", False)
        entries.append(row)
        write(out / "validation_progress.json", entries)
    for row in entries:
        if digest(row["checkpoint"]) != row["checkpoint_sha256"] or digest(row["raw_path"]) != expected[row["dataset"]]:
            raise ValueError("Frozen checkpoint or data changed before test")
        run = Path(row["run_dir"])
        execute([interpreter, "-u", str(ROOT / "baselines/sugar/adapters/evaluate_reference.py"),
                 "--checkpoint", row["checkpoint"], "--raw-path", row["raw_path"],
                 "--output-dir", str(run / "test"), "--device", args.device,
                 "--test-protocol", str(run / "frozen_test_protocol.json"),
                 "--frozen-validation", str(run / "validation/completed.json")], run / "test.log")
        result = json.loads((run / "test/completed.json").read_text())
        if not result["test_evaluated"] or set(result["variants"]) != {"all_mean"}:
            raise ValueError("Unexpected reference test scope")
        row["metrics"]["ood_test"] = dict(auc=result["variants"]["all_mean"]["auc"])
        reference_predictions(row, "ood_test", False)
        write(out / "test_progress.json", entries)
    report = dict(plan=plan, runs=entries, test_evaluated=True)
    write(out / "completed.json", report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/ki_reference_controls_20261024.json'))
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--stage',choices=['audit','pilot','full','downloaded-reference'],default='pilot')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--reference-python',type=Path,help='Existing SuGAr environment interpreter; required for real training')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(argv);spec=json.loads(Path(args.config).read_text());validate(spec)
    if args.smoke and args.stage=='audit':parser.error('Audit inspects real files only')
    if args.execute and args.stage!='audit' and not args.smoke and args.reference_python is None:
        parser.error('Real training requires explicit --reference-python; no dependencies are installed automatically')
    # Do not resolve the executable symlink: that bypasses a venv's interpreter path.
    reference_python=str(args.reference_python.expanduser().absolute()) if args.reference_python else sys.executable
    project=args.project_root.resolve();out=args.output_dir.resolve()
    if args.stage=='downloaded-reference':
        if args.smoke:parser.error('Downloaded reference uses real splits')
        return downloaded_reference(args,spec,project,out,reference_python)
    seeds=[17] if args.smoke else [spec['pilot_seed']] if args.stage=='pilot' else spec['seeds']
    fragment_epochs=2 if args.smoke else spec['pilot_epochs'] if args.stage=='pilot' else spec['fragment_epochs']
    reference_epochs=2 if args.smoke else spec['pilot_epochs'] if args.stage=='pilot' else spec['reference_epochs']
    test=args.stage=='full'
    plan=dict(study=spec['study'],stage=args.stage,config_sha256=digest(args.config),
        datasets=list(DATASETS),arms=list(ARMS),seeds=seeds,
        training_runs=0 if args.stage=='audit' else len(DATASETS)*len(ARMS)*len(seeds),
        fragment_epochs=fragment_epochs,reference_epochs=reference_epochs,
        test_evaluation_planned=test,synthetic=args.smoke,search=False,retries=0,automatic_continuation=False,
        reference_recipe=spec['reference'],
        project_python=sys.executable,reference_python=reference_python,
        primary_comparison='fragment_learned minus fragment_random; both learned message gates',
        checkpoint_rules={'project':'best raw pooled validation ensemble AUC; independent inference',
                          'reference':'each view best validation AUC; shared all_mean uniform voting'},
        caveats=['SuGAr recipe transferred from EC50; not a Ki-tuned paper reproduction',
                 'Reference uses 10 small views/40 epochs; project arms 3 larger views/150 epochs',
                 'ERM is prediction-only; fragment arms retain auxiliary losses',
                 'Same edge ratios do not impose equal retained atom or bond budgets',
                 'Previously inspected benchmark test sets; no automatic superiority claim',
                 'Assay-cluster bootstrap is conditional on fitted models; seed SD is a different uncertainty'],
        decision='Report all outcomes and paired differences; no test-driven recipe selection or automatic follow-up')
    if not args.execute:
        print(json.dumps(plan,indent=2));return plan
    out.mkdir(parents=True,exist_ok=False);write(out/'plan.json',plan)
    started=time.perf_counter();entries=[]
    try:
        if args.stage!='audit':
            write(out/'status.json',dict(stage='reference_dependency_preflight',trained=0))
            write(out/'reference_environment.json',reference_preflight(reference_python,args.device))
        audits=verify_inputs(spec,project,cache=args.stage!='audit') if not args.smoke else {'synthetic':True}
        write(out/'data_audit.json',audits)
        if args.stage=='audit':return audits
        torch.set_num_threads(1)
        bases={}
        for dataset in DATASETS:
            base=base_config(spec,project,dataset,args.device,fragment_epochs)
            if args.smoke:
                _,fixture=make_plan(out/'fixtures'/dataset,args.device,True)
                base.update(fixture)
                base.update(num_views=2,view_fragmentations={'group_graph':2},epochs=2,
                            fragment_feature_mode='static_pool',metadata_keys=[])
                prepare_cache(base)
            bases[dataset]=base
        for dataset in DATASETS:
            raw=contained(project,spec['data'][dataset]['raw_path'])
            for seed in seeds:
                for arm in ARMS:
                    if not args.smoke and shutil.disk_usage(out).free<20*1024**3:
                        raise RuntimeError('Less than 20 GiB free; no new run started')
                    run=out/dataset/arm/f'seed{seed}';run.parent.mkdir(parents=True,exist_ok=True)
                    write(out/'status.json',dict(stage='training',dataset=dataset,seed=seed,arm=arm,trained=len(entries)))
                    print(f'{dataset} {arm} seed {seed}',flush=True)
                    entry=dict(dataset=dataset,arm=arm,seed=seed,run_dir=str(run),raw_path=str(raw))
                    if arm=='sugar_reference':
                        run.mkdir()
                        entry['reference_python']=reference_python
                        command=reference_command(raw,run,args.device,seed,reference_epochs,spec,args.smoke,reference_python)
                        execute(command,run/'training.log')
                        checkpoint=run/'train/best_views.pt'
                        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                        entry.update(selected_epochs=state['selected_epochs'],
                            resources=json.loads((run/'train/completed.json').read_text()))
                    else:
                        values=arm_config(bases[dataset],arm)
                        result=run_seed(values,run,seed,False,('independent',))
                        best=result['best']['independent'];checkpoint=run/'best_independent.pt'
                        entry.update(selected_auc=best['auc'],selected_epoch=best['epoch'],
                                     cache=values['processed_dir'],resources=result['resources'])
                    entry.update(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint))
                    entries.append(entry);write(out/'training_progress.json',entries)
        write(out/'status.json',dict(stage='validation',trained=len(entries)))
        evaluate_batch(entries,out,spec,project,args.device,args.smoke,test)
        report=dict(plan=plan,test_evaluated=test,results=summarize(entries,test),runs=entries,
                    wall_seconds=time.perf_counter()-started)
        if test:
            from src.evaluation.paired_uncertainty import cluster_difference
            report['primary_assay_cluster_uncertainty']={dataset:cluster_difference(
                [out/dataset/'fragment_learned'/f'seed{s}'/'test_predictions.npz' for s in seeds],
                [out/dataset/'fragment_random'/f'seed{s}'/'test_predictions.npz' for s in seeds])
                for dataset in DATASETS}
        write(out/'completed.json',report);write(out/'status.json',dict(stage='completed'))
        return report
    except Exception as error:
        write(out/'status.json',dict(stage='failed',trained=len(entries),error=str(error),retry=False))
        raise
