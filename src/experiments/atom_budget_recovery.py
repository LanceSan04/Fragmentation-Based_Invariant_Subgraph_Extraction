"""Validate and import a completed reference trajectory without retraining it."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
from src.experiments.assay_followup import evaluate, require_parity
from src.experiments.selection import write
from src.training.checkpoint import load_checkpoint

AUC_TOLERANCE = 1e-4  # 0.01 AUC percentage points, 50x smaller than the study screen.


def parity_report(result):
    selected = result['best']['independent']['auc']
    restored = result['results']['independent']['independent']['ood_val']['auc']
    require_parity(restored, selected, AUC_TOLERANCE)
    return dict(selected_auc=selected, restored_auc=restored,
                absolute_difference=abs(restored-selected), tolerance=AUC_TOLERANCE,
                passed=True)


def check_completed_reference(config, history, result, payload, expected, epochs, seed):
    """Refuse different experiments, incomplete trajectories or wrong snapshots."""
    values = dict(expected, seed=seed, sampler_seed=seed, early_stopping=False,
                  selection_ma_window=1, selection_weight_avg=False,
                  greedy_model_selection=False, loss_log_path=None,
                  test_split=None, report_splits=[], loader_split_names=['train','ood_val'])
    if config != values:
        mismatch = [k for k in config.keys() | values.keys() if config.get(k)!=values.get(k)]
        raise ValueError(f'Reference configuration mismatch: {mismatch}')
    if [r['epoch'] for r in history] != list(range(1,epochs+1)):
        raise ValueError('Reference training trajectory is incomplete')
    if result['seed'] != seed or set(result['results']) != {'independent'}:
        raise ValueError('Reference seed or policies mismatch')
    if set(result['results']['independent']['independent']) != {'ood_val'}:
        raise ValueError('Only validation-only references are allowed')
    scores = [r['validation']['independent']['auc'] for r in history]
    if not all(math.isfinite(v) for v in scores): raise ValueError('Nonfinite validation history')
    best_epoch = int(np.argmax(scores))+1
    if result['best']['independent'] != dict(auc=scores[best_epoch-1], epoch=best_epoch):
        raise ValueError('Reference result does not match validation history')
    if payload['epoch'] != best_epoch or payload['config'] != values:
        raise ValueError('Checkpoint config or selected epoch mismatch')
    parity_report(result)



def require_replay_stability(comparisons):
    """Guard meaningful drift, allowing tiny local hard-selection sensitivity.

    These are engineering tolerances, not evidence of scientific equivalence.
    AUC parity is checked separately for every replay, including CPU.
    """
    for row in comparisons:
        maximum, mean = row['max_probability_difference'], row['mean_probability_difference']
        if not all(math.isfinite(x) for x in (maximum,mean)) or maximum > .001 or mean > .00001:
            raise ValueError('Repeated same-device predictions differ materially; investigate before training')


def recover_reference(source, target, expected, plan, device, replay_records=None):
    """Read-only source; copy artifacts into fresh output after replay checks."""
    source = Path(source).resolve()
    source_plan = json.loads((source/'plan.json').read_text())
    manifest = json.loads((source.parent/'source-manifest.json').read_text())['files']
    current_root = Path(__file__).resolve().parents[2]
    checked_source = {}
    for name, digest in manifest.items():
        if (name.startswith(('src/models/','src/training/','src/data/','src/losses/','src/fragmentation/','src/utils/','configs/'))
                or name in ('train_sugar.py','src/experiments/selection.py','src/experiments/plans.py','src/experiments/common.py')):
            if hashlib.sha256((current_root/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Training source changed: {name}')
            checked_source[name] = digest
    for key in ['seeds','epochs','arms','target_atom_fraction','primary','evaluate_test','data_sha256','raw_sha256']:
        # JSON normalizes the arm tuples to lists.
        wanted = json.loads(json.dumps(plan.get(key)))
        if source_plan.get(key) != wanted:
            raise ValueError(f'Reference study plan mismatch: {key}')
    origin = source/'reference'/'seed8401'
    config = json.loads((origin/'config.json').read_text())
    history = json.loads((origin/'history.json').read_text())
    result = json.loads((origin/'results.json').read_text())
    payload = load_checkpoint(origin/'best_independent.pt')
    check_completed_reference(config,history,result,payload,expected,plan['epochs'],8401)
    diagnostic = target.parent.parent/'recovery_verification'
    diagnostic.mkdir(parents=True,exist_ok=False)
    reports = []
    for i, evaluation_device in enumerate([device,device,device,'cpu']):
        if replay_records:
            report = json.loads((Path(replay_records)/f'validation_{i}_{evaluation_device}.json').read_text())
            wanted_config = dict(config,processed_dir=expected['processed_dir'],device=evaluation_device,
                                 num_workers=0,pin_memory=False,test_split=None,report_splits=[],loader_split_names=['ood_val'])
            if (report['checkpoint_sha256'] != hashlib.sha256((origin/'best_independent.pt').read_bytes()).hexdigest()
                    or report['effective_config'] != wanted_config or report['evaluation_policy']!='independent'):
                raise ValueError('Saved validation replay does not match this checkpoint/configuration')
        else:
            report = evaluate(origin/'best_independent.pt', Path(expected['processed_dir']),
                              evaluation_device, 'ood_val', 8401)
        write(diagnostic/f'validation_{i}_{evaluation_device}.json',report)
        require_parity(report['pooled_auc'],result['best']['independent']['auc'],AUC_TOLERANCE)
        reports.append(report)
    baseline = reports[0]['molecules']
    probabilities = np.array([r['probability'] for r in baseline])
    labels = np.array([r['label'] for r in baseline])
    comparisons = []
    for report in reports:
        rows = report['molecules']
        if [(r['index'],r['label'],r['assay']) for r in rows] != [(r['index'],r['label'],r['assay']) for r in baseline]:
            raise ValueError('Validation examples changed across replay')
        current = np.array([r['probability'] for r in rows])
        before = np.sign(probabilities[labels==1,None]-probabilities[labels==0])
        after = np.sign(current[labels==1,None]-current[labels==0])
        comparisons.append(dict(auc=report['pooled_auc'],mean_probability_difference=float(np.mean(np.abs(current-probabilities))),
                                max_probability_difference=float(np.max(np.abs(current-probabilities))),
                                changed_positive_negative_pair_orderings=int(np.count_nonzero(before!=after))))
    require_replay_stability(comparisons[:3])
    summary = dict(checkpoint=str(origin/'best_independent.pt'), checkpoint_sha256=reports[0]['checkpoint_sha256'],
                   parity=parity_report(result),replays=comparisons,passed=True,unchanged_training_source=checked_source,
                   replay_records_source=str(replay_records) if replay_records else None,
                   same_device_probability_limits=dict(maximum_absolute=.001,mean_absolute=.00001),
                   limitation='Epoch-23 original predictions were not saved; replay cannot prove the exact source of that historical difference.')
    write(diagnostic/'summary.json',summary)
    target.mkdir(parents=True,exist_ok=False)
    hashes = {}
    for name in ['best_independent.pt','config.json','runtime_spec.json','history.json','best.json','results.json']:
        src = origin/name
        hashes[name] = hashlib.sha256(src.read_bytes()).hexdigest()
        shutil.copyfile(src,target/name)
        if hashlib.sha256((target/name).read_bytes()).hexdigest()!=hashes[name]:
            raise ValueError('Recovery copy hash mismatch')
    write(target/'recovered_from.json',dict(source=str(origin),sha256=hashes,training_repeated=False))
    print('REFERENCE RECOVERY VERIFIED: '+json.dumps({k:v for k,v in summary.items() if k!='unchanged_training_source'}),flush=True)
    return result
