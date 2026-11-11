"""Test every frozen contextual-feature checkpoint after global validation gating."""
from copy import deepcopy
import math
from pathlib import Path
import statistics
import numpy as np

from src.experiments.checkpoints import digest, evaluate_saved, require_parity
from src.experiments.sugar_reference_search import write


def summarize(rows):
    summaries = []
    for dataset in sorted({r['dataset'] for r in rows}):
        group = [r for r in rows if r['dataset']==dataset]
        summary = dict(dataset=dataset, conditions={}, paired={})
        for arm in ('scorer_only','scorer_and_classifier'):
            selected = sorted([r for r in group if r['arm']==arm],key=lambda r:r['seed'])
            summary['conditions'][arm] = dict(seeds=[r['seed'] for r in selected],
                **{split:dict(mean=statistics.mean(r['metrics'][split]['auc'] for r in selected),
                    sample_sd=statistics.stdev(r['metrics'][split]['auc'] for r in selected)
                        if len(selected)>1 else None,
                    values=[float(r['metrics'][split]['auc']) for r in selected])
                   for split in ('ood_val','ood_test')})
        a,b=[summary['conditions'][arm] for arm in ('scorer_only','scorer_and_classifier')]
        if a['seeds'] != b['seeds']: raise ValueError('Unpaired test comparison')
        for split in ('ood_val','ood_test'):
            delta=[y-x for x,y in zip(a[split]['values'],b[split]['values'])]
            summary['paired'][split]=dict(values=delta,mean=statistics.mean(delta),
                sample_sd=statistics.stdev(delta) if len(delta)>1 else None,
                positive_seeds=sum(int(x>0) for x in delta))
        summaries.append(summary)
    return summaries


def require_exact_predictions(actual,reference):
    """AUC equality alone cannot establish per-molecule prediction identity."""
    with np.load(actual) as a,np.load(reference) as b:
        for key in ('labels','probabilities'):
            if a[key].dtype!=b[key].dtype or not np.array_equal(a[key],b[key]):
                raise ValueError('CPU validation prediction reference mismatch: '+key)


def evaluate_frozen(rows, cache_by_dataset, out, device,validation_references=None):
    """No training, checkpoint substitution or data modification. Fresh output only."""
    entries=deepcopy(rows)
    if not entries or len({(r['dataset'],r['arm'],r['seed']) for r in entries})!=len(entries):
        raise ValueError('Empty/duplicate frozen entries')
    for r in entries:
        if r['status']!='completed' or not math.isfinite(r['validation']['auc']):
            raise ValueError('Incomplete training or invalid validation metric')
        if digest(r['checkpoint'])!=r['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint hash mismatch')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write(out/'frozen_checkpoints.json',entries)
    test_attempted=False
    try:
        # Validate the entire batch before the first test-loader/model forward.
        for r in entries:
            if digest(r['checkpoint'])!=r['checkpoint_sha256']:
                raise ValueError('Checkpoint changed before validation replay')
            run=out/r['dataset']/r['arm']/('seed'+str(r['seed']));run.mkdir(parents=True)
            val=evaluate_saved(r['checkpoint'],'ood_val',Path(cache_by_dataset[r['dataset']]),
                device,'independent',r['seed'],prediction_path=run/'validation_predictions.npz')
            if validation_references is not None:
                key=(r['dataset'],r['arm'],r['seed'])
                require_exact_predictions(run/'validation_predictions.npz',validation_references[key])
            require_parity(val['auc'],r['validation']['auc'])
            r['metrics']=dict(ood_val=val)
            write(out/'validation_progress.json',entries)
        for r in entries:
            if digest(r['checkpoint'])!=r['checkpoint_sha256']:
                raise ValueError('Checkpoint changed; no test evaluation allowed')
        for r in entries:
            if digest(r['checkpoint'])!=r['checkpoint_sha256']:
                raise ValueError('Checkpoint changed before test evaluation')
            run=out/r['dataset']/r['arm']/('seed'+str(r['seed']))
            test_attempted=True
            test=evaluate_saved(r['checkpoint'],'ood_test',Path(cache_by_dataset[r['dataset']]),
                device,'independent',r['seed'],prediction_path=run/'test_predictions.npz')
            if not math.isfinite(test['auc']): raise ValueError('Nonfinite test metric')
            r['metrics']['ood_test']=test
            write(run/'metrics.json',r['metrics']);write(out/'test_progress.json',entries)
        report=dict(runs=entries,summary=summarize(entries),test_evaluated=True,
            inference_policy='independent',checkpoint_rule='best_raw_pooled_validation_auc',
            no_training=True,no_test_based_model_selection=True,
            verification_policy=('exact_cpu_prediction_reference' if validation_references is not None
                                 else 'scalar_auc_parity'),
            limitation='Previously inspected benchmark families; seed SD is not uncertainty across new datasets.')
        write(out/'completed.json',report);return report
    except Exception as error:
        write(out/'failed.json',dict(error=str(error),runs=entries,
            test_started=test_attempted))
        raise
