"""Freeze contrastive references, diagnose retention, then evaluate fixed checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import torch

from scripts.audit_validation_groups import group_report
from src.data.dataloader import build_split_dataloaders, load_processed_split
from src.experiments.common import set_all_seeds
from src.experiments.retention_audit import measure
from src.experiments.selection import verify_study_cache, write
from src.models.sugar_ensemble import SuGAREnsemble
from src.training.checkpoint import load_checkpoint, load_view_state_dicts
from train_sugar import Config, build_views_from_feature_dims, strip_non_tensor_batch_fields


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def statistics_of(values):
    return dict(n=len(values), mean=statistics.mean(values),
                sample_std=statistics.stdev(values) if len(values)>1 else None)


def require_parity(observed, expected, tolerance=1e-4):
    if not all(__import__('math').isfinite(v) for v in [observed, expected]) or abs(observed-expected)>tolerance:
        raise ValueError(f'Validation parity failed: {observed} versus {expected}; no test evaluation allowed')


def restore(path, cache, device):
    payload = load_checkpoint(path, map_location='cpu')
    config = Config(dict(payload['config'], processed_dir=str(cache), device=device, num_workers=0,
                         pin_memory=False, test_split=None, report_splits=[]))
    views = build_views_from_feature_dims(config, payload['feature_dims'])
    load_view_state_dicts(views, payload)
    for view in views:
        view.to(device).eval()
    return config, views


def evaluate(path, cache, device, split, seed, synthetic=False):
    set_all_seeds(seed)
    config, views = restore(path, cache, device)
    config = Config(dict(config.to_dict(), loader_split_names=[split]))
    loaders = build_split_dataloaders(cache, config)
    assays = []
    for d in loaders[split].dataset:
        if 'assay_id' not in d.metadata and not synthetic:
            raise ValueError('Missing assay metadata')
        assays.append(str(d.metadata.get('assay_id', d.metadata.get('domain_id', 'synthetic'))))
    strip_non_tensor_batch_fields(loaders)
    model = SuGAREnsemble(views, aggregation_mode=config.aggregation_mode, inference_policy='independent')
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loaders[split]:
            batch = batch.to(device)
            probabilities = model(batch).exp()[:, 1].cpu().tolist()
            for label, probability in zip(batch.y.reshape(-1).cpu().tolist(), probabilities):
                rows.append(dict(index=len(rows), assay=assays[len(rows)], label=int(label), probability=probability))
    return dict(group_report(rows), molecules=rows, effective_config=config.to_dict(),
                evaluation_policy='independent', checkpoint=str(path), checkpoint_sha256=digest(path))


def freeze(roots, seeds, cache, raw, smoke):
    verify_study_cache(cache, raw)
    records = []
    configs = {}
    source_manifests = {}
    for arm, root in roots.items():
        runs = json.loads((root/'summary.json').read_text())['runs']
        if sorted(r['seed'] for r in runs) != seeds:
            raise ValueError(f'Unexpected seed set for {arm}')
        if not smoke:
            launch_root = root.parent.parent
            launch = json.loads((launch_root/'launch.json').read_text())
            if launch.get('status') != 'succeeded' or launch.get('exit_code') != 0:
                raise ValueError(f'Incomplete source run: {arm}')
            manifest = launch_root/'source-manifest.json'
            source_manifests[arm] = dict(path=str(manifest), sha256=digest(manifest))
            files = json.loads(manifest.read_text())['files']
            for name, expected in files.items():
                if name.startswith(('src/models/', 'src/data/')):
                    if digest(Path(__file__).resolve().parents[2]/name) != expected:
                        raise ValueError(f'Inference source changed: {name}')
        for run in runs:
            seed = run['seed']
            if 'ood_test' in json.dumps(run):
                raise ValueError('Source summary already contains test evaluation')
            cfg = json.loads((root/f'seed{seed}'/'config.json').read_text())
            if not smoke:
                if cfg['seed'] != seed or cfg['sampler_seed'] != seed:
                    raise ValueError('Training/sampler seed mismatch')
                if Path(cfg['processed_dir']).resolve() != cache or Path(cfg['raw_path']).resolve() != raw:
                    raise ValueError('Data source mismatch')
                history = json.loads((root/f'seed{seed}'/'history.json').read_text())
                if len(history) != 100 or history[-1]['epoch'] != 100:
                    raise ValueError('Incomplete training history')
            configs[(arm,seed)] = cfg
            for policy in ['independent','shared']:
                checkpoint = root/f'seed{seed}'/f'best_{policy}.pt'
                payload = load_checkpoint(checkpoint, map_location='cpu')
                if payload['epoch'] != run['best'][policy]['epoch'] or payload['config']['inference_policy'] != policy:
                    raise ValueError('Checkpoint does not match selection record')
                records.append(dict(arm=arm, seed=seed, selected_by=policy, checkpoint=str(checkpoint),
                                    checkpoint_sha256=digest(checkpoint), epoch=payload['epoch'],
                                    expected_validation_auc=run['results'][policy]['independent']['ood_val']['auc'],
                                    checkpoint_config=payload['config']))
    if not smoke:
        for seed in seeds:
            base = configs[('pooled',seed)]
            if base['scoring_mode']!='ordered' or base['lambda_contrast']!=.1:
                raise ValueError('Wrong pooled reference')
            for arm in ['off','per_view']:
                cfg = configs[(arm,seed)]
                allowed = {'lambda_contrast','contrast_scope','log_gradient_influence'}
                if any(base.get(k)!=cfg.get(k) for k in base.keys()|cfg.keys() if k not in allowed):
                    raise ValueError('Unmatched training configuration')
                if cfg['lambda_contrast'] != (0 if arm=='off' else .1):
                    raise ValueError('Wrong contrastive coefficient')
                if cfg.get('contrast_scope') != ('pooled' if arm=='off' else 'per_view'):
                    raise ValueError('Wrong contrastive scope')
    return records, source_manifests


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['pooled-dir','off-dir','per-view-dir','processed-dir','output-dir']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--raw-path')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(argv)
    torch.set_num_threads(4)
    out, cache = Path(a.output_dir).resolve(), Path(a.processed_dir).resolve()
    if out.exists(): raise FileExistsError(out)
    if not a.smoke and not a.raw_path: p.error('Real evaluation requires raw-path')
    raw = Path(a.raw_path).resolve() if a.raw_path else None
    roots = dict(pooled=Path(a.pooled_dir).resolve(), off=Path(a.off_dir).resolve(), per_view=Path(a.per_view_dir).resolve())
    seeds = [5001] if a.smoke else list(range(8301,8306))
    records, sources = freeze(roots, seeds, cache, raw, a.smoke)
    out.mkdir(parents=True)
    plan = dict(created_at=now(), synthetic=a.smoke, reference='pooled', seeds=seeds,
                records=records, source_manifests=sources, training_runs=0,
                primary='independent/independent', secondary='shared/independent',
                test_plan='Evaluate all three fixed arms, both saved selections, independent inference. No ratio changes or test-based selection.',
                caveat='Assay test has been inspected previously; this remains exploratory, not an untouched holdout.',
                cache=str(cache), data_sha256={s:digest(cache/f'{s}.pt') for s in ['ood_val','ood_test']},
                raw_sha256=digest(raw) if raw else None,
                ratio_grid=dict(edge=[.3,.5,.7],sample=[.5,.7,.9],draws=2,
                                evaluation_molecules=100,sampling_probe_molecules=40))
    write(out/'frozen_plan.json', plan)
    validation = []
    for record in records:
        report = evaluate(record['checkpoint'], cache, a.device, 'ood_val', record['seed'], a.smoke)
        require_parity(report['pooled_auc'], record['expected_validation_auc'])
        name = f"{record['arm']}_{record['seed']}_{record['selected_by']}"
        write(out/f'validation_{name}.json', report)
        validation.append(dict(record, auc=report['pooled_auc'], macro=report['macro_assay_auc'], within=report['within_pair_weighted_auc']))
        print('Validation parity passed:',name,report['pooled_auc'],flush=True)
    write(out/'validation_summary.json', validation)
    data = load_processed_split(cache/'ood_val.pt')['molecule_data']
    for record in records:
        if record['selected_by']!='independent': continue
        cfg, views = restore(record['checkpoint'],cache,'cpu')
        name = f"{record['arm']}_{record['seed']}"
        directory = out/'retention'/name
        directory.mkdir(parents=True)
        ratios = [.5] if record['arm']!='pooled' else [.3,.5,.7]
        for ratio in ratios:
            report = measure(views,cfg,data,ratio,count=4 if a.smoke else 100)
            write(directory/f'eval_edge{ratio}.json', report)
        if record['arm']=='pooled':
            for ratio in [.3,.5,.7]:
                for sample in [.5,.7,.9]:
                    for draw in range(1 if a.smoke else 2):
                        report=measure(views,cfg,data,ratio,sample,count=4 if a.smoke else 40,draw=draw)
                        write(directory/f'sample_edge{ratio}_keep{sample}_draw{draw}.json',report)
        print('Retention diagnostics complete:',name,flush=True)
    # The decision is made only from validation; ratio probes never nominate a test variant.
    means = {arm:statistics.mean(r['auc'] for r in validation if r['arm']==arm and r['selected_by']=='independent') for arm in roots}
    decision = dict(frozen_at=now(), primary_reference='pooled', validation_mean_auc=means,
                    highest_validation_mean=max(means,key=means.get),
                    action='Retain pooled reference; test all predeclared arms without changing weights, ratios or selection.',
                    ratio_action='Descriptive sensitivity only; no retraining or ratio promotion in this batch.')
    write(out/'decision_before_test.json',decision)
    test_rows=[]
    for record in records:
        if digest(record['checkpoint'])!=record['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after freeze')
        report=evaluate(record['checkpoint'],cache,a.device,'ood_test',record['seed'],a.smoke)
        name=f"{record['arm']}_{record['seed']}_{record['selected_by']}"
        write(out/f'test_{name}.json',report)
        test_rows.append(dict(arm=record['arm'],seed=record['seed'],selected_by=record['selected_by'],
                              auc=report['pooled_auc'],macro=report['macro_assay_auc'],within=report['within_pair_weighted_auc']))
        write(out/'test_progress.json',test_rows)
        print('Test evaluation:',name,report['pooled_auc'],flush=True)
    summary = {arm:{policy:statistics_of([r['auc'] for r in test_rows if r['arm']==arm and r['selected_by']==policy])
                    for policy in ['independent','shared']} for arm in roots}
    write(out/'summary.json',dict(test_auc=summary,per_seed=test_rows,decision=decision))
    write(out/'complete.json',dict(complete=True,finished_at=now(),training_runs=0))
