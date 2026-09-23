"""Sequential server follow-up: extra Assay seeds, audited new datasets, smoke, summary."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import os
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from src.experiments.search import run_eval
from src.experiments.plans import make_plan
from src.experiments.inference import compare
from src.data.preprocess import run_preprocessing
from train_sugar import Config


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


from src.experiments.plans import base_config


def audit_raw(path):
    from rdkit import Chem
    splits = json.loads(Path(path).read_text())['split']
    report, canonical = {}, {}
    for name in ['train', 'ood_val', 'ood_test']:
        rows = splits[name]
        labels = Counter(r['cls_label'] for r in rows)
        if set(labels) != {0, 1}:
            raise ValueError(f'{name}: expected both binary labels, got {labels}')
        ids, invalid = [], 0
        for r in rows:
            if 'domain_id' not in r:
                raise ValueError(f'{name}: missing domain_id')
            mol = Chem.MolFromSmiles(r['smiles'])
            if mol is None:
                invalid += 1
            else:
                ids.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        canonical[name] = set(ids)
        report[name] = dict(rows=len(rows), labels=dict(labels), domains=len({r['domain_id'] for r in rows}),
                            invalid_smiles=invalid, repeated_canonical_rows=len(ids)-len(set(ids)))
    report['canonical_overlap'] = {f'{a}__{b}': len(canonical[a] & canonical[b])
                                   for a,b in [('train','ood_val'),('train','ood_test'),('ood_val','ood_test')]}
    report['note'] = 'Canonical isomeric SMILES overlap is diagnostic; supplied benchmark splits are preserved.'
    return report


def evaluate(base, output, first_seed, runs, model='fragment'):
    base = dict(base)
    if model == 'erm':
        base.update(erm_baseline=True, lambda_contrast=0., lambda_diversity=0., lambda_irm=0.)
    # Retain five classifiers in ERM for a matched ensemble-size comparison.
    output.mkdir(parents=True, exist_ok=False)
    params = output/'params.json'
    write(params, {'params': {}})
    write(output/'frozen_config.json', base)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [*ROOT.glob('*.py'), *(ROOT/'src').rglob('*.py'), *(ROOT/'scripts').glob('*.py')]}
    write(output/'provenance.json', dict(python=sys.version, torch=torch.__version__, source_sha256=hashes))
    spec = dict(output_dir=str(output), eval_runs=runs, eval_base_seed=first_seed,
                auc_ma_window=1, test_split='ood_test', retain_eval_artifacts=True)
    run_eval(spec, base, SimpleNamespace(params=str(params), epochs=None, device=base['device'], runs=None, dry_run=False))


def summarize(paths, output):
    records, params = [], None
    for path in paths:
        path = Path(path)
        data = json.loads((path/'eval/eval_results.json').read_text())
        for record in data['runs']:
            config = json.loads((path/'eval/runs'/f'{record["run_name"]}.json').read_text())['config']
            keys = ['dataset','learning_rate','weight_decay','dropout','edge_ratio','fragment_sample_ratio',
                    'lambda_contrast','lambda_diversity','lambda_irm','num_views','erm_baseline','epochs',
                    'early_stopping','patience','early_stopping_min_epochs','selection_ma_window','aggregation_mode']
            keys += ['hidden_dim','gnn_layers','batch_size','view_fragmentations','selection_weight_avg',
                     'greedy_model_selection','inference_policy','contrast_temperature','contrast_sampling','processed_dir']
            current = {k: config.get(k) for k in keys}
            if params is not None and params != current:
                raise ValueError('Cannot pool different experiment configurations')
            params = current
            if record['seed'] in [r['seed'] for r in records]:
                raise ValueError('Duplicate seed; do not pool reruns as independent seeds')
            records.append(record)
    from src.experiments.common import _summarize
    result = dict(config=params, runs=records, summary={s: _summarize([r[s]['auc'] for r in records]) for s in ['ood_val','ood_test']})
    with Path(output).open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result['summary'], indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare','train','assay-extra','smoke','summarize'])
    p.add_argument('--dataset', choices=['scaffold','size'])
    p.add_argument('--model', choices=['fragment','erm','both'], default='both')
    p.add_argument('--cache-dir')
    p.add_argument('--output-dir')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--results', nargs='+')
    args = p.parse_args(argv)
    os.chdir(ROOT)
    if args.action == 'summarize':
        if not args.results or not args.output_dir:
            p.error('summarize needs --results ROOT... and --output-dir FILE.json')
        summarize(args.results, args.output_dir)
        return
    if args.action in ['prepare','train'] and not args.dataset:
        p.error('--dataset required')
    cache = Path(args.cache_dir or f'data/processed/ec50_followup_v1/{args.dataset}').resolve()
    out = Path(args.output_dir or f'outputs/{args.dataset or "assay"}/benchmark/{args.action}_{datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
    base = base_config(args.dataset or 'assay', cache, args.device)
    print(f'{args.action}: dataset={args.dataset or "assay"}, output={out}, cache={base["processed_dir"]}', flush=True)
    print('Budget: Assay extra=2 runs; each new dataset=5 seeds x 2 models=10 runs; max100 epochs/run.', flush=True)
    if args.dry_run:
        return
    if args.action != 'prepare' and args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    if args.action == 'prepare':
        if cache.exists():
            raise FileExistsError(f'Fresh cache required: {cache}; use --cache-dir for a new location')
        report = audit_raw(base['raw_path'])
        cache.mkdir(parents=True)
        write(cache/'raw_audit.json', report)
        print(json.dumps(report, indent=2), flush=True)
        run_preprocessing(Config(base))
        counts = {}
        for split in base['split_names']:
            payload = torch.load(cache/f'{split}.pt', weights_only=False)
            counts[split] = len(payload['molecule_data'])
            if counts[split] == 0:
                raise ValueError(f'No processed examples in {split}')
        from src.data.dataloader import build_split_dataloaders
        check = dict(base, num_workers=0, batch_size=2)
        loaders = build_split_dataloaders(str(cache), Config(check))
        for split in base['split_names']:
            batch = next(iter(loaders[split]))
            if batch.x.ndim != 2 or not torch.isfinite(batch.x).all():
                raise ValueError(f'Invalid atom features in {split}')
        write(cache/'ready.json', dict(dataset=base['dataset'], raw_sha256=hashlib.sha256(Path(base['raw_path']).read_bytes()).hexdigest(), processed_counts=counts))
        print('Prepared:', counts, flush=True)
        return
    if args.action == 'smoke':
        out.mkdir(parents=True, exist_ok=False)
        _, fixture = make_plan(out, args.device, True)
        for k in ['dataset','raw_path','raw_format','splits','label_key','processed_dir','strict','batch_size','num_workers','pin_memory','hidden_dim','gnn_layers']:
            base[k] = fixture[k]
        base.update(epochs=2, early_stopping_min_epochs=0)
        torch.set_num_threads(1)
        run_preprocessing(Config(base))
        evaluate(base, out/'fragment', 3004, 1)
        evaluate(base, out/'erm', 3004, 1, 'erm')
        checkpoint = next((out/'fragment').rglob('best_model.pt'))
        expected = json.loads((out/'fragment/eval/eval_results.json').read_text())['runs'][0]
        compare(checkpoint, out/'comparison', args.device, base['processed_dir'], expected)
        print('SMOKE PASSED: synthetic results only', flush=True)
        return
    for split in base['loader_split_names']:
        if not (Path(base['processed_dir'])/f'{split}.pt').is_file():
            raise FileNotFoundError(f'Missing processed {split}: {base["processed_dir"]}')
    if args.action == 'assay-extra':
        evaluate(base, out, 3004, 2)
    else:
        ready = json.loads((cache/'ready.json').read_text())
        if ready['dataset'] != base['dataset'] or ready['raw_sha256'] != hashlib.sha256(Path(base['raw_path']).read_bytes()).hexdigest():
            raise ValueError('Cache source provenance mismatch; prepare a fresh cache')
        out.mkdir(parents=True, exist_ok=False)
        for model in (['fragment','erm'] if args.model == 'both' else [args.model]):
            evaluate(base, out/model, 3001, 5, model)


if __name__ == '__main__':
    main()
