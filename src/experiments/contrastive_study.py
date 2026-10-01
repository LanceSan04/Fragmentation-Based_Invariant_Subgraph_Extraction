"""Bounded validation-only comparison of pooled, absent and per-view contrast."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from src.experiments import selection


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-dir')
    p.add_argument('--raw-path')
    p.add_argument('--reference-dir', help='Completed scorer-study results directory')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(argv)
    if not a.smoke and not all([a.processed_dir,a.raw_path,a.reference_dir]):
        p.error('Real study requires processed-dir, raw-path and reference-dir')
    out = Path(a.output_dir).resolve()
    seeds = [5001] if a.smoke else list(range(8301,8306))
    plan = dict(seeds=seeds, epochs=3 if a.smoke else 100, new_runs=2*len(seeds),
                arms={'off':0.,'per_view':.1}, test_evaluation=False,
                primary='independent/independent', secondary='shared/independent',
                note='Per-view SUM changes both pair definition and loss scale; not a pure pair-set ablation.')
    references = {}
    if not a.smoke:
        ref = Path(a.reference_dir).resolve()
        complete = json.loads((ref/'complete.json').read_text())
        if not complete.get('complete') or complete['plan']['seeds'] != seeds or complete['plan']['epochs'] != 100:
            raise ValueError('Reference must be the completed five-seed 100-epoch scorer study')
        expected = selection.base_config('assay',Path(a.processed_dir),a.device)
        expected.update(epochs=100,learning_rate=.0003,dropout=.5,weight_decay=0.,lambda_contrast=.1,lambda_diversity=0.,lambda_irm=0.)
        keys = ['num_views','hidden_dim','gnn_layers','batch_size','edge_ratio','fragment_sample_ratio',
                'contrast_sampling','contrast_temperature','aggregation_mode','epochs','learning_rate','dropout',
                'weight_decay','lambda_contrast','lambda_diversity','lambda_irm','fragmentation_methods']
        for mode in ['ordered','symmetric']:
            references[mode] = json.loads((ref/mode/'summary.json').read_text())
            for seed in seeds:
                config = json.loads((ref/mode/f'seed{seed}'/'config.json').read_text())
                mismatch = {k:(config.get(k),expected.get(k)) for k in keys if config.get(k)!=expected.get(k)}
                if mismatch: raise ValueError(f'Reference configuration mismatch: {mismatch}')
                if not (ref/mode/f'seed{seed}'/'best_independent.pt').is_file():
                    raise FileNotFoundError('Missing reference checkpoint')
    out.mkdir(parents=True,exist_ok=False)
    selection.write(out/'plan.json',plan)
    if references:
        selection.write(out/'reference_summaries.json',references)
        checkpoints = [ref/m/f'seed{s}'/'best_independent.pt' for m in ['ordered','symmetric'] for s in seeds]
        selection.write(out/'reference_hashes.json',{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in checkpoints})
        subprocess.run([sys.executable,'scripts/audit_validation_groups.py',*[str(f) for f in checkpoints],
                        '--processed-dir',a.processed_dir,'--output-dir',str(out/'assay_diagnostics')],check=True)
        for checkpoint in checkpoints:
            name = checkpoint.parent.parent.name+'_'+checkpoint.parent.name
            subprocess.run([sys.executable,'scripts/audit_view_behavior.py',str(checkpoint),
                            '--processed-dir',a.processed_dir,'--output-dir',str(out/'view_diagnostics'/name),
                            '--molecules','100'],check=True)
    for arm, coefficient in plan['arms'].items():
        args = ['--model','fragment','--dataset','assay','--device',a.device,
                '--seeds',*[str(s) for s in seeds],'--epochs',str(plan['epochs']),
                '--scoring-mode','ordered','--contrast-scope','per_view' if arm=='per_view' else 'pooled',
                '--lambda-contrast',str(coefficient),'--learning-rate','0.0003','--dropout','0.5',
                '--weight-decay','0','--log-gradient-influence','--output-dir',str(out/arm)]
        args += ['--smoke'] if a.smoke else ['--processed-dir',a.processed_dir,'--raw-path',a.raw_path]
        selection.main(args)
    summary = {arm:json.loads((out/arm/'summary.json').read_text()) for arm in plan['arms']}
    if references: summary['pooled_reference'] = references['ordered']
    selection.write(out/'comparison.json',summary)
    selection.write(out/'complete.json',dict(complete=True,plan=plan))
