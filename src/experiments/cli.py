"""Unified experiment CLI; delegates to reusable workflow modules."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
WORKFLOWS={
    'selection-gating-probe':'selection_gating_probe',
    'cross-domain':'cross_domain',
    'contrastive-sweep':'contrastive_sweep', 'erm-tune':'erm_tune',
    'visualize-views':'visualize_views', 'tune':'parameter_search', 'benchmark':'benchmark', 'select':'selection',
    'baseline-eval':'baseline_eval',
    'scorer-study':'scorer_study',
    'contrastive-study':'contrastive_study',
    'assay-followup':'assay_followup',
    'connectivity-probe':'connectivity_probe',
    'fragment-ratio-sweep':'fragment_ratio_sweep',
    'atom-budget-study':'atom_budget_study',
    'compare-inference':'inference', 'compare-views':'views',
    'diagnose':'diagnostics', 'overnight':'overnight',
    'ablation':'ablation', 'diversity':'diversity',
}
OUTPUT_WORKFLOWS={'tune','erm-tune','benchmark','select','baseline-eval','compare-inference','compare-views','diagnose','overnight'}

def option(args,name,default=None):
    for i,arg in enumerate(args):
        if arg==name:return args[i+1] if i+1<len(args) else default
        if arg.startswith(name+'='):return arg.split('=',1)[1]
    return default

def main(argv=None):
    parser=argparse.ArgumentParser(description='Unified experiments. Example: main.py experiment select --help')
    parser.add_argument('workflow',choices=WORKFLOWS)
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    parsed=parser.parse_args(argv)
    args=parsed.arguments;workflow=parsed.workflow
    os.chdir(ROOT)
    output=None
    summarizing=workflow=='benchmark' and args and args[0]=='summarize'
    preparing=workflow=='benchmark' and args and args[0]=='prepare'
    if workflow in OUTPUT_WORKFLOWS and not summarizing and not preparing:
        supplied=option(args,'--output-dir')
        if supplied:output=Path(supplied).resolve()
        else:
            smoke='--smoke' in args or (workflow=='benchmark' and args and args[0]=='smoke')
            dataset='smoke' if smoke else option(args,'--dataset','assay' if workflow in {'tune','select','benchmark','overnight'} else 'analysis')
            output=ROOT/'outputs'/dataset/workflow/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            args=args+['--output-dir',str(output)]
    module=importlib.import_module('src.experiments.'+WORKFLOWS[workflow])
    dry='--dry-run' in args or '--help' in args or '-h' in args
    # Never overwrite a previous experiment's metadata, even if a workflow fails early.
    existed=output.exists() if output else False
    record=dict(workflow=workflow,arguments=args,started_at=datetime.now(timezone.utc).isoformat(),status='failed')
    try:
        result=module.main(args)
        record['status']='returned'  # Inspect native trial/run states for completeness.
        return result
    finally:
        if output and output.is_dir() and not existed and not dry:
            record['result_files']=[str(p.relative_to(output)) for pattern in ['summary.json','eval_results.json','comparison_all.json','comparison.json','best_params.json','leaderboard.csv'] for p in output.rglob(pattern)]
            with (output/'experiment.json').open('x') as f:json.dump(record,f,indent=2)

if __name__=='__main__':main()
