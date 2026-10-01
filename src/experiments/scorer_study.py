"""Fixed validation-only ordered versus symmetric scorer study, sequential arms."""
import argparse
import json
import statistics
from pathlib import Path
from src.experiments import selection


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-dir');p.add_argument('--raw-path')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--smoke',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if not a.smoke and (not a.processed_dir or not a.raw_path):
        p.error('Real study requires processed-dir and raw-path')
    seeds=[5001] if a.smoke else list(range(8301,8306))
    out=Path(a.output_dir)
    if out.exists():raise FileExistsError(out)
    plan=dict(modes=['ordered','symmetric'],seeds=seeds,epochs=3 if a.smoke else 100,
              runs=2 if a.smoke else 10,device=a.device,processed_dir=a.processed_dir,
              primary='independent/independent',secondary='shared/independent',test_evaluation=False,
              topk_tie_policy='stable_original_edge_order',lambda_contrast=.1,lambda_diversity=0.,lambda_irm=0.,
              edge_ratio=.5,fragment_sample_ratio=.7,learning_rate=.0003,dropout=.5,weight_decay=0.)
    if a.dry_run:print(json.dumps(plan,indent=2));return plan
    out.mkdir(parents=True)
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    for mode in plan['modes']:
        args=['--model','fragment','--dataset','assay','--device',a.device,'--seeds',*[str(s) for s in seeds],
              '--epochs',str(plan['epochs']),'--scoring-mode',mode,'--lambda-contrast','0.1',
              '--learning-rate','0.0003','--dropout','0.5','--weight-decay','0',
              '--output-dir',str(out/mode)]
        if a.smoke:args.append('--smoke')
        else:args+=['--processed-dir',a.processed_dir,'--raw-path',a.raw_path]
        selection.main(args)
    records={mode:json.loads((out/mode/'summary.json').read_text())['runs'] for mode in plan['modes']}
    paired=[]
    for left,right in zip(records['ordered'],records['symmetric']):
        if left['seed']!=right['seed']:raise ValueError('Paired seeds do not match')
        ordered=left['results']['independent']['independent']['ood_val']['auc']
        symmetric=right['results']['independent']['independent']['ood_val']['auc']
        paired.append(dict(seed=left['seed'],ordered=ordered,symmetric=symmetric,delta=symmetric-ordered))
    (out/'comparison.json').write_text(json.dumps(dict(primary=plan['primary'],paired=paired,
        mean_delta=statistics.mean(r['delta'] for r in paired),
        sample_std_delta=statistics.stdev(r['delta'] for r in paired) if len(paired)>1 else None),indent=2))
    (out/'complete.json').write_text(json.dumps(dict(complete=True,plan=plan),indent=2))
