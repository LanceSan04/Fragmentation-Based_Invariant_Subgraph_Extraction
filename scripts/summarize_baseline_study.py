"""Summarize a frozen, paired Assay study without selecting on test results."""

import argparse
import json
from pathlib import Path
import statistics


def load_arm(path):
    root=Path(path).resolve()
    summary=json.loads((root/'summary.json').read_text())
    records={int(row['seed']):row for row in summary['runs']}
    if len(records)!=len(summary['runs']):raise ValueError(f'Duplicate seeds in {root}')
    return root,records


def stats(values):
    return dict(n=len(values),mean=statistics.mean(values),
                sample_std=statistics.stdev(values) if len(values)>1 else None)


def primary_auc(record,model,split):
    policy='shared' if model=='erm' else 'independent'
    return float(record['results'][policy][policy][split]['auc'])


def summarize(learned_path,random_path,erm_path):
    arms={name:load_arm(path) for name,path in
          [('learned',learned_path),('random',random_path),('erm',erm_path)]}
    seeds=set(arms['learned'][1])
    if not seeds or any(set(records)!=seeds for _,records in arms.values()):
        raise ValueError('All arms must have the same nonempty seed set')
    seeds=sorted(seeds)
    for seed in seeds:
        learned=json.loads((arms['learned'][0]/f'seed{seed}'/'config.json').read_text())
        random=json.loads((arms['random'][0]/f'seed{seed}'/'config.json').read_text())
        learned.pop('selection_mode',None);random.pop('selection_mode',None)
        if learned!=random:
            raise ValueError(f'Learned/random settings differ beyond selection_mode for seed {seed}')
    per_seed=[]
    for seed in seeds:
        row={'seed':seed}
        for name,(_,records) in arms.items():
            row[name]={split:primary_auc(records[seed],name,split)
                       for split in ('ood_val','ood_test')}
        row['learned_minus_random']=row['learned']['ood_test']-row['random']['ood_test']
        row['learned_minus_erm']=row['learned']['ood_test']-row['erm']['ood_test']
        per_seed.append(row)
    return dict(seeds=seeds,primary_policy=dict(learned='independent/independent',
                 random='independent/independent',erm='shared/shared'),
        per_seed=per_seed,
        summaries={name:{split:stats([row[name][split] for row in per_seed])
                         for split in ('ood_val','ood_test')}
                   for name in arms},
        paired_differences={name:stats([row[name] for row in per_seed])
                            for name in ('learned_minus_random','learned_minus_erm')},
        caveat='Assay test results were previously inspected; this comparison is exploratory.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('learned','random','erm'):p.add_argument('--'+name,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    result=summarize(a.learned,a.random,a.erm)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(summaries=result['summaries'],
                          paired_differences=result['paired_differences']),indent=2))


if __name__=='__main__':main()
