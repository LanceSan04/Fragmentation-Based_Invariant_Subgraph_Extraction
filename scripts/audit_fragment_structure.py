"""Read-only topology/ratio audit of existing train and validation caches."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from rdkit import Chem
from src.data.dataloader import load_processed_split


def topology(n, edges):
    neighbors = [set() for _ in range(n)]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    seen = set()
    components = 0
    for start in range(n):
        if start in seen:
            continue
        components += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(neighbors[node] - seen)
    return dict(components=components, degrees=[len(x) for x in neighbors])


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), min=float(a.min()),
                p25=float(np.quantile(a, .25)), median=float(np.median(a)),
                p75=float(np.quantile(a, .75)), max=float(a.max()))


def inspect(data, index):
    n = data.fragment_x_static.size(0)
    edges = data.fragment_edge_index.T.tolist()
    graph = topology(n, edges)
    sizes = torch.bincount(data.atom_to_fragment, minlength=n).tolist()
    metadata = data.metadata
    return dict(index=index, atoms=data.x.size(0), bonds=data.bond_index.size(1),
                fragments=n, connections=len(edges), components=graph['components'],
                mean_degree=sum(graph['degrees']) / n, max_degree=max(graph['degrees']),
                terminal_fragments=sum(d == 1 for d in graph['degrees']),
                isolated_fragments=sum(d == 0 for d in graph['degrees']),
                largest_fragment_atoms=max(sizes), fragment_sizes=sizes,
                label=int(data.y.item()), assay=str(metadata.get('assay_id')),
                smiles=Chem.MolToSmiles(Chem.MolFromSmiles(data.smiles)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--processed-dir', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(splits={}, ratio_budget={}, note='No model inference or test data loaded. Ratios below describe connection budgets, not retained atoms.')
    all_rows = {}
    for split in ['train', 'ood_val']:
        path = Path(args.processed_dir) / (split + '.pt')
        payload = load_processed_split(path)
        rows = [inspect(d, i) for i, d in enumerate(payload['molecule_data'])]
        all_rows[split] = rows
        domains = {}
        for r in rows:
            domains.setdefault(r['assay'], []).append(r)
        result['splits'][split] = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            metadata=payload['metadata'], n=len(rows),
            summary={k: describe([r[k] for r in rows]) for k in
                     ['atoms', 'bonds', 'fragments', 'connections', 'mean_degree', 'largest_fragment_atoms']},
            tiny_connections=dict(Counter(str(min(r['connections'], 4)) for r in rows)),
            disconnected=sum(r['components'] > 1 for r in rows),
            positive_fraction=sum(r['label'] for r in rows) / len(rows),
            domain_count=len(domains), single_class_domains=sum(len({r['label'] for r in group}) == 1 for group in domains.values()),
            domain_sizes=describe([len(group) for group in domains.values()]),
            duplicate_smiles=len(rows)-len({r['smiles'] for r in rows}))
        for sample in [.5, .7, .9, 1.]:
            for edge in [.3, .5, .7]:
                budget=[]
                for r in rows:
                    e=r['connections']
                    eligible=max(1, int(e*sample)) if e else 0
                    target=max(1, int(e*edge)) if e else 0
                    budget.append((eligible, min(eligible,target), target))
                result['ratio_budget'][f'{split}:sample={sample}:edge={edge}'] = dict(
                    training_mean_selected=float(np.mean([b[1] for b in budget])),
                    eval_mean_selected=float(np.mean([b[2] for b in budget])),
                    candidate_limited=sum(b[1]<b[2] for b in budget),
                    all_candidates_selected=sum(b[0]>0 and b[0]==b[1] for b in budget))
        (out/(split+'_molecules.json')).write_text(json.dumps(rows, indent=2))
    result['cross_split'] = dict(
        common_assays=sorted({r['assay'] for r in all_rows['train']} & {r['assay'] for r in all_rows['ood_val']}),
        common_canonical_smiles=len({r['smiles'] for r in all_rows['train']} & {r['smiles'] for r in all_rows['ood_val']}))
    ordered=sorted(all_rows['ood_val'], key=lambda r:(r['connections'],r['index']))
    result['visual_indices']=[ordered[round(q*(len(ordered)-1))]['index'] for q in [0,.25,.5,.75,1]]
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(visual_indices=result['visual_indices'],cross_split=result['cross_split']),indent=2))


if __name__ == '__main__':
    main()
