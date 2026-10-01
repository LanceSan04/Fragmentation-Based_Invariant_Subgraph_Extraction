"""Read-only EC50 integrity/config audit against SuGAr's pinned DrugOOD curator.

Writes a fresh audit directory only; never changes data, labels or splits.
Upstream Python configs are parsed as restricted literal ASTs, never executed.
"""
import argparse
import ast
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import posixpath
import urllib.request

from rdkit import Chem, rdBase

COMMIT = 'eeb00b8da7646e1947ca7aec93041052a48bd45e'
BASE = f'https://raw.githubusercontent.com/tencent-ailab/DrugOOD/{COMMIT}/'


def literal(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
        if node.args or any(k.arg is None for k in node.keywords):
            raise ValueError('Unsupported dictionary expression')
        return {k.arg: literal(k.value) for k in node.keywords}
    return ast.literal_eval(node)


def merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = value
    return target


def upstream_config(path, out, sources):
    text = urllib.request.urlopen(BASE + path, timeout=30).read()
    saved = out / 'upstream' / path
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_bytes(text)
    sources[path] = dict(url=BASE + path, sha256=hashlib.sha256(text).hexdigest())
    values = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ValueError('Unsupported config assignment')
            values[node.targets[0].id] = literal(node.value)
    result = {}
    for parent in values.pop('_base_', []):
        merge(result, upstream_config(posixpath.normpath(posixpath.join(posixpath.dirname(path), parent)), out, sources))
    return merge(result, values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', default='data/raw/drugood')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(upstream_commit=COMMIT, rdkit_version=rdBase.rdkitVersion,
                  sources={}, datasets={}, limitations=[
                      'Config equality does not establish which curator commit originally generated local data.',
                      'No original ChEMBL database, exact SuGAr JSON, or curation runtime was compared.',
                      'Canonical identity checks do not exclude near-duplicate chemistry.',
                      'Canonicalization uses the recorded local RDKit version.'])
    for domain in ('assay', 'scaffold', 'size'):
        name = f'lbap_core_ec50_{domain}'
        path = Path(args.raw_dir) / (name + '.json')
        raw = path.read_bytes()
        data = json.loads(raw)
        official = upstream_config(f'configs/curators/{name}.py', out, report['sources'])
        actual = json.loads(json.dumps(data['cfg']))
        # Physical file locations are not scientific curation parameters.
        for cfg in (official, actual):
            cfg['path'].pop('source_root', None)
            cfg['path'].pop('target_root', None)
        rows = [r for values in data['split'].values() for r in values]
        vals = sorted(r['reg_label'] for r in rows)
        median = vals[len(vals)//2]
        cfg = actual['classification_threshold']
        threshold = median if cfg['lower_bound'] <= median <= cfg['upper_bound'] else cfg['fix_value']
        result = dict(sha256=hashlib.sha256(raw).hexdigest(),
                      config_matches_pinned_upstream=actual == official,
                      normalized_local_config=actual, normalized_upstream_config=official,
                      total_rows=len(rows), inferred_threshold_from_all_exported_rows=threshold,
                      recorded_threshold=data.get('statistics', {}).get('thr_for_cls'),
                      label_mismatches=sum(r['cls_label'] != int(r['reg_label'] >= threshold) for r in rows),
                      splits={}, overlaps={})
        keys, domains = {}, {}
        for split, records in data['split'].items():
            canonical = []
            invalid = 0
            sizes = []
            for row in records:
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol is None:
                    invalid += 1
                    continue
                canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True))
                sizes.append(mol.GetNumAtoms())
            keys[split] = set(canonical)
            domains[split] = set(r['domain_id'] for r in records)
            result['splits'][split] = dict(n=len(records), labels=dict(Counter(r['cls_label'] for r in records)),
                invalid_smiles=invalid, duplicate_canonical_rows=len(canonical)-len(keys[split]),
                domains=len(domains[split]), atom_range=[min(sizes), max(sizes)] if sizes else None)
        for a, b in combinations(keys, 2):
            result['overlaps'][a + '__' + b] = dict(canonical_molecules=len(keys[a] & keys[b]),
                                                      domain_ids=len(domains[a] & domains[b]))
        report['datasets'][name] = result
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, result in report['datasets'].items():
        print(name, 'config_match=', result['config_matches_pinned_upstream'],
              'threshold=', result['inferred_threshold_from_all_exported_rows'],
              'recorded=', result['recorded_threshold'], 'label_mismatches=', result['label_mismatches'])
        print('splits:', result['splits'])
        print('overlaps:', result['overlaps'])


if __name__ == '__main__':
    main()
