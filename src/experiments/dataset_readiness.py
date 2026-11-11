"""Read-only source audit before frozen molecular dataset comparisons."""
import argparse
from collections import Counter,defaultdict
import itertools
from pathlib import Path

from rdkit import Chem
from src.experiments.cross_domain import dataset_config
from src.data.preprocess import load_raw_dataset
from src.data.good_hiv import file_sha256
from src.experiments.study_queue import write
from src.training.pipeline import Config


def audit_dataset(dataset,raw_root):
    cfg=dataset_config(dataset,raw_root,'data/processed/readiness_not_created','cpu')
    records,source=load_raw_dataset(Config(cfg))
    split_info={};canonical={};ids={}
    for split in ('train','ood_val','ood_test'):
        rows=records[split];labels=Counter();counts=Counter();assays=defaultdict(Counter)
        source_ids=[];invalid=0;threshold_mismatch=0
        for row in rows:
            label=row[cfg['label_key']]
            if label not in (0,1):raise ValueError('Non-binary label')
            labels[int(label)]+=1
            mol=Chem.MolFromSmiles(row[cfg['smiles_key']])
            if mol is None or mol.GetNumAtoms()==0:
                invalid+=1;continue
            counts[Chem.MolToSmiles(mol,isomericSmiles=True)]+=1
            assay=row.get('assay_id')
            if assay is not None:assays[str(assay)][int(label)]+=1
            if 'good_source_id' in row:source_ids.append(row['good_source_id'])
            if dataset.startswith('ki_') and row.get('reg_label') is not None:
                threshold_mismatch+=int(int(label)!=int(float(row['reg_label'])>=5.0))
        if set(labels)!={0,1} or invalid:
            raise ValueError(f'{dataset}/{split}: invalid molecules or missing class')
        if source_ids and len(set(source_ids))!=len(source_ids):
            raise ValueError('Duplicate official source identifiers')
        canonical[split]=set(counts);ids[split]=set(source_ids)
        split_info[split]=dict(molecules=len(rows),negative=labels[0],positive=labels[1],
            positive_fraction=labels[1]/len(rows),canonical_duplicates=sum(n-1 for n in counts.values()),
            assays=len(assays),eligible_assays=sum(c[0]>=5 and c[1]>=5 for c in assays.values()),
            ki_label_disagreements_with_p_activity_5=threshold_mismatch)
    overlaps={f'{a}:{b}':len(canonical[a]&canonical[b]) for a,b in itertools.combinations(canonical,2)}
    id_overlaps={f'{a}:{b}':len(ids[a]&ids[b]) for a,b in itertools.combinations(ids,2)}
    if any(id_overlaps.values()):raise ValueError('Official source index crosses splits')
    return dict(dataset=dataset,raw_path=cfg['raw_path'],raw_sha256=file_sha256(cfg['raw_path']),
                splits=split_info,canonical_overlap=overlaps,official_id_overlap=id_overlaps,source=source,
                needs_review=any(overlaps.values()) or any(x['ki_label_disagreements_with_p_activity_5'] for x in split_info.values()),
                note='Source audit, not training. Canonical overlap must be interpreted against official benchmark membership; nothing is removed or resplit. Ki threshold diagnostic assumes p-activity labels; it is not independent provenance certification.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',default='data/raw')
    p.add_argument('--datasets',nargs='+',default=['ki_scaffold','ki_size','good_hiv_scaffold','good_hiv_size'],
                   choices=['ki_scaffold','ki_size','good_hiv_scaffold','good_hiv_size'])
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cpu',help='Accepted for guarded launch; audit is CPU only')
    a=p.parse_args(argv)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    report={}
    for dataset in a.datasets:
        print('Auditing '+dataset,flush=True)
        report[dataset]=audit_dataset(dataset,a.raw_root)
        write(out/'audit.json',report)
    return report
