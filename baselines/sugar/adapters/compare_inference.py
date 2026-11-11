"""Exploratory same-checkpoint shared-versus-independent comparison; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from scipy.stats import t
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from compat import BASE,verify_sources,install_masks,feature_module
from ensemble import probabilities
from run import convert,write


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(manifest):
    seeds=[r['initialization'] for r in manifest['runs']]
    expected=[2,3] if manifest['synthetic'] else [2,3,4,5,6]
    if manifest['schema']!=1 or seeds!=expected:
        raise ValueError('Unexpected comparison seed set')
    if manifest['policies']!=['shared_raw_uniform','independent_uniform']:
        raise ValueError('Unexpected comparison policies')


def summarize(rows):
    result={}
    for split in ('ood_val','ood_test'):
        subset=[r for r in rows if r['split']==split]
        deltas=[r['independent_auc']-r['shared_auc'] for r in subset]
        sd=statistics.stdev(deltas) if len(deltas)>1 else None
        margin=float(t.ppf(.975,len(deltas)-1)*sd/len(deltas)**.5) if sd is not None else None
        result[split]=dict(n=len(deltas),shared_mean=statistics.mean(r['shared_auc'] for r in subset),
            independent_mean=statistics.mean(r['independent_auc'] for r in subset),
            shared_sample_sd=statistics.stdev(r['shared_auc'] for r in subset) if len(deltas)>1 else None,
            independent_sample_sd=statistics.stdev(r['independent_auc'] for r in subset) if len(deltas)>1 else None,
            mean_paired_difference=statistics.mean(deltas),sample_sd_difference=sd,
            independent_wins=sum(d>0 for d in deltas),
            approximate_seed_t_interval_95=None if margin is None else
                [statistics.mean(deltas)-margin,statistics.mean(deltas)+margin])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--replication-root',required=True)
    p.add_argument('--raw-path',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();verify_sources();torch.set_num_threads(1)
    manifest=json.loads(Path(a.manifest).read_text());base=Path(a.replication_root).resolve()
    validate_manifest(manifest)
    if digest(base/'completed.json')!=manifest['replication_completed_sha256']:
        raise ValueError('Replication summary changed')
    raw=Path(a.raw_path);synthetic=manifest['synthetic']
    if not synthetic and digest(raw)!=manifest['raw_sha256']:raise ValueError('Dataset hash changed')
    # Verify every frozen checkpoint and prior record before any evaluation.
    for entry in manifest['runs']:
        run=base/'runs'/f"init{entry['initialization']}"
        for name,expected in entry['files'].items():
            if digest(run/name)!=expected:raise ValueError('Reference file changed: '+name)
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'protocol.json',manifest)
    sys.path.insert(0,str(BASE/'upstream'))
    from models import ciga
    install_masks(ciga);features=feature_module();rows=[];start=time.perf_counter()
    for split,folder in [('ood_val','validation'),('ood_test','test')]:
        if synthetic:
            records=[dict(smiles=s,cls_label=i%2) for i,s in enumerate(
                ['CCOC(=O)C','CCN(CC)CC','c1ccccc1CC(=O)NC','CCCOC'])]
        else:records=json.loads(raw.read_bytes())['split'][split]
        data=convert(records,features);labels=np.array([int(r['cls_label']) for r in records])
        for entry in manifest['runs']:
            seed=entry['initialization'];run=base/'runs'/f'init{seed}'
            spec=json.loads((run/'train/run_spec.json').read_text())
            if spec.get('contrastive_indexing')!='released':raise ValueError('Unexpected corrected training')
            state=torch.load(run/'train/best_views.pt',map_location='cpu',weights_only=False)
            args=state['args'];models=[]
            for weights in state['view_states']:
                model=ciga.CIGA(ratio=args['r'],input_dim=39,edge_dim=10,out_dim=2,
                    gnn_type=args['model'],num_layers=args['num_layers'],emb_dim=args['emb_dim'],
                    drop_ratio=args['dropout'],graph_pooling=args['pooling'],virtual_node=args['virtual_node'],
                    c_dim=args['classifier_emb_dim'],c_in=args['classifier_input_feat'],
                    c_rep=args['contrast_rep'],c_pool=args['contrast_pooling'],s_rep=args['spurious_rep'])
                model.load_state_dict(weights);models.append(model.to(a.device).eval())
            result=dict(initialization=seed,split=split)
            for name,policy in [('shared','shared_raw_uniform'),('independent','independent_uniform')]:
                predictions=[]
                for batch in DataLoader(data,batch_size=args['batch_size'],shuffle=False):
                    predictions.append(probabilities(models,batch.to(a.device),policy).cpu().numpy())
                pred=np.concatenate(predictions);auc=float(roc_auc_score(labels,pred))
                if name=='shared':
                    previous=json.loads((run/folder/'completed.json').read_text())['variants']['all_mean']['auc']
                    if abs(auc-previous)>1e-4:raise RuntimeError('Shared reference replay failed')
                np.savez(out/f'init{seed}_{split}_{name}.npz',y_true=labels,y_pred=pred)
                result[name+'_auc']=auc
            rows.append(result);write(out/'results.json',rows)
            print(json.dumps(result),flush=True)
    report=dict(rows=rows,summary=summarize(rows),wall_seconds=time.perf_counter()-start,
        synthetic=synthetic,training_runs=0,checkpoint_selection_changed=False,
        scope='Exploratory inference comparison on already observed test split; no automatic policy promotion',
        checkpoint_selection='Existing per-view best validation epoch; identical weights for both policies')
    write(out/'completed.json',report)


if __name__=='__main__':main()
