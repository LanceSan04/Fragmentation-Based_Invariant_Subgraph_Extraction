"""Validation-only implementation of SuGAr Algorithms 1 and 3 (ENS).

Primary: all views, mean raw scores, shared top-k, uniform probability voting.
Secondary: maximum scores and optional validation-only greedy view selection.
Test evaluation requires an explicit frozen protocol and its validation record.
Never trains or imports the executable upstream ens.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from compat import BASE,verify_sources,install_masks,feature_module
from ensemble import probabilities
from run import convert,write


def greedy_select(individual_auc,evaluate):
    """Algorithm 1: descending individual validation score, accept ties (>=)."""
    order=sorted(range(len(individual_auc)),key=lambda i:(-individual_auc[i],i))
    selected=[];best=-float('inf');trace=[]
    for view in order:
        candidate=selected+[view];auc=float(evaluate(candidate))
        if not np.isfinite(auc):raise ValueError('Nonfinite candidate AUC')
        accepted=auc>=best
        trace.append(dict(candidate=candidate,auc=auc,accepted=accepted))
        if accepted:selected=candidate;best=auc
    return selected,trace


def frozen_variants(protocol,validation,checkpoint_sha,validation_sha,raw_sha,n_views):
    """Reject changes in weights, data or validation-defined membership."""
    for key,value in [('checkpoint_sha256',checkpoint_sha),('validation_sha256',validation_sha),
                      ('raw_sha256',raw_sha)]:
        if protocol[key]!=value:raise ValueError('Frozen protocol mismatch: '+key)
    if validation['checkpoint_sha256']!=checkpoint_sha or validation['test_evaluated']:
        raise ValueError('Expected validation-only record for the same checkpoint')
    names=set(protocol['variants'])
    allowed=({'all_mean'},{'all_mean','all_max'},{'all_mean','all_max','greedy_mean','greedy_max'})
    if names not in allowed or protocol['primary']!='all_mean':
        raise ValueError('Expected predeclared shared-graph variants and primary all_mean')
    result={}
    for name,ids in protocol['variants'].items():
        if (not ids or len(set(ids))!=len(ids) or
            any(type(i) is not int or not 0<=i<n_views for i in ids)):
            raise ValueError('Invalid frozen view IDs')
        if ids!=validation['variants'][name]['views']:
            raise ValueError('Frozen views differ from validation selection')
        if name.startswith('all_') and ids!=list(range(n_views)):
            raise ValueError('All-view variant must contain every view')
        result[name]=ids
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--raw-path')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--greedy',action='store_true')
    parser.add_argument('--retention-samples',type=int,default=0,help='Validation-only structural description')
    parser.add_argument('--test-protocol',help='Explicit frozen JSON protocol enables test evaluation')
    parser.add_argument('--frozen-validation',help='Completed validation record referenced by test protocol')
    a=parser.parse_args();verify_sources();torch.set_num_threads(1)
    if a.retention_samples<0 or (a.retention_samples and a.test_protocol):
        raise ValueError('Retention samples must be nonnegative and validation-only')
    if bool(a.test_protocol)!=bool(a.frozen_validation):
        raise ValueError('Test protocol and frozen validation record are required together')
    if a.test_protocol and a.greedy:raise ValueError('Cannot select greedy subsets during test evaluation')
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    checkpoint=Path(a.checkpoint).resolve()
    training_spec=json.loads((checkpoint.parent/'run_spec.json').read_text())
    if training_spec.get('contrastive_indexing','released')!='released':
        raise ValueError('Reference reproduction evaluation requires released-training checkpoints')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    args=state['args'];synthetic=training_spec['synthetic']
    if args['erm']:raise ValueError('ENS reference requires SuGAr checkpoints')
    if synthetic:
        rows=[dict(smiles=s,cls_label=i%2) for i,s in enumerate(
            ['CCOC(=O)C','CCN(CC)CC','c1ccccc1CC(=O)NC','CCCOC'])]
    else:
        if not a.raw_path:raise ValueError('--raw-path required for real validation data')
        raw=Path(a.raw_path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=training_spec['source']['raw_sha256']:
            raise ValueError('Dataset hash differs from training source')
        raw_payload=json.loads(raw)
        rows=raw_payload['split']['ood_val']
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(BASE/'upstream'))
    from models import ciga
    install_masks(ciga);models=[];device=torch.device(a.device)
    for weights in state['view_states']:
        model=ciga.CIGA(ratio=args['r'],input_dim=39,edge_dim=10,out_dim=2,
            gnn_type=args['model'],num_layers=args['num_layers'],emb_dim=args['emb_dim'],
            drop_ratio=args['dropout'],graph_pooling=args['pooling'],virtual_node=args['virtual_node'],
            c_dim=args['classifier_emb_dim'],c_in=args['classifier_input_feat'],
            c_rep=args['contrast_rep'],c_pool=args['contrast_pooling'],s_rep=args['spurious_rep'])
        model.load_state_dict(weights);models.append(model.to(device).eval())
    data=convert(rows,feature_module());y=np.array([int(row['cls_label']) for row in rows])
    def prediction(indices,aggregation):
        values=[]
        for batch in DataLoader(data,batch_size=args['batch_size'],shuffle=False):
            values.append(probabilities([models[i] for i in indices],batch.to(device),
                'shared_raw_uniform',aggregation).cpu().numpy())
        p=np.concatenate(values)
        if p.shape!=y.shape or not np.isfinite(p).all():raise ValueError('Invalid predictions')
        return p
    start=time.perf_counter();individual=[]
    for i,expected in enumerate(state['selected_validation_auc']):
        observed=float(roc_auc_score(y,prediction([i],'mean')))
        if abs(observed-expected)>1e-4:raise RuntimeError('Individual validation replay failed')
        individual.append(observed)
    report=dict(scope='Validation only, no training or test evaluation',synthetic=synthetic,
        primary='all_mean',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint=str(checkpoint),individual_validation_auc=individual,molecules=len(y),
        provenance=training_spec,variants={},test_evaluated=False)
    if a.test_protocol:
        protocol_path=Path(a.test_protocol);protocol=json.loads(protocol_path.read_text())
        validation_path=Path(a.frozen_validation);validation_raw=validation_path.read_bytes()
        validation=json.loads(validation_raw)
        variants=frozen_variants(protocol,validation,report['checkpoint_sha256'],
            hashlib.sha256(validation_raw).hexdigest(),training_spec['source'].get('raw_sha256'),len(models))
        # Replay frozen ensemble scores before constructing any test graphs.
        for name,ids in variants.items():
            observed=float(roc_auc_score(y,prediction(ids,name.rsplit('_',1)[1])))
            if abs(observed-validation['variants'][name]['auc'])>1e-4:
                raise RuntimeError('Frozen ensemble validation replay failed: '+name)
        if not synthetic:
            rows=raw_payload['split']['ood_test']
            data=convert(rows,feature_module());y=np.array([int(row['cls_label']) for row in rows])
        report.update(scope='Frozen test evaluation; no training or model selection',
            molecules=len(y),test_evaluated=True,
            test_protocol=protocol,protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            validation_sha256=hashlib.sha256(validation_raw).hexdigest())
        for name,ids in variants.items():
            p=prediction(ids,name.rsplit('_',1)[1])
            np.savez(out/(name+'.npz'),y_true=y,y_pred=p)
            report['variants'][name]=dict(views=ids,auc=float(roc_auc_score(y,p)),
                selection='Frozen from validation; never fitted on test')
        report['wall_seconds']=time.perf_counter()-start
        write(out/'completed.json',report)
        print(json.dumps({name:value['auc'] for name,value in report['variants'].items()},indent=2))
        return
    for aggregation in ('mean','max'):
        all_views=list(range(len(models)));p=prediction(all_views,aggregation)
        name='all_'+aggregation
        np.savez(out/(name+'.npz'),y_true=y,y_pred=p)
        report['variants'][name]=dict(views=all_views,auc=float(roc_auc_score(y,p)))
        write(out/'evaluation.json',report)
        if a.greedy:
            chosen,trace=greedy_select(individual,
                lambda ids:roc_auc_score(y,prediction(ids,aggregation)))
            p=prediction(chosen,aggregation);name='greedy_'+aggregation
            np.savez(out/(name+'.npz'),y_true=y,y_pred=p)
            report['variants'][name]=dict(views=chosen,auc=float(roc_auc_score(y,p)),trace=trace,
                selection='Fit and scored on validation; not an unbiased generalization estimate')
            write(out/'evaluation.json',report)
    if a.retention_samples:
        from ensemble import shared_retention
        report['validation_retention']=shared_retention(models,data,a.retention_samples)
        write(out/'retention.json',report['validation_retention'])
    report['wall_seconds']=time.perf_counter()-start
    write(out/'completed.json',report)
    print(json.dumps({name:value['auc'] for name,value in report['variants'].items()},indent=2))


if __name__=='__main__':main()
