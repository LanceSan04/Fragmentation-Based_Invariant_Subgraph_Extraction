"""Execute the released SuGAr training body with explicit data/reporting adapters.

This entry point is a VALIDATION-ONLY pilot, not a paper-result reproduction.
No ensemble voting/test evaluation is provided until released-code ambiguities
and label-dependent ROC-AUC voting are reconciled with the paper.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version,PackageNotFoundError
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from compat import BASE,verify_sources,feature_module,functions_from,install_masks,seed_function,install_contrastive_indexing

PROJECT=BASE.parents[1]
DATASETS=tuple('drugood_lbap_core_'+endpoint+'_'+split
               for endpoint in ('ec50','ki') for split in ('assay','scaffold','size'))


def dataset_identity(raw_path, requested=None):
    """Keep dataset metadata accurate without changing released model behavior."""
    inferred='drugood_'+Path(raw_path).stem if raw_path else None
    if requested is not None and requested not in DATASETS:
        raise ValueError('Unsupported DrugOOD dataset identity')
    if inferred in DATASETS:
        if requested is not None and requested!=inferred:
            raise ValueError('Dataset identity disagrees with raw filename')
        return inferred
    return requested or 'drugood_lbap_core_ec50_assay'


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def convert(rows,features):
    result=[]
    for index,row in enumerate(rows):
        graph=features.smile2graph(row['smiles'])
        if graph is None:raise ValueError(f'Official featurization failed at row {index}')
        source,target=graph.edges()
        if not len(source):raise ValueError('Released top-k requires at least one edge; no silent removal')
        label=row['cls_label']
        if label not in (0,1):raise ValueError('Binary labels required')
        result.append(Data(x=graph.ndata['x'],edge_attr=graph.edata['x'],
            edge_index=torch.stack([source,target]),y=torch.tensor([int(label)]),
            idx=torch.tensor([index]),node_label=torch.zeros(graph.num_nodes()),
            group=torch.tensor([int(row.get('domain_id',0))])))
    return result


def original_parser():
    tree=ast.parse((BASE/'upstream/main.py').read_text())
    block=next(n for n in tree.body if isinstance(n,ast.If) and
               any(isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='parser' for t in x.targets) for x in n.body))
    nodes=[]
    for node in block.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in node.targets):break
        nodes.append(node)
    scope=dict(argparse=argparse)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream_parser','exec'),scope)
    return scope['parser']


class ReportingAdapter(ast.NodeTransformer):
    """Assert patch counts; never rewrite forward, loss or optimizer statements."""
    def __init__(self,reference_test=False):
        self.reference_test=reference_test
        self.patches=dict(seed_literal=0,data_branch=0,test_metric=0,selected_epoch=0,save=0,validation_observer=0)

    def visit_Assign(self,node):
        text=ast.unparse(node)
        if text=='args.seed = eval(args.seed)':
            self.patches['seed_literal']+=1
            return ast.parse('args.seed = json.loads(args.seed)').body[0]
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='test_acc':
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='eval_model':
                self.patches['test_metric']+=1
                if self.reference_test:return self.generic_visit(node)
                return ast.parse('test_acc = deferred_test_metric()').body[0]
        if text.startswith('best_weights[i] = deepcopy('):
            self.patches['selected_epoch']+=1
            return [node,*ast.parse('selected_epochs[i] = epoch + 1').body]
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='val_acc':
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='eval_model':
                self.patches['validation_observer']+=1
                return [node,*ast.parse('observe_validation(i, epoch, val_acc)').body]
        return self.generic_visit(node)

    def visit_If(self,node):
        test=ast.unparse(node.test)
        if test=="args.dataset.lower().startswith('drugood')":
            self.patches['data_branch']+=1
            node.body=ast.parse('train_loader, valid_loader, test_loader, evaluator = adapter_loaders(args)\ninput_dim = 39\nedge_dim = 10\nnum_classes = 2').body
        if test=='args.save_model' and any(isinstance(x,ast.For) for x in node.body):
            self.patches['save']+=1
            return ast.parse('save_views(models, best_weights, last_val_acc, args)').body[0]
        return self.generic_visit(node)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-path')
    p.add_argument('--dataset',choices=DATASETS,help='Explicit source identity; standard raw filenames are checked')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--views',type=int,default=10)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--contrast',type=float,default=4.)
    p.add_argument('--diversity-penalty',type=float,default=1.)
    p.add_argument('--sample-rate',type=int,choices=[85,90,95],default=95)
    p.add_argument('--contrastive-indexing',choices=['released','corrected'],default='released')
    p.add_argument('--report-ensemble',action='store_true',help='Validation-only label-independent shared/independent voting')
    p.add_argument('--reference-checkpoint',help='Previously trained released checkpoint for matched validation-only inference')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--reference-test-evaluation',action='store_true',help='Synthetic-only parity comparison with original per-epoch test passes')
    a=p.parse_args()
    dataset=dataset_identity(a.raw_path,a.dataset)
    if a.reference_test_evaluation and not a.smoke:
        raise ValueError('Reference test evaluation is allowed only on synthetic fixtures')
    if a.reference_checkpoint and not a.report_ensemble:
        raise ValueError('Reference checkpoint requires --report-ensemble')
    if a.epochs<1 or a.views<2:raise ValueError('Positive epochs and >=2 views required')
    if not math.isfinite(a.contrast) or a.contrast<0:raise ValueError('Invalid contrast coefficient')
    if not math.isfinite(a.diversity_penalty) or a.diversity_penalty<0:raise ValueError('Invalid diversity penalty')
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    verify_sources();torch.set_num_threads(1)
    sys.path.insert(0,str(BASE/'upstream'))
    sys.path.insert(1,str(PROJECT))
    from models import ciga
    from models.losses import get_contrast_loss,get_irm_loss
    from ogb.graphproppred import Evaluator
    install_masks(ciga)
    correction=install_contrastive_indexing(ciga) if a.contrastive_indexing=='corrected' else None
    features=feature_module()
    if a.smoke:
        smiles=['CCOC(=O)C','CCN(CC)CC','c1ccccc1CC(=O)NC','CCCOC']
        records={split:[dict(smiles=s,cls_label=i%2) for i,s in enumerate(smiles)] for split in ['train','ood_val']}
        source=dict(synthetic=True)
    else:
        if not a.raw_path:raise ValueError('--raw-path required')
        payload=json.loads(Path(a.raw_path).read_text())
        records={split:payload['split'][split] for split in ['train','ood_val']}
        import hashlib
        source=dict(raw_path=str(Path(a.raw_path).resolve()),raw_sha256=hashlib.sha256(Path(a.raw_path).read_bytes()).hexdigest(),
                    source_cfg=payload.get('cfg'),split_counts={k:len(v) for k,v in records.items()})
    data={name:convert(rows,features) for name,rows in records.items()}
    parser=original_parser()
    args=parser.parse_args(['--dataset',dataset,'--eval_metric','rocauc',
        '--epoch',str(a.epochs),'--seed',json.dumps(list(range(1,a.views+1))),'--init',str(a.seed),
        '--batch_size','4' if a.smoke else '128','--num_layers','2' if a.smoke else '3',
        '--emb_dim','8' if a.smoke else '32','-c_dim','8' if a.smoke else '32',
        '-c_in','feat','-c_rep','feat','--dropout','0.1','--lr','0.001','--contrast',str(a.contrast),
        '--contrast_t','1','--sample_rate',str(a.sample_rate),'--penalty',str(a.diversity_penalty),'--save_model','--no_tqdm'])
    if a.device=='cpu':
        # Released main selects CUDA whenever available. CPU invocation must not
        # silently occupy a GPU; reject rather than monkeypatch torch availability.
        if torch.cuda.is_available():raise RuntimeError('CPU requested on CUDA host; hide GPUs explicitly')
    selected_epochs=[None]*a.views;history=[];captured={}
    def adapter_loaders(options):
        return (DataLoader(data['train'],batch_size=options.batch_size,shuffle=True),
                DataLoader(data['ood_val'],batch_size=options.batch_size,shuffle=False),
                DataLoader(data['ood_val'],batch_size=options.batch_size,shuffle=False) if a.reference_test_evaluation else None,
                Evaluator('ogbg-molhiv'))
    def deferred_test_metric():
        # Original unshuffled test DataLoader iterator consumes one RNG seed.
        # Preserve that draw without constructing/evaluating a test graph.
        torch.empty((),dtype=torch.int64).random_()
        return float('nan')
    def observe_validation(view,epoch,auc):
        if not math.isfinite(auc):raise RuntimeError('Nonfinite validation AUC')
        history.append(dict(view=view,epoch=epoch+1,auc=float(auc)))
        write(out/'history.json',history)
    def save_views(models,best_weights,aucs,options):
        if any(w is None for w in best_weights):raise RuntimeError('Some views have no selected checkpoint')
        cpu_weights=[{k:v.detach().cpu() for k,v in w.items()} for w in best_weights]
        torch.save(dict(view_states=cpu_weights,selected_epochs=selected_epochs,selected_validation_auc=aucs,
                        args=vars(options),contrastive_indexing=a.contrastive_indexing),out/'best_views.pt')
        captured['models']=models;captured['weights']=cpu_weights;captured['auc']=aucs
    namespace=dict(torch=torch,np=np,random=random,F=F,os=os,datetime=datetime,deepcopy=deepcopy,
        tqdm=tqdm,Data=Data,DataLoader=DataLoader,Evaluator=Evaluator,json=json,
        CIGA=ciga.CIGA,GNNERM=ciga.GNNERM,GNNPooling=ciga.GNNPooling,
        get_contrast_loss=get_contrast_loss,get_irm_loss=get_irm_loss,set_seed=seed_function(),
        adapter_loaders=adapter_loaders,deferred_test_metric=deferred_test_metric,
        observe_validation=observe_validation,selected_epochs=selected_epochs,save_views=save_views)
    tree=ast.parse((BASE/'upstream/main.py').read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['main','eval_model','delete_edge']]
    transformer=ReportingAdapter(a.reference_test_evaluation)
    module=transformer.visit(ast.Module(body=functions,type_ignores=[]));ast.fix_missing_locations(module)
    if transformer.patches!={k:1 for k in transformer.patches}:
        raise ValueError('Unexpected upstream shape; inspect patches: '+str(transformer.patches))
    exec(compile(module,'pinned_sugar_with_reporting_adapter','exec'),namespace)
    packages={}
    for name in ['torch','torch-geometric','ogb','rdkit','dgl','numpy','scikit-learn']:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    write(out/'run_spec.json',dict(args=vars(args),source=source,synthetic=a.smoke,device=a.device,
        packages=packages,python=sys.version,ast_patches=transformer.patches,
        contrastive_indexing=a.contrastive_indexing,training_correction=correction,
        report_ensemble=a.report_ensemble,
        synthetic_reference_test_passes=a.reference_test_evaluation,
        test_evaluated=False,scope='Validation-only released-training-body compatibility pilot, not a complete paper reproduction'))
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();namespace['main'](args)
    # Restore each released per-view best snapshot and verify its validation AUC.
    replay=[]
    for i,(model,state,expected) in enumerate(zip(captured['models'],captured['weights'],captured['auc'])):
        model.load_state_dict(state)
        _,loader,_,evaluator=adapter_loaders(args)
        observed=namespace['eval_model'](model,torch.device(a.device),loader,evaluator,'rocauc')
        delta=float(observed-expected)
        if abs(delta)>1e-4:raise RuntimeError('Upstream view validation replay failed')
        replay.append(dict(view=i,epoch=selected_epochs[i],auc=float(observed),delta=delta))
    if a.report_ensemble:
        from ensemble import probabilities
        from sklearn.metrics import roc_auc_score
        def report(prefix):
            metrics={}
            for policy in ('shared_raw_uniform','independent_uniform'):
                labels=[];predictions=[]
                loader=DataLoader(data['ood_val'],batch_size=args.batch_size,shuffle=False)
                for batch in loader:
                    batch=batch.to(torch.device(a.device))
                    predictions.append(probabilities(captured['models'],batch,policy).cpu().numpy())
                    labels.append(batch.y.cpu().numpy())
                y=np.concatenate(labels);p=np.concatenate(predictions)
                np.savez(out/(prefix+'validation_'+policy+'.npz'),y_true=y,y_pred=p)
                metrics[policy]=dict(auc=float(roc_auc_score(y,p)),molecules=len(y))
            write(out/(prefix+'validation_ensemble.json'),dict(metrics=metrics,test_evaluated=False,
                selection='All views, each restored to its own best validation epoch',
                note='Documented label-independent inference adaptation, not unchanged released voting'))
        report('')
        if a.reference_checkpoint:
            reference_path=Path(a.reference_checkpoint).resolve()
            reference_spec=json.loads((reference_path.parent/'run_spec.json').read_text())
            if reference_spec['source']!=source:raise ValueError('Reference dataset provenance mismatch')
            reference=torch.load(reference_path,map_location='cpu',weights_only=False)
            if reference['args']!=vars(args):raise ValueError('Reference training recipe mismatch')
            if len(reference['view_states'])!=a.views:raise ValueError('Reference view count mismatch')
            reference_replay=[]
            for model,state,expected in zip(captured['models'],reference['view_states'],reference['selected_validation_auc']):
                model.load_state_dict(state)
                _,loader,_,evaluator=adapter_loaders(args)
                observed=namespace['eval_model'](model,torch.device(a.device),loader,evaluator,'rocauc')
                if abs(observed-expected)>1e-4:raise RuntimeError('Reference validation replay failed')
                reference_replay.append(dict(auc=float(observed),delta=float(observed-expected)))
            import hashlib
            write(out/'reference_replay.json',dict(checkpoint=str(reference_path),
                sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),views=reference_replay))
            report('reference_')
    elapsed=time.perf_counter()-start
    write(out/'completed.json',dict(epochs=a.epochs,views=a.views,wall_seconds=elapsed,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if a.device=='cuda' else None,
        replay=replay,test_evaluated=False,synthetic=a.smoke))


if __name__=='__main__':
    main()
