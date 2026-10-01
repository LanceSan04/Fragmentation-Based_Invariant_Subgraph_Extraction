"""Bounded own-view diagnostics using original atom/bond identities; no training."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Batch
from train_sugar import Config,build_views_from_feature_dims
from src.training.checkpoint import load_checkpoint,load_view_state_dicts
from src.data.dataloader import load_processed_split
from scripts.audit_fragment_structure import topology,describe
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.fragment_masker import reverse_fragment_edge_attr


def overlap(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if a|b else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint');parser.add_argument('--processed-dir',required=True)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--molecules',type=int,default=100)
    parser.add_argument('--orientation-probe',action='store_true')
    args=parser.parse_args()
    if args.molecules<1:parser.error('molecules must be positive')
    torch.set_num_threads(4);torch.manual_seed(17)
    payload=load_checkpoint(args.checkpoint);cfg=Config(payload['config'])
    views=build_views_from_feature_dims(cfg,payload['feature_dims']);load_view_state_dicts(views,payload)
    if any(v.erm_baseline for v in views):raise ValueError('Fragment views required')
    for v in views:v.eval()
    data=load_processed_split(Path(args.processed_dir)/'ood_val.pt')['molecule_data']
    order=sorted(range(len(data)),key=lambda i:(data[i].fragment_edge_index.size(1),i))
    indices=[order[i] for i in np.linspace(0,len(order)-1,min(args.molecules,len(order)),dtype=int)]
    rows=[];pairs=[];molecules=[]
    with torch.no_grad():
        for index in indices:
            original=data[index];d=original.clone()
            for key in ['fragment_to_atoms','smiles','molecule_id','metadata']:
                if key in d:del d[key]
            batch=Batch.from_data_list([d],follow_batch=cfg.follow_batch)
            states=[]
            for i,v in enumerate(views):
                pred,t=v(batch,return_intermediates=True)
                atoms=t['selected_atom_ids'].tolist();bonds=t['selected_bond_ids'].tolist()
                atom_map={a:j for j,a in enumerate(atoms)}
                edges=[(atom_map[a],atom_map[b]) for a,b in original.bond_index[:,bonds].T.tolist()]
                row=dict(index=index,view=i,atom_ids=atoms,bond_ids=bonds,
                         connections=int(t['selected_fragment_mask'].sum()),
                         atom_fraction=len(atoms)/original.x.size(0),
                         bond_fraction=len(bonds)/original.bond_index.size(1) if original.bond_index.size(1) else None,
                         components=topology(len(atoms),edges)['components'],
                         positive_probability=float(pred.exp()[0,1]))
                if args.orientation_probe:
                    # Reverse fragment IDs, then restore canonical src <= dst.
                    # Edge columns still identify the same original bonds.
                    g=build_fragment_graph_batch(batch,method=v.fragmentation_method)
                    r=g.clone();n=g.x.size(0)
                    r.x=g.x.flip(0);r.batch=g.batch.flip(0)
                    r.edge_index=(n-1-g.edge_index).flip(0)
                    r.edge_attr=reverse_fragment_edge_attr(g.edge_attr)
                    a,ma=v.fragment_masker(g);b,mb=v.fragment_masker(r)
                    row['relabel_max_score_delta']=float((a-b).abs().max()) if a.numel() else 0.
                    row['relabel_changed_selected_edges']=int((ma!=mb).sum())
                states.append(row);rows.append(row)
            for a,b in combinations(states,2):
                pairs.append(dict(index=index,a=a['view'],b=b['view'],
                                  atom_jaccard=overlap(a['atom_ids'],b['atom_ids']),
                                  bond_jaccard=overlap(a['bond_ids'],b['bond_ids'])))
            molecules.append(dict(index=index,label=int(original.y.item()),
                                  assay=str(original.metadata.get('assay_id')),
                                  probability=float(np.mean([r['positive_probability'] for r in states]))))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    result=dict(checkpoint=args.checkpoint,checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                checkpoint_config=payload['config'],effective_config=dict(payload['config'],device='cpu',batch_size=1,num_workers=0,processed_dir=str(Path(args.processed_dir).resolve()),loader_split_names=['ood_val']),
                device='cpu',mode='evaluation',indices=indices,orientation_probe=args.orientation_probe,
                sample_rule='Evenly spaced ranks after sorting validation molecules by fragment-connection count; descriptive, not random or a benchmark AUC.',
                summary={k:describe([r[k] for r in rows if r[k] is not None]) for k in ['atom_fraction','bond_fraction','components']},
                overlap_summary={k:describe([r[k] for r in pairs if r[k] is not None]) for k in ['atom_jaccard','bond_jaccard']},
                views=rows,pairs=pairs,molecules=molecules)
    if len(indices)==len(data):
        groups={}
        for row in molecules:groups.setdefault(row['assay'],[]).append(row)
        def auc(group):
            return float(roc_auc_score([r['label'] for r in group],[r['probability'] for r in group]))
        valid={k:g for k,g in groups.items() if len({r['label'] for r in g})==2}
        result['validation_auc']=auc(molecules)
        result['assay_auc']={k:dict(n=len(g),auc=auc(g)) for k,g in valid.items()}
        result['single_class_assays_excluded']=len(groups)-len(valid)
        result['macro_assay_auc']=float(np.mean([auc(g) for g in valid.values()])) if valid else None
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(summary=result['summary'],overlap=result['overlap_summary']),indent=2))


if __name__=='__main__':main()
