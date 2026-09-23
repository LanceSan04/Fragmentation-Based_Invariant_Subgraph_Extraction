"""Render actual own-view forward traces; evaluation or sampler-only demonstration."""
import argparse, hashlib, html, json
from datetime import datetime
from pathlib import Path
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from train_sugar import Config,build_views_from_feature_dims,strip_non_tensor_batch_fields
from src.training.checkpoint import load_checkpoint,load_view_state_dicts
from src.data.dataloader import build_split_dataloaders
from src.models.fragment_graph import get_fragment_view_value

def drawing(mol,atoms,bonds,png=False):
    d=rdMolDraw2D.MolDraw2DCairo(450,330) if png else rdMolDraw2D.MolDraw2DSVG(450,330)
    d.drawOptions().addAtomIndices=True
    d.DrawMolecule(mol,highlightAtoms=atoms,highlightBonds=bonds)
    d.FinishDrawing()
    return d.GetDrawingText()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('--processed-dir');p.add_argument('--indices',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--mode',choices=['evaluation','sampling-demo'],default='evaluation')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cpu',choices=['cpu','cuda']);p.add_argument('--output-dir')
    a=p.parse_args(argv)
    if min(a.indices)<0 or len(set(a.indices))!=len(a.indices):p.error('Unique nonnegative indices required')
    payload=load_checkpoint(a.checkpoint);values=dict(payload['config']);split=values['validation_split']
    if 'test' in split.lower():raise ValueError('Validation split required')
    values.update(batch_size=1,num_workers=0,loader_split_names=[split])
    if a.processed_dir:values['processed_dir']=a.processed_dir
    cfg=Config(values);views=build_views_from_feature_dims(cfg,payload['feature_dims']);load_view_state_dicts(views,payload)
    if any(v.erm_baseline for v in views):raise ValueError('Fragment checkpoint required')
    torch.manual_seed(a.seed)
    for i,v in enumerate(views):
        v.to(a.device).eval()
        v.fragment_edge_sampler.train(a.mode=='sampling-demo')
        v.fragment_edge_sampler.seed=a.seed+i
        v.fragment_edge_sampler._call_count.zero_()
    loaders=build_split_dataloaders(cfg.processed_dir,cfg);dataset=loaders[split].dataset
    if max(a.indices)>=len(dataset):raise ValueError('Molecule index out of range')
    smiles={i:dataset[i].smiles for i in a.indices}
    strip_non_tensor_batch_fields(loaders)
    out=Path(a.output_dir or f'outputs/assay/visualize_views/{datetime.now():%Y%m%d_%H%M%S_%f}')
    out.mkdir(parents=True,exist_ok=False)
    pages=[]
    with torch.no_grad():
        for index,batch in enumerate(loaders[split]):
            if index not in a.indices:continue
            batch=batch.to(a.device);mol=Chem.MolFromSmiles(smiles[index]);AllChem.Compute2DCoords(mol)
            if mol.GetNumAtoms()!=batch.x.size(0):raise ValueError('Atom mapping mismatch')
            folder=out/f'molecule_{index}';folder.mkdir();(folder/'input.svg').write_text(drawing(mol,[],[]))
            traces=[];rows=[]
            for i,v in enumerate(views):
                pred,t=v(batch,return_intermediates=True)
                eligible=t['fragment_edge_eligible'];selected=t['selected_fragment_mask']
                assert not (selected & ~eligible).any()
                boundary=get_fragment_view_value(batch,'fragment_edge_to_bond_index',v.fragmentation_method)
                def rd_bonds(ids):
                    ans=[]
                    for j in ids:
                        u,w=batch.bond_index[:,j].tolist();b=mol.GetBondBetweenAtoms(u,w)
                        if b is None:raise ValueError('Bond mapping mismatch')
                        ans.append(b.GetIdx())
                    return ans
                sample_bonds=rd_bonds(boundary[eligible].tolist())
                selected_bonds=rd_bonds(t['selected_bond_ids'].tolist());atoms=t['selected_atom_ids'].tolist()
                (folder/f'view{i+1}_sample.svg').write_text(drawing(mol,[],sample_bonds))
                (folder/f'view{i+1}_selected.svg').write_text(drawing(mol,atoms,selected_bonds))
                (folder/f'view{i+1}_selected.png').write_bytes(drawing(mol,atoms,selected_bonds,True))
                trace={k:t[k].detach().cpu().tolist() for k in ['fragment_scores','fragment_edge_eligible','selected_fragment_mask','selected_atom_ids','selected_bond_ids']}
                trace.update(view=i+1,method=v.fragmentation_method,probabilities=pred.exp().cpu().tolist()[0]);traces.append(trace)
                rows.append(f'<tr><td>View {i+1}<br>P(positive)={trace["probabilities"][1]:.4f}</td><td><img src="view{i+1}_sample.svg"></td><td><img src="view{i+1}_selected.svg"></td></tr>')
            (folder/'trace.json').write_text(json.dumps(traces,indent=2))
            mean=sum(t['probabilities'][1] for t in traces)/len(traces)
            page=f'<html><body><h1>Molecule {index}: {a.mode}</h1><p>{html.escape(smiles[index])}</p><img src="input.svg"><p>Highlights show eligible boundary bonds and selected atoms/bonds on the original molecule. Unhighlighted structure is context, not selected input. Scores and exact masks are in trace.json. Own-view mean P(positive): {mean:.4f}. Sampling-demo freezes dropout/BatchNorm; it is not normal inference.</p><table><tr><th>View</th><th>Eligible boundary bonds</th><th>Selected atom graph highlighted</th></tr>{"".join(rows)}</table></body></html>'
            (folder/'index.html').write_text(page);pages.append(f'<li><a href="molecule_{index}/index.html">Molecule {index}</a></li>')
            if index>=max(a.indices):break
    (out/'index.html').write_text('<html><body><h1>Own-view forward comparison</h1><ul>'+''.join(pages)+'</ul></body></html>')
    (out/'manifest.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint),sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),split=split,indices=a.indices,mode=a.mode,seed=a.seed),indent=2))
    print(out/'index.html')

