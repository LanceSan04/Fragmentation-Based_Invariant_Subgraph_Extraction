"""Validation-only own-view coherence/faithfulness pilot; frozen classifiers/gates.

Graph interventions are not chemically validated molecular counterfactuals.
No training, final-ensemble explanation claim, or benchmark AUC is produced.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from torch_geometric.data import Batch,Data

from src.analysis.fragment_structure import topology
from src.data.dataloader import load_processed_split
from src.models.atom_reconstructor import reconstruct_batch_fast
from src.models.fragment_graph import get_fragment_view_value
from src.training.checkpoint import load_checkpoint,load_view_state_dicts
from src.training.pipeline import Config,build_views_from_feature_dims

PATTERNS={name:Chem.MolFromSmarts(smarts) for name,smarts in {
    'amide':'[CX3](=[OX1])[NX3]','carbonyl':'[CX3]=[OX1]',
    'hydroxyl':'[OX2H]','nitrile':'[CX2]#[NX1]'}.items()}


def induced(graph,keep):
    """Keep atom features and surviving message gates exactly as captured."""
    keep=torch.as_tensor(keep,dtype=torch.bool,device=graph.x.device)
    if keep.numel()!=graph.x.size(0):raise ValueError('Atom mask shape mismatch')
    ids=keep.nonzero(as_tuple=True)[0]
    mapping=keep.new_full((keep.numel(),),-1,dtype=torch.long)
    mapping[ids]=torch.arange(len(ids),device=ids.device)
    edge_keep=keep[graph.edge_index[0]] & keep[graph.edge_index[1]]
    return Data(x=graph.x[ids],edge_index=mapping[graph.edge_index[:,edge_keep]],
        edge_attr=graph.edge_attr[edge_keep],edge_weight=graph.edge_weight[edge_keep],
        batch=graph.batch[ids],selected_original_atom_indices=graph.selected_original_atom_indices[ids])


def geometry(graph):
    n=graph.x.size(0)
    edges={tuple(sorted(pair)) for pair in graph.edge_index.T.tolist()}
    return dict(atoms=n,bonds=len(edges),components=topology(n,edges)['components'])


def chemical_metrics(mol,atoms,bonds):
    atoms=set(atoms);bonds={tuple(sorted(pair)) for pair in bonds}
    groups=[('ring',tuple(r)) for r in mol.GetRingInfo().AtomRings()]
    for name,pattern in PATTERNS.items():groups.extend((name,match) for match in mol.GetSubstructMatches(pattern))
    counts={}
    for name,members in groups:
        members=set(members);touched=bool(members & atoms)
        required={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in mol.GetBonds()
                  if b.GetBeginAtomIdx() in members and b.GetEndAtomIdx() in members}
        complete=members<=atoms and required<=bonds
        entry=counts.setdefault(name,dict(total=0,touched=0,complete=0,cut=0))
        entry['total']+=1;entry['touched']+=int(touched);entry['complete']+=int(complete)
        entry['cut']+=int(touched and not complete)
    return counts


def aligned_molecule(original):
    mol=Chem.MolFromSmiles(original.smiles)
    if mol is None or mol.GetNumAtoms()!=original.x.size(0):return None
    actual={tuple(sorted(pair)) for pair in original.bond_index.T.tolist()}
    expected={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in mol.GetBonds()}
    return mol if actual==expected else None


def describe_graph(graph,mol=None):
    row=geometry(graph);ids=graph.selected_original_atom_indices.tolist();row['atom_ids']=ids
    edges={tuple(sorted((ids[a],ids[b]))) for a,b in graph.edge_index.T.tolist()}
    row['bond_atom_pairs']=sorted(edges)
    row['chemical_groups']=chemical_metrics(mol,ids,edges) if mol is not None else None
    return row


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--processed-dir',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--molecules',type=int,default=20);p.add_argument('--random-draws',type=int,default=64)
    p.add_argument('--controls',type=int,default=8);p.add_argument('--max-fragments',type=int,default=3)
    p.add_argument('--seed',type=int,default=1841);p.add_argument('--visual-molecules',type=int,default=5)
    a=p.parse_args(argv)
    if min(a.molecules,a.random_draws,a.controls,a.max_fragments)<1:raise ValueError('Positive pilot budgets required')
    torch.set_num_threads(1);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    payload=load_checkpoint(a.checkpoint);cfg=Config(payload['config'])
    split=cfg.get('validation_split','ood_val')
    if not split or 'test' in split.lower():raise ValueError('Validation split required')
    views=build_views_from_feature_dims(cfg,payload['feature_dims']);load_view_state_dicts(views,payload)
    if any(v.erm_baseline or v.full_graph_gating or v.selection_mode!='learned' for v in views):
        raise ValueError('Pilot requires cropped learned-score views; random ranking with learned gates is allowed')
    for view in views:view.to(a.device).eval()
    cache=Path(a.processed_dir)/(split+'.pt');cached=load_processed_split(cache)
    if cached['metadata']['dataset']!=cfg.get('dataset'):raise ValueError('Checkpoint/cache dataset mismatch')
    data=cached['molecule_data']
    order=sorted(range(len(data)),key=lambda i:(data[i].x.size(0),i))
    indices=[order[i] for i in np.linspace(0,len(order)-1,min(a.molecules,len(order)),dtype=int)]
    visual_positions=set(np.linspace(0,len(indices)-1,min(max(0,a.visual_molecules),len(indices)),dtype=int).tolist())
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);records=[];pictures=[]
    def predict(view,graph,target):
        return float(view.atom_classifier(graph).softmax(-1)[0,target])
    with torch.no_grad():
        for position,index in enumerate(indices):
            original=data[index];mol=aligned_molecule(original);d=original.clone()
            for key in ['fragment_to_atoms','smiles','molecule_id','metadata']:
                if key in d:del d[key]
            batch=Batch.from_data_list([d],follow_batch=cfg.follow_batch).to(a.device)
            for vi,view in enumerate(views):
                captured=[]
                hook=view.atom_classifier.register_forward_pre_hook(lambda module,args:captured.append(args[0].clone()))
                try:prediction,info=view(batch,return_intermediates=True)
                finally:hook.remove()
                if len(captured)!=1:raise RuntimeError('Unexpected classifier calls')
                graph=captured[0];target=int(prediction.argmax(-1)[0]);baseline=float(prediction.exp()[0,target])
                if abs(predict(view,graph,target)-baseline)>1e-5:raise RuntimeError('Captured graph replay mismatch')
                row=dict(index=index,view=vi,target_class=target,baseline_probability=baseline,
                    label=int(original.y.item()),baseline_correct=target==int(original.y.item()),
                    molecule_id=str(getattr(original,'molecule_id',index)),
                    original_atoms=original.x.size(0),original_bonds=original.bond_index.size(1),
                    selected=describe_graph(graph,mol),random_alternatives=[],random_atom_nulls=[],ablations=[])
                # Random boundary-edge masks with the same edge-count setting.
                # Filter on actual atom/bond budgets, never on prediction quality.
                mask=info['selected_fragment_mask'];scores=info['fragment_scores'].detach()
                eligible=info['fragment_edge_eligible'].nonzero(as_tuple=True)[0].cpu().numpy()
                k=int(mask.sum());seen={tuple(mask.nonzero(as_tuple=True)[0].tolist())}
                atom_tol=max(1,int(np.ceil(original.x.size(0)*.05)))
                bond_tol=max(1,int(np.ceil(original.bond_index.size(1)*.05)))
                weights=None if view.message_gate_mode=='unit' else scores
                if k<=len(eligible) and k>0:
                    for _ in range(a.random_draws):
                        chosen=tuple(sorted(rng.choice(eligible,k,replace=False).tolist()))
                        if chosen in seen:continue
                        seen.add(chosen);other=torch.zeros_like(mask);other[list(chosen)]=True
                        candidate=reconstruct_batch_fast(batch,other,selected_fragment_edge_weight=weights,
                            fragmentation_method=view.fragmentation_method)
                        g=geometry(candidate)
                        if not g['atoms']:continue
                        da=g['atoms']-row['selected']['atoms'];db=g['bonds']-row['selected']['bonds']
                        if abs(da)>atom_tol or abs(db)>bond_tol:continue
                        detail=describe_graph(candidate,mol)
                        detail.update(atom_budget_error=da,bond_budget_error=db,
                            probability=predict(view,candidate,target))
                        row['random_alternatives'].append(detail)
                        if len(row['random_alternatives'])>=a.controls:break
                # Structural null: random atoms, same atom count and similar
                # bond count. Not a separately trained atom-level model.
                full=Data(x=batch.x,edge_index=batch.edge_index,edge_attr=batch.edge_attr,
                    edge_weight=batch.x.new_ones(batch.edge_index.size(1)),batch=batch.batch,
                    selected_original_atom_indices=torch.arange(batch.x.size(0),device=batch.x.device))
                seen_atoms={tuple(sorted(graph.selected_original_atom_indices.tolist()))}
                for _ in range(a.random_draws):
                    chosen=tuple(sorted(rng.choice(full.x.size(0),graph.x.size(0),replace=False).tolist()))
                    if chosen in seen_atoms:continue
                    seen_atoms.add(chosen);keep=torch.zeros(full.x.size(0),dtype=torch.bool,device=full.x.device)
                    keep[list(chosen)]=True;candidate=induced(full,keep);g=geometry(candidate)
                    if abs(g['bonds']-row['selected']['bonds'])>bond_tol:continue
                    detail=describe_graph(candidate,mol)
                    detail.update(atom_budget_error=0,bond_budget_error=g['bonds']-row['selected']['bonds'])
                    row['random_atom_nulls'].append(detail)
                    if len(row['random_atom_nulls'])>=a.controls:break
                # Ablate high-scoring retained fragments from the captured graph;
                # leave all surviving features/gates fixed and do not reselect.
                mapping=get_fragment_view_value(batch,'atom_to_fragment',view.fragmentation_method)
                edge=get_fragment_view_value(batch,'fragment_edge_index',view.fragmentation_method)
                selected_ids=graph.selected_original_atom_indices
                retained=mapping[selected_ids];priorities=[]
                for fragment in retained.unique().tolist():
                    incident=(edge==fragment).any(0)
                    importance=float(scores[incident].max()) if incident.any() else 0.
                    priorities.append((importance,fragment))
                for importance,fragment in sorted(priorities,reverse=True)[:a.max_fragments]:
                    keep=retained!=fragment;removed=int((~keep).sum())
                    entry=dict(fragment=fragment,incident_score=importance,removed_atoms=removed,controls=[])
                    if not keep.any():
                        entry['status']='untestable_empty_remainder';row['ablations'].append(entry);continue
                    changed=induced(graph,keep);g=geometry(changed)
                    entry.update(status='evaluated',remaining=describe_graph(changed,mol),
                        probability_drop=baseline-predict(view,changed,target))
                    for _ in range(a.random_draws):
                        deleted=rng.choice(len(keep),removed,replace=False)
                        control_keep=torch.ones_like(keep);control_keep[deleted.tolist()]=False
                        if torch.equal(control_keep,keep):continue
                        control=induced(graph,control_keep);cg=geometry(control)
                        if abs(cg['bonds']-g['bonds'])>1:continue
                        entry['controls'].append(dict(probability_drop=baseline-predict(view,control,target),
                            removed_atoms=removed,bond_budget_error=cg['bonds']-g['bonds'],
                            components=cg['components']))
                        if len(entry['controls'])>=a.controls:break
                    row['ablations'].append(entry)
                records.append(row)
                if position in visual_positions and mol is not None:
                    directory=out/f'molecule{index}_view{vi}';directory.mkdir()
                    Draw.MolToFile(mol,str(directory/'input.svg'),size=(520,300))
                    pictures.append(f'<h2>Molecule {index}, view {vi}</h2><p>Input molecule</p><img src="{directory.name}/input.svg">')
                    visual_cases=[('selected',row['selected'])]
                    if row['random_alternatives']:visual_cases.append(('random_fragment_matched',row['random_alternatives'][0]))
                    if row['random_atom_nulls']:visual_cases.append(('random_atom_matched',row['random_atom_nulls'][0]))
                    for name,detail in visual_cases:
                        bondset={tuple(pair) for pair in detail['bond_atom_pairs']}
                        bondids=[b.GetIdx() for b in mol.GetBonds() if tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) in bondset]
                        Draw.MolToFile(mol,str(directory/(name+'.svg')),size=(520,300),
                            highlightAtoms=detail['atom_ids'],highlightBonds=bondids)
                        pictures.append(f'<p>{html.escape(name)}: {detail["atoms"]} atoms, {detail["bonds"]} bonds, {detail["components"]} components; highlighted retained atoms/bonds</p><img src="{directory.name}/{name}.svg">')
    report=dict(scope='Validation-only, own-view frozen-classifier pilot; not ensemble fidelity or biological causality',
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),cache_metadata=cached['metadata'],checkpoint_config=payload['config'],
        effective_arguments=vars(a),sample_rule='Evenly spaced atom-count ranks; descriptive, not random or benchmark AUC',
        indices=indices,test_evaluated=False,training=False,records=records,
        summary=dict(molecule_views=len(records),with_matched_random=sum(bool(r['random_alternatives']) for r in records),
            with_atom_null=sum(bool(r['random_atom_nulls']) for r in records),
            evaluated_ablations=sum(e['status']=='evaluated' for r in records for e in r['ablations']),
            ablations_with_controls=sum(bool(e['controls']) for r in records for e in r['ablations'])),
        limitations=['Fixed original scorer context/gates; classifier-only interventions',
            'Random alternatives are fragment-based and also preserve fragments by construction',
            'Random atom deletion can cut rings or create unrealistic graphs; connectivity is recorded',
            'Controls are approximately size-matched; unmatched cases are retained',
            'SMARTS group counts are predefined structural diagnostics, not ground-truth rationales'])
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Selection pilot</title><h1>Validation own-view selections</h1><p>Highlighted parts are retained. Graph interventions are not biological counterfactuals.</p>'+''.join(pictures),encoding='utf-8')
    return report


if __name__=='__main__':main()
