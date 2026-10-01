"""Validation-only, frozen-weight interventions with exactly matched atom sets."""
import argparse
import json
from pathlib import Path
import statistics

import torch

from scripts.audit_validation_groups import group_report
from src.data.dataloader import build_split_dataloaders
from src.experiments.assay_followup import digest, now, restore, require_parity
from src.experiments.common import set_all_seeds
from src.experiments.selection import verify_study_cache, write
from src.models.atom_reconstructor import reconstruct_batch_fast
from src.models.fragment_graph import get_fragment_view_value
from train_sugar import Config, strip_non_tensor_batch_fields


def close_selected_bonds(batch, trace, method):
    """Restore original boundary bonds whose atoms are already selected."""
    atoms=trace['selected_atom_ids']
    present=torch.zeros(batch.x.size(0),dtype=torch.bool,device=batch.x.device)
    present[atoms]=True
    mapping=get_fragment_view_value(batch,'fragment_edge_to_bond_index',method)
    ends=batch.bond_index[:,mapping]
    mask=trace['selected_fragment_mask'] | (present[ends[0]] & present[ends[1]])
    result=reconstruct_batch_fast(batch,mask,selected_fragment_edge_weight=trace['fragment_scores'],fragmentation_method=method)
    if not torch.equal(result.selected_original_atom_indices,atoms):
        raise AssertionError('Closure changed selected atoms')
    if not torch.isin(trace['selected_bond_ids'],result.selected_original_bond_indices).all():
        raise AssertionError('Closure removed an original selected bond')
    return result


def random_bonds_same_budget(induced, original, batch, seed, view, draw, offset):
    """Uniform bond subset on fixed atoms, exact per-molecule original bond counts.

    This is a bond-topology diagnostic, not the random-fragment training baseline.
    The directed-edge layout is verified before subsetting. All atoms, including
    those isolated by the draw, remain in the classifier's pooling operation.
    """
    bonds=induced.selected_original_bond_indices
    n=bonds.numel()
    if induced.edge_index.size(1)!=2*n: raise AssertionError('Unexpected edge layout')
    if not torch.equal(induced.edge_index[:,n:],induced.edge_index[:,:n].flip(0)):
        raise AssertionError('Reverse edges do not align')
    bond_graph=batch.batch[batch.bond_index[0,bonds]].cpu()
    original_graph=batch.batch[batch.bond_index[0,original.selected_original_bond_indices]]
    counts=torch.bincount(original_graph,minlength=batch.num_graphs).cpu()
    keep=torch.zeros(n,dtype=torch.bool)
    for g in range(batch.num_graphs):
        candidates=(bond_graph==g).nonzero(as_tuple=True)[0]
        k=int(counts[g])
        if k>candidates.numel():raise AssertionError('Insufficient induced bonds for matched control')
        generator=torch.Generator().manual_seed(seed*100000000+view*1000000+draw*100000+offset+g)
        keep[candidates[torch.randperm(candidates.numel(),generator=generator)[:k]]]=True
    keep=keep.to(bonds.device)
    directed=torch.cat([keep,keep])
    result=induced.clone()
    result.edge_index=induced.edge_index[:,directed]
    result.edge_attr=induced.edge_attr[directed]
    result.edge_weight=induced.edge_weight[directed]
    result.selected_original_bond_indices=bonds[keep]
    new_graph=batch.batch[batch.bond_index[0,result.selected_original_bond_indices]]
    if not torch.equal(torch.bincount(new_graph,minlength=batch.num_graphs).cpu(),counts):
        raise AssertionError('Random control changed bond budget')
    return result


def run_seed(checkpoint, cache, device, seed, output, expected, smoke=False):
    set_all_seeds(seed)
    config,views=restore(checkpoint,cache,device)
    if any(v.erm_baseline or v.selection_mode!='learned' for v in views):
        raise ValueError('Learned fragment model required')
    config=Config(dict(config.to_dict(),loader_split_names=['ood_val']))
    loaders=build_split_dataloaders(cache,config)
    metadata=[d.metadata for d in loaders['ood_val'].dataset]
    if not smoke and any('assay_id' not in m for m in metadata):raise ValueError('Assay metadata missing')
    assays=[str(m.get('assay_id',m.get('domain_id','synthetic'))) for m in metadata]
    strip_non_tensor_batch_fields(loaders)
    modes=['current','induced','random0','random1','random2']
    rows={m:[] for m in modes}
    changed_molecules=set();offset=0
    with torch.no_grad():
        for batch in loaders['ood_val']:
            batch=batch.to(device);probabilities={m:[] for m in modes}
            for number,view in enumerate(views):
                pred,trace=view(batch,return_intermediates=True)
                original=reconstruct_batch_fast(batch,trace['selected_fragment_mask'],
                    selected_fragment_edge_weight=trace['fragment_scores'],fragmentation_method=view.fragmentation_method)
                induced=close_selected_bonds(batch,trace,view.fragmentation_method)
                new_bonds=induced.selected_original_bond_indices[~torch.isin(induced.selected_original_bond_indices,original.selected_original_bond_indices)]
                changed_molecules.update(offset+int(g) for g in batch.batch[batch.bond_index[0,new_bonds]].unique())
                probabilities['current'].append(pred.exp()[:,1])
                probabilities['induced'].append(view.atom_classifier(induced).softmax(-1)[:,1])
                for draw in range(3):
                    random_graph=random_bonds_same_budget(induced,original,batch,seed,number,draw,offset)
                    probabilities[f'random{draw}'].append(view.atom_classifier(random_graph).softmax(-1)[:,1])
            labels=batch.y.reshape(-1).cpu().tolist()
            for mode in modes:
                probs=torch.stack(probabilities[mode]).mean(0).cpu().tolist()
                for j,(label,prob) in enumerate(zip(labels,probs)):
                    rows[mode].append(dict(index=offset+j,assay=assays[offset+j],label=int(label),probability=prob))
            offset+=len(labels)
    metrics={m:group_report(rows[m]) for m in modes}
    require_parity(metrics['current']['pooled_auc'],expected)
    result=dict(seed=seed,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
                expected_validation_auc=expected,metrics=metrics,changed_molecules=len(changed_molecules),
                num_molecules=offset,config=config.to_dict())
    write(output/'metrics.json',result)
    write(output/'predictions.json',rows)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference-dir','processed-dir','output-dir']:p.add_argument('--'+name,required=True)
    p.add_argument('--raw-path')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args(argv);torch.set_num_threads(4)
    root,cache,out=[Path(x).resolve() for x in [a.reference_dir,a.processed_dir,a.output_dir]]
    if out.exists():raise FileExistsError(out)
    if not a.smoke and not a.raw_path:p.error('Real study requires raw-path')
    verify_study_cache(cache,a.raw_path)
    seeds=[5001] if a.smoke else list(range(8301,8306))
    runs=json.loads((root/'summary.json').read_text())['runs']
    if sorted(r['seed'] for r in runs)!=seeds:raise ValueError('Reference seed mismatch')
    records=[]
    for r in runs:
        path=root/f"seed{r['seed']}"/'best_independent.pt'
        cfg=json.loads((path.parent/'config.json').read_text())
        if not a.smoke and (cfg.get('scoring_mode')!='ordered' or cfg['lambda_contrast']!=.1 or cfg.get('contrast_scope','pooled')!='pooled'):
            raise ValueError('Wrong frozen reference')
        records.append(dict(seed=r['seed'],checkpoint=str(path),sha256=digest(path),
                            expected=r['results']['independent']['independent']['ood_val']['auc']))
    out.mkdir(parents=True)
    write(out/'plan.json',dict(created_at=now(),records=records,synthetic=a.smoke,
        validation_sha256=digest(cache/'ood_val.pt'),test_loaded=False,training_runs=0,
        hypothesis='Missing bonds between retained atoms may impair frozen-model validation predictions.',
        modes=['current','induced','random0','random1','random2'],
        control='Random modes keep EXACT same selected atoms and per-molecule bond count as current; sample bonds from induced graph. Existing per-bond gate values retained.',
        caveat='Induced mode adds bonds, so not bond-budget matched. Random modes change bond identity, gate distribution and connectivity together. This is not a random-fragment baseline or retraining result. Three draws are reported separately, not extra ensemble members.'))
    results=[]
    for r in records:
        folder=out/f"seed{r['seed']}";folder.mkdir()
        results.append(run_seed(r['checkpoint'],cache,a.device,r['seed'],folder,r['expected'],a.smoke))
        write(out/'progress.json',results)
        print(r['seed'],{m:x['pooled_auc'] for m,x in results[-1]['metrics'].items()},flush=True)
    summary={mode:dict(mean=statistics.mean(r['metrics'][mode]['pooled_auc'] for r in results),
                      sample_std=statistics.stdev(r['metrics'][mode]['pooled_auc'] for r in results) if len(results)>1 else None)
             for mode in ['current','induced','random0','random1','random2']}
    write(out/'summary.json',dict(validation_auc=summary,runs=results))
    write(out/'complete.json',dict(complete=True,finished_at=now(),test_loaded=False,training_runs=0))
