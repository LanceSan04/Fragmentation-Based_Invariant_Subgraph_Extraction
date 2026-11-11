"""Label-independent inference adapters; not the released label-weighted vote."""
import torch


@torch.no_grad()
def probabilities(models,batch,policy='shared_raw_uniform',aggregation='mean'):
    if not models:raise ValueError('At least one view required')
    if policy not in ('shared_raw_uniform','independent_uniform'):
        raise ValueError('Unknown inference policy')
    if aggregation not in ('mean','max'):raise ValueError('Unknown edge aggregation')
    if len({getattr(model,'ratio',None) for model in models})!=1:
        raise ValueError('Shared top-k requires a common selection ratio')
    for model in models:model.eval()
    shared=None
    if policy=='shared_raw_uniform':
        # Read logits before split_graph can mutate them. No labels accessed.
        scores=[]
        row,col=batch.edge_index
        for model in models:
            h=model.gnn_encoder(batch)
            scores.append(model.edge_att(torch.cat([h[row],h[col]],dim=-1)).view(-1))
        stacked=torch.stack(scores)
        shared=stacked.mean(0) if aggregation=='mean' else stacked.max(0).values
    predictions=[]
    for model in models:
        if shared is None:logits=model(batch)
        else:
            # Released split_graph mutates supplied scores; views must each see
            # the same original aggregate rather than another view's mutations.
            logits=model(batch,subgraph_avg=True,pred_edge_weight=shared.clone())
        predictions.append(logits.softmax(-1)[:,1])
    return torch.stack(predictions).mean(0)


@torch.no_grad()
def shared_retention(models, data, count):
    """Describe the actual primary shared graph on size-stratified validation rows."""
    import numpy as np
    from torch_geometric.data import Batch
    if count < 1 or not data:
        raise ValueError('Positive retention count and nonempty validation data required')
    order=sorted(range(len(data)),key=lambda i:(data[i].num_nodes,i))
    indices=[order[i] for i in np.linspace(0,len(order)-1,min(count,len(order)),dtype=int)]
    device=next(models[0].parameters()).device
    rows=[]
    for index in indices:
        batch=Batch.from_data_list([data[index]]).to(device)
        src,dst=batch.edge_index
        scores=[]
        for model in models:
            model.eval()
            h=model.gnn_encoder(batch)
            scores.append(model.edge_att(torch.cat([h[src],h[dst]],dim=-1)).view(-1))
        mean=torch.stack(scores).mean(0)
        selected,_=models[0].split_graph(batch,mean.clone(),models[0].ratio)
        edges=selected[0].cpu().T.tolist()
        atoms=sorted({a for edge in edges for a in edge})
        bonds={tuple(sorted(edge)) for edge in edges}
        original={tuple(sorted(edge)) for edge in batch.edge_index.cpu().T.tolist()}
        rows.append(dict(index=index,original_atoms=batch.num_nodes,selected_atoms=len(atoms),
            original_bonds=len(original),selected_bonds=len(bonds),
            atom_fraction=len(atoms)/batch.num_nodes,bond_fraction=len(bonds)/len(original),
            atom_ids=atoms,bond_pairs=sorted(bonds)))
    return dict(indices=indices,rows=rows,
        summary={key:dict(mean=float(np.mean([r[key] for r in rows])),
                          min=float(min(r[key] for r in rows)),max=float(max(r[key] for r in rows)))
                 for key in ('atom_fraction','bond_fraction')},
        policy='all_mean shared graph; every classifier uses the same selected structure',
        sample_rule='Evenly spaced ranks by original atom count; descriptive, not a random sample',
        scope='Validation only; structural retention is not evidence of explanation faithfulness')
