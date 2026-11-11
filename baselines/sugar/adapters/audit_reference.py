"""Tiny executable checks of pinned reference behavior; no training or data access."""
import ast
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from compat import BASE, functions_from, verify_sources


def check():
    verify_sources()
    tree=ast.parse((BASE/'upstream/models/ciga.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CIGA')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='split_graph')
    ns={'torch':torch}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'pinned_split_graph','exec'),ns)
    graph=Data(x=torch.arange(4.).reshape(-1,1),batch=torch.zeros(4,dtype=torch.long),
        edge_index=torch.tensor([[0,1,2,3],[1,0,3,2]]),edge_attr=torch.ones(4,1))
    score=torch.tensor([-4.,-3.,-2.,-1.]);original=score.clone()
    first=ns['split_graph'](None,graph,score,.5)[0][0]
    after_first=score.clone()
    second=ns['split_graph'](None,graph,score,.5)[0][0]
    assert not torch.equal(original,after_first)
    assert not torch.equal(first,second)
    scope=functions_from(BASE/'upstream/utils/get_subgraph.py',['relabel'],{'torch':torch})
    edges=torch.tensor([[2,3],[3,2]])
    correct,remapped,_,_=scope['relabel'](graph.x,edges,graph.batch)
    repeated,_,_,_=scope['relabel'](graph.x,remapped,graph.batch)
    assert correct.flatten().tolist()==[2.,3.]
    assert repeated.flatten().tolist()==[0.,1.]
    # Execute the original voting function with fixed predictions and two label
    # assignments. Only labels change; capture its actual aggregate predictions.
    class Batch:
        x=torch.zeros(4,1)
        def __init__(self,y):self.y=torch.tensor(y)
        def to(self,device):return self
    class Model:
        def __init__(self,p):self.logits=torch.stack([torch.zeros(4),torch.tensor(p)],1)
        def eval(self):return self
        def __call__(self,batch,**kwargs):return self.logits
    class Evaluator:
        def eval(self,value):
            self.pred=torch.as_tensor(value['y_pred']).clone()
            return {'rocauc':0.5}
    models=[Model([-2.,-1.,1.,2.]),Model([2.,1.,-1.,-2.])]
    def metric(model,device,loader,evaluator,**kwargs):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(loader[0].y.numpy(),model.logits[:,1].numpy())
    ns=functions_from(BASE/'upstream/ens.py',['soft_vote'],
        {'torch':torch,'F':F,'np':__import__('numpy'),'eval_model':metric})
    votes=[]
    for labels in ([0,0,1,1],[1,1,0,0]):
        evaluator=Evaluator()
        ns['soft_vote'](models,'cpu',[Batch(labels)],evaluator,'rocauc')
        votes.append(evaluator.pred)
    assert not torch.equal(*votes)
    return dict(source_integrity=True,labels_change_vote_with_fixed_predictions=True,
        split_graph_mutates_supplied_scores=True,reusing_scores_changes_selected_edges=True,
        original_atom_ids=[2,3],second_relabel_uses_atom_ids=[0,1],
        scope='Counterexamples establish code behavior, not benchmark effect size or published-result provenance.')


if __name__=='__main__':print(json.dumps(check(),indent=2))
