import importlib.util
import copy
from pathlib import Path
import subprocess
import sys
import unittest


class PaperInferenceTests(unittest.TestCase):
    def test_test_protocol_rejects_changed_data_weights_and_membership(self):
        path=str(Path(__file__).resolve().parents[1]/'baselines/sugar/adapters');sys.path.insert(0,path)
        try:
            spec=importlib.util.spec_from_file_location('frozen_evaluator',Path(path)/'evaluate_reference.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            variants={'all_mean':[0,1],'all_max':[0,1],'greedy_mean':[1],'greedy_max':[0]}
            protocol=dict(checkpoint_sha256='weights',validation_sha256='validation',raw_sha256='data',
                          primary='all_mean',variants=variants)
            validation=dict(checkpoint_sha256='weights',test_evaluated=False,
                            variants={k:dict(views=v) for k,v in variants.items()})
            self.assertEqual(module.frozen_variants(protocol,validation,'weights','validation','data',2),variants)
            for key in ('checkpoint_sha256','validation_sha256','raw_sha256'):
                bad=copy.deepcopy(protocol);bad[key]='changed'
                with self.assertRaises(ValueError):module.frozen_variants(bad,validation,'weights','validation','data',2)
            bad=copy.deepcopy(protocol);bad['variants']['greedy_mean']=[0]
            with self.assertRaises(ValueError):module.frozen_variants(bad,validation,'weights','validation','data',2)
        finally:sys.path.remove(path)

    def test_average_and_max_match_manual_shared_graph_vote(self):
        script=r'''
import sys,torch
sys.path.insert(0,'baselines/sugar/upstream');sys.path.insert(0,'baselines/sugar/adapters')
from models import ciga
from compat import install_masks
from ensemble import probabilities
from torch_geometric.data import Data
torch.set_num_threads(1);torch.manual_seed(21);install_masks(ciga)
models=[ciga.CIGA(ratio=.5,input_dim=39,edge_dim=10,out_dim=2,gnn_type='gin',num_layers=2,emb_dim=8,drop_ratio=0,c_dim=8,c_in='feat',virtual_node=False).eval() for _ in range(2)]
batch=Data(x=torch.randn(6,39),edge_index=torch.tensor([[0,1,2,3,4,5],[1,0,3,2,5,4]]),edge_attr=torch.randn(6,10),batch=torch.zeros(6,dtype=torch.long),y=torch.tensor([0]))
row,col=batch.edge_index
with torch.no_grad():
 scores=[]
 for m in models:
  h=m.gnn_encoder(batch);scores.append(m.edge_att(torch.cat([h[row],h[col]],-1)).view(-1))
 for aggregation in ('mean','max'):
  stacked=torch.stack(scores);shared=stacked.mean(0) if aggregation=='mean' else stacked.max(0).values
  frozen=shared.clone();votes=[];retained=[]
  for m in models:
   causal,_=m.split_graph(batch,shared.clone(),m.ratio);retained.append(causal[0])
   votes.append(m(batch,subgraph_avg=True,pred_edge_weight=shared.clone()).softmax(-1)[:,1])
  assert torch.equal(retained[0],retained[1]);assert torch.equal(shared,frozen)
  expected=torch.stack(votes).mean(0)
  actual=probabilities(models,batch,'shared_raw_uniform',aggregation)
  torch.testing.assert_close(actual,expected,rtol=0,atol=0)
  batch.y=1-batch.y
  torch.testing.assert_close(actual,probabilities(models,batch,'shared_raw_uniform',aggregation),rtol=0,atol=0)
  torch.testing.assert_close(actual,probabilities(models[::-1],batch,'shared_raw_uniform',aggregation))
'''
        result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr[-3000:])

    def test_greedy_accepts_ties_and_rejects_worse_additions(self):
        root=Path(__file__).resolve().parents[1]
        # The evaluation adapter imports its sibling modules by design.
        path=str(root/'baselines/sugar/adapters');sys.path.insert(0,path)
        try:
            spec=importlib.util.spec_from_file_location('paper_evaluator',Path(path)/'evaluate_reference.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            values={(1,):.8,(1,0):.8,(1,0,2):.7}
            selected,trace=module.greedy_select([.7,.9,.6],lambda ids:values[tuple(ids)])
            self.assertEqual(selected,[1,0])
            self.assertEqual([r['accepted'] for r in trace],[True,True,False])
        finally:sys.path.remove(path)
