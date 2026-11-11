from pathlib import Path
import subprocess
import sys
import unittest


class SugarCorrectionTests(unittest.TestCase):
    def test_selected_atom_embeddings_and_gradients(self):
        script=r'''
import sys,torch
sys.path.insert(0,'baselines/sugar/upstream')
sys.path.insert(0,'baselines/sugar/adapters')
from models import ciga
from compat import install_masks,install_contrastive_indexing,verify_sources
from torch_geometric.data import Data
torch.set_num_threads(1);torch.manual_seed(3);verify_sources();install_masks(ciga)
class Encoder(torch.nn.Module):
 def forward(self,batch):return batch.x
class Scores(torch.nn.Module):
 def forward(self,rep):return rep.sum(-1,keepdim=True)*0+rep.new_tensor([[-4.],[-3.],[9.],[10.],[-2.],[-1.]])
model=ciga.CIGA(ratio=1/3,input_dim=8,edge_dim=10,out_dim=2,gnn_type='gin',num_layers=2,emb_dim=8,drop_ratio=0,c_dim=8,c_in='feat',c_rep='feat',c_pool='add',virtual_node=False).eval()
model.gnn_encoder=Encoder();model.edge_att=Scores()
x=torch.arange(48.).reshape(6,8).requires_grad_()
batch=Data(x=x,edge_index=torch.tensor([[0,1,2,3,4,5],[1,0,3,2,5,4]]),edge_attr=torch.ones(6,10),batch=torch.zeros(6,dtype=torch.long))
pred_before,old=model(batch,return_data='feat');old.sum().backward()
assert torch.equal(x.grad.abs().sum(1)>0,torch.tensor([True,True,False,False,False,False]))
x.grad=None;install_contrastive_indexing(ciga)
pred_after,new=model(batch,return_data='feat')
torch.testing.assert_close(pred_before,pred_after,rtol=0,atol=0)
torch.testing.assert_close(new,(x[2]+x[3]).unsqueeze(0))
new.sum().backward()
assert torch.equal(x.grad.abs().sum(1)>0,torch.tensor([False,False,True,True,False,False]))
verify_sources()
'''
        self.run_script(script)

    def test_vote_is_label_independent_and_view_order_invariant(self):
        script=r'''
import sys,torch
sys.path.insert(0,'baselines/sugar/adapters')
from ensemble import probabilities
from torch_geometric.data import Data
class Model:
 def __init__(self,scale):self.scale=scale
 def eval(self):return self
 def gnn_encoder(self,batch):return batch.x*self.scale
 def edge_att(self,rep):return rep.sum(-1)
 def __call__(self,batch,subgraph_avg=False,pred_edge_weight=None):
  if subgraph_avg:
   s=pred_edge_weight.clone().sum();pred_edge_weight.mul_(-10)
  else:s=batch.x.sum()*self.scale
  return torch.stack([s*0,s]).reshape(1,2)
b=Data(x=torch.tensor([[.1],[.2]]),edge_index=torch.tensor([[0,1],[1,0]]),y=torch.tensor([0]))
m=[Model(1.),Model(2.)]
for policy in ('shared_raw_uniform','independent_uniform'):
 p=probabilities(m,b,policy);b.y=torch.tensor([1]);q=probabilities(m,b,policy)
 torch.testing.assert_close(p,q,rtol=0,atol=0)
 torch.testing.assert_close(p,probabilities(list(reversed(m)),b,policy))
'''
        self.run_script(script)

    def run_script(self,script):
        result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr[-4000:])
