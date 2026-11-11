from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch


class UpstreamReferenceTests(unittest.TestCase):
    def test_deferred_test_reporting_preserves_training_on_fixture(self):
        # Run in isolated interpreters: upstream uses top-level models/utils names.
        root=Path(__file__).resolve().parents[1]
        import importlib.util
        if any(importlib.util.find_spec(name) is None for name in ('ogb','dgl')):
            self.skipTest('Optional upstream reference dependencies unavailable')
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for mode in ('deferred','reference'):
                out=Path(d)/mode;paths.append(out)
                command=[sys.executable,str(root/'baselines/sugar/adapters/run.py'),'--smoke',
                         '--epochs','2','--views','2','--device','cpu','--output-dir',str(out)]
                if mode=='reference':command+=['--reference-test-evaluation']
                result=subprocess.run(command,cwd=root,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr[-3000:])
            a,b=[torch.load(p/'best_views.pt',map_location='cpu',weights_only=False) for p in paths]
            self.assertEqual(a['selected_epochs'],b['selected_epochs'])
            self.assertEqual(a['selected_validation_auc'],b['selected_validation_auc'])
            for first,second in zip(a['view_states'],b['view_states']):
                for key in first:self.assertTrue(torch.equal(first[key],second[key]),key)

    def test_modern_gate_matches_explicit_gine_message_sum(self):
        root=Path(__file__).resolve().parents[1]
        import importlib.util
        if importlib.util.find_spec('ogb') is None:self.skipTest('Optional OGB unavailable')
        script=r'''
import sys,torch
sys.path.insert(0,'baselines/sugar/upstream')
sys.path.insert(0,'baselines/sugar/adapters')
from models.conv import GINConv
from compat import modern_masks,modern_clear
torch.set_num_threads(1);torch.manual_seed(5)
layer=GINConv(8,edge_dim=10).eval()
x=torch.randn(4,8);edge=torch.tensor([[0,1,1,2],[1,0,2,1]])
attributes=torch.randn(4,10);mask=torch.tensor([-2.,-.5,.5,2.],requires_grad=True)
modern_masks(mask,layer);observed=layer(x,edge,attributes);modern_clear(layer)
messages=torch.relu(x[edge[0]]+layer.edge_encoder(attributes))*mask.sigmoid()[:,None]
summed=x.new_zeros(x.shape).index_add(0,edge[1],messages)
expected=layer.mlp((1+layer.eps)*x+summed)
torch.testing.assert_close(observed,expected,rtol=1e-6,atol=1e-6)
observed.sum().backward();assert mask.grad.abs().sum()>0
'''
        result=subprocess.run([sys.executable,'-c',script],cwd=root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr[-3000:])
