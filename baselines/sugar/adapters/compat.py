"""Explicit compatibility layer; the pinned upstream files are never edited."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]


def verify_sources():
    for directory,manifest in [(BASE/'upstream',BASE/'provenance.json'),
                               (BASE/'drugood_reference',BASE/'drugood_reference/provenance.json')]:
        value=json.loads(manifest.read_text())
        for name,expected in value['files'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Pinned reference changed: '+name)


def functions_from(path,names,namespace):
    tree=ast.parse(Path(path).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Reference functions missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def feature_module():
    path=BASE/'drugood_reference/drugood/utils/smile_to_dgl.py'
    spec=importlib.util.spec_from_file_location('pinned_drugood_features',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def modern_masks(mask,network):
    import torch
    from torch_geometric.nn import MessagePassing
    for layer in network.modules():
        if isinstance(layer,MessagePassing):
            layer.explain=True
            layer._edge_mask=mask
            layer._loop_mask=torch.ones(mask.numel(),dtype=torch.bool,device=mask.device)
            layer._apply_sigmoid=True


def modern_clear(network):
    from torch_geometric.nn import MessagePassing
    for layer in network.modules():
        if isinstance(layer,MessagePassing):
            layer.explain=False;layer._edge_mask=None;layer._loop_mask=None


def install_masks(ciga):
    ciga.set_masks=modern_masks;ciga.clear_masks=modern_clear


def install_contrastive_indexing(ciga):
    """Opt-in, one-expression correction; preserve the pinned source on disk."""
    tree=ast.parse((BASE/'upstream/models/ciga.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CIGA')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    class Correct(ast.NodeTransformer):
        count=0
        def visit_Assign(self,node):
            if (len(node.targets)==1 and isinstance(node.targets[0],ast.Tuple)
                and isinstance(node.targets[0].elts[0],ast.Name)
                and node.targets[0].elts[0].id=='causal_h'
                and ast.unparse(node.value)=='relabel(h, causal_edge_index, batch.batch)'):
                node.value.args[1]=ast.Name(id='unmapped_casual_edge',ctx=ast.Load())
                self.count+=1
            return self.generic_visit(node)
    correction=Correct();method=correction.visit(method)
    if correction.count!=1:raise ValueError('Unexpected upstream contrastive indexing site')
    module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
    namespace={}
    exec(compile(module,'sugar_corrected_contrastive_indexing','exec'),ciga.__dict__,namespace)
    ciga.CIGA.forward=namespace['forward']
    return dict(name='original_atom_ids_for_feature_contrast',expression_changes=1)


def seed_function():
    import torch,numpy as np,random
    scope=dict(torch=torch,np=np,random=random)
    return functions_from(BASE/'upstream/utils/util.py',['set_seed'],scope)['set_seed']
