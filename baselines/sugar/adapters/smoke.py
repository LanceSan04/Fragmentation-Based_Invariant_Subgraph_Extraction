"""Isolated compatibility probe, not a benchmark/reproduction training runner."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import MessagePassing
    from rdkit import Chem
    torch.set_num_threads(1); torch.manual_seed(17)
    sys.path.insert(0, str(BASE / "upstream"))
    from models import ciga
    from models.losses import get_contrast_loss
    spec = importlib.util.spec_from_file_location("reference_featurizer", BASE / "drugood_reference/drugood/utils/smile_to_dgl.py")
    featurizer = importlib.util.module_from_spec(spec); spec.loader.exec_module(featurizer)
    data = []
    for i, smiles in enumerate(("CCOC(=O)C", "CCN(CC)CC", "c1ccccc1CC(=O)NC", "CCCOC")):
        molecule = Chem.MolFromSmiles(smiles)
        edges, attributes = [], []
        for bond in molecule.GetBonds():
            a,b = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            edges += [(a,b),(b,a)]
            attributes += [featurizer.get_bond_features(bond)] * 2
        import numpy as np
        data.append(Data(x=torch.tensor(np.asarray([featurizer.get_atom_features(a) for a in molecule.GetAtoms()]),dtype=torch.float32),
                         edge_index=torch.tensor(edges).T.contiguous(),
                         edge_attr=torch.tensor(np.asarray(attributes),dtype=torch.float32),
                         y=torch.tensor([i%2]), idx=torch.tensor([i]), node_label=torch.zeros(molecule.GetNumAtoms())))
    batch = Batch.from_data_list(data)
    model = ciga.CIGA(input_dim=39, edge_dim=10, out_dim=2, emb_dim=16, num_layers=2,
                      virtual_node=False, drop_ratio=0, ratio=-1, c_in="raw").eval()
    logits = torch.zeros(batch.edge_index.size(1))
    with torch.no_grad():
        legacy_low = model(batch.clone(), subgraph_avg=True, pred_edge_weight=logits-8)
        legacy_high = model(batch.clone(), subgraph_avg=True, pred_edge_weight=logits+8)

    def modern_masks(mask, network):
        # PyG >=2: old double-underscore fields no longer enable explain_message.
        # Upstream mask is a logit: retain its original implicit sigmoid.
        for layer in network.modules():
            if isinstance(layer, MessagePassing):
                layer.explain = True
                layer._edge_mask = mask
                layer._loop_mask = torch.ones(mask.numel(), dtype=torch.bool, device=mask.device)
                layer._apply_sigmoid = True

    def modern_clear(network):
        for layer in network.modules():
            if isinstance(layer, MessagePassing):
                layer.explain = False
                layer._edge_mask = None
                layer._loop_mask = None

    ciga.set_masks, ciga.clear_masks = modern_masks, modern_clear
    with torch.no_grad():
        modern_low = model(batch.clone(), subgraph_avg=True, pred_edge_weight=logits-8)
        modern_high = model(batch.clone(), subgraph_avg=True, pred_edge_weight=logits+8)
    delta = float((modern_low-modern_high).abs().max())
    if not delta>0:
        raise RuntimeError("Compatibility mask bridge has no prediction effect")
    model.ratio=.5; model.train()
    prediction, representation = model(batch.clone(), return_data="feat")
    contrast = get_contrast_loss(representation, batch.y, norm=torch.nn.functional.normalize,
                                 contrast_t=1.0, sampling="mul")
    loss = torch.nn.functional.cross_entropy(prediction,batch.y) + .5*contrast
    loss.backward()
    scorer_grad = sum(float(p.grad.abs().sum()) for p in model.edge_att.parameters() if p.grad is not None)
    if not math_finite(loss.item()) or scorer_grad<=0:
        raise RuntimeError("Original scorer does not receive finite nonzero gradients")
    # Execute only the pinned perturbation function, avoiding main.py's external imports.
    tree = ast.parse((BASE/"upstream/main.py").read_text())
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="delete_edge")
    scope = dict(random=random, Data=Data)
    exec(compile(ast.Module(body=[function],type_ignores=[]),"upstream_delete_edge","exec"),scope)
    perturb = scope['delete_edge']; options=SimpleNamespace(sample_rate=95,dataset='drugood_probe')
    a=perturb(data[0].clone(),options,1); b=perturb(data[0].clone(),options,1)
    if not torch.equal(a.edge_index,b.edge_index):
        raise RuntimeError("Pinned perturbation changed across repeated calls")
    report=dict(synthetic=True, benchmark_results=False, atom_dim=39,bond_dim=10,
                legacy_gate_logit_intervention_max_delta=float((legacy_low-legacy_high).abs().max()),
                bridged_gate_logit_intervention_max_delta=delta,
                prediction_loss=float(torch.nn.functional.cross_entropy(prediction,batch.y)),
                contrastive_loss=float(contrast), scorer_gradient_l1=scorer_grad,
                fixed_perturbation_verified=True,
                note="Original CIGA + official DrugOOD feature functions; in-memory PyG mask compatibility bridge. Full reproduction still pending.")
    destination=Path(args.output); destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def math_finite(value):
    import math
    return math.isfinite(value)


if __name__ == "__main__":
    main()
