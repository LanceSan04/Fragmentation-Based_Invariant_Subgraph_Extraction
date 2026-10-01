"""Render actual own-view forward traces; evaluation or sampler-only demonstration."""

import argparse, hashlib, html, json
from datetime import datetime
from pathlib import Path
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from src.training.pipeline import (
    Config,
    build_views_from_feature_dims,
    strip_non_tensor_batch_fields,
)
from src.training.checkpoint import load_checkpoint, load_view_state_dicts
from src.data.dataloader import build_split_dataloaders
from src.models.fragment_graph import get_fragment_view_value


def fragment_drawing(batch, method, eligible, selected, path):
    """Draw the actual fragment topology with atom-count labels and stable layout."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    edge_index = get_fragment_view_value(batch, "fragment_edge_index", method).cpu()
    mapping = get_fragment_view_value(batch, "atom_to_fragment", method).cpu()
    n = get_fragment_view_value(batch, "fragment_x_static", method).size(0)
    sizes = torch.bincount(mapping, minlength=n).tolist()
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(n))
    colors = []
    for j, (u, v) in enumerate(edge_index.T.tolist()):
        graph.add_edge(u, v, key=j)
    for u, v, j in graph.edges(keys=True):
        colors.append(
            "#15803d" if bool(selected[j]) else "#d97706" if bool(eligible[j]) else "#cbd5e1"
        )
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(graph, seed=17)
    nx.draw_networkx(
        graph,
        pos,
        ax=ax,
        labels={i: f"{i}\n{sizes[i]} atoms" for i in range(n)},
        node_size=600 if n < 40 else 180,
        font_size=8 if n < 40 else 4,
        node_color="#dbeafe",
        edge_color=colors,
        width=2,
    )
    ax.set_title("Fragment ID / atom count\nGreen: selected; orange: eligible; gray: sampled out")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def drawing(mol, atoms, bonds, png=False):
    d = rdMolDraw2D.MolDraw2DCairo(450, 330) if png else rdMolDraw2D.MolDraw2DSVG(450, 330)
    d.drawOptions().addAtomIndices = True
    d.DrawMolecule(mol, highlightAtoms=atoms, highlightBonds=bonds)
    d.FinishDrawing()
    return d.GetDrawingText()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--processed-dir")
    p.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--mode", choices=["evaluation", "sampling-demo"], default="evaluation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--output-dir")
    a = p.parse_args(argv)
    if min(a.indices) < 0 or len(set(a.indices)) != len(a.indices):
        p.error("Unique nonnegative indices required")
    payload = load_checkpoint(a.checkpoint)
    values = dict(payload["config"])
    split = values["validation_split"]
    if "test" in split.lower():
        raise ValueError("Validation split required")
    values.update(batch_size=1, num_workers=0, loader_split_names=[split], device=a.device)
    if a.processed_dir:
        values["processed_dir"] = a.processed_dir
    cfg = Config(values)
    views = build_views_from_feature_dims(cfg, payload["feature_dims"])
    load_view_state_dicts(views, payload)
    if any(v.erm_baseline for v in views):
        raise ValueError("Fragment checkpoint required")
    torch.manual_seed(a.seed)
    for i, v in enumerate(views):
        v.to(a.device).eval()
        v.fragment_edge_sampler.train(a.mode == "sampling-demo")
        v.fragment_edge_sampler.seed = a.seed + i
        v.fragment_edge_sampler._call_count.zero_()
    loaders = build_split_dataloaders(cfg.processed_dir, cfg)
    dataset = loaders[split].dataset
    if max(a.indices) >= len(dataset):
        raise ValueError("Molecule index out of range")
    smiles = {i: dataset[i].smiles for i in a.indices}
    strip_non_tensor_batch_fields(loaders)
    out = Path(a.output_dir or f"outputs/assay/visualize_views/{datetime.now():%Y%m%d_%H%M%S_%f}")
    out.mkdir(parents=True, exist_ok=False)
    pages = []
    with torch.no_grad():
        for index, batch in enumerate(loaders[split]):
            if index not in a.indices:
                continue
            batch = batch.to(a.device)
            mol = Chem.MolFromSmiles(smiles[index])
            AllChem.Compute2DCoords(mol)
            if mol.GetNumAtoms() != batch.x.size(0):
                raise ValueError("Atom mapping mismatch")
            folder = out / f"molecule_{index}"
            folder.mkdir()
            (folder / "input.svg").write_text(drawing(mol, [], []))
            traces = []
            rows = []
            for i, v in enumerate(views):
                pred, t = v(batch, return_intermediates=True)
                eligible = t["fragment_edge_eligible"]
                selected = t["selected_fragment_mask"]
                assert not (selected & ~eligible).any()
                boundary = get_fragment_view_value(
                    batch, "fragment_edge_to_bond_index", v.fragmentation_method
                )

                def rd_bonds(ids):
                    ans = []
                    for j in ids:
                        u, w = batch.bond_index[:, j].tolist()
                        b = mol.GetBondBetweenAtoms(u, w)
                        if b is None:
                            raise ValueError("Bond mapping mismatch")
                        ans.append(b.GetIdx())
                    return ans

                sample_bonds = rd_bonds(boundary[eligible].tolist())
                selected_bonds = rd_bonds(t["selected_bond_ids"].tolist())
                atoms = t["selected_atom_ids"].tolist()
                (folder / f"view{i+1}_sample.svg").write_text(drawing(mol, [], sample_bonds))
                (folder / f"view{i+1}_selected.svg").write_text(drawing(mol, atoms, selected_bonds))
                (folder / f"view{i+1}_selected.png").write_bytes(
                    drawing(mol, atoms, selected_bonds, True)
                )
                trace = {
                    k: t[k].detach().cpu().tolist()
                    for k in [
                        "fragment_scores",
                        "fragment_edge_eligible",
                        "selected_fragment_mask",
                        "selected_atom_ids",
                        "selected_bond_ids",
                    ]
                }
                trace.update(
                    view=i + 1,
                    method=v.fragmentation_method,
                    probabilities=pred.exp().cpu().tolist()[0],
                )
                traces.append(trace)
                trace.update(
                    original_atoms=mol.GetNumAtoms(),
                    original_bonds=mol.GetNumBonds(),
                    retained_atom_fraction=len(atoms) / mol.GetNumAtoms(),
                    retained_bond_fraction=(
                        len(selected_bonds) / mol.GetNumBonds() if mol.GetNumBonds() else None
                    ),
                    original_fragment_connections=int(selected.numel()),
                    eligible_connections=int(eligible.sum()),
                    selected_connections=int(selected.sum()),
                )
                fragment_drawing(
                    batch,
                    v.fragmentation_method,
                    eligible,
                    selected,
                    folder / f"view{i+1}_topology.png",
                )
                rows.append(
                    f'<tr><td>View {i+1}<br>P(positive)={trace["probabilities"][1]:.4f}<br>Atoms retained: {len(atoms)}/{mol.GetNumAtoms()}<br>Bonds retained: {len(selected_bonds)}/{mol.GetNumBonds()}<br>Connections: {int(selected.sum())}/{selected.numel()}</td><td><img src="view{i+1}_sample.svg"></td><td><img src="view{i+1}_selected.svg"></td><td><img width="600" src="view{i+1}_topology.png"></td></tr>'
                )
            (folder / "trace.json").write_text(json.dumps(traces, indent=2))
            mean = sum(t["probabilities"][1] for t in traces) / len(traces)
            page = f'<html><body><h1>Molecule {index}: {a.mode}</h1><p>{html.escape(smiles[index])}</p><img src="input.svg"><p>Highlights show eligible boundary bonds and selected atoms/bonds on the original molecule. Unhighlighted structure is context, not selected input. Scores and exact masks are in trace.json. Own-view mean P(positive): {mean:.4f}. Sampling-demo freezes dropout/BatchNorm; it is not normal inference.</p><table><tr><th>View</th><th>Eligible boundary bonds</th><th>Selected atom graph highlighted</th><th>Fragment topology (layout is not molecular geometry)</th></tr>{"".join(rows)}</table></body></html>'
            (folder / "index.html").write_text(page)
            pages.append(f'<li><a href="molecule_{index}/index.html">Molecule {index}</a></li>')
            if index >= max(a.indices):
                break
    (out / "index.html").write_text(
        "<html><body><h1>Own-view forward comparison</h1><ul>"
        + "".join(pages)
        + "</ul></body></html>"
    )
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                checkpoint=str(a.checkpoint),
                sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                split=split,
                indices=a.indices,
                mode=a.mode,
                seed=a.seed,
                checkpoint_config=payload["config"],
                effective_config=cfg.to_dict(),
                sampler_seeds=[a.seed + i for i in range(len(views))],
                device=a.device,
            ),
            indent=2,
        )
    )
    print(out / "index.html")
