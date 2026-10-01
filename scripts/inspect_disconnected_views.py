"""Render saved validation selections and decompose their disconnectedness."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from src.data.dataloader import load_processed_split


def graph(atoms, bonds):
    g=nx.Graph();g.add_nodes_from(atoms);g.add_edges_from(bonds);return g


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();root=a.results
    expected=json.loads((root/'frozen_plan.json').read_text())['data_sha256']['ood_val']
    if hashlib.sha256(a.cache.read_bytes()).hexdigest()!=expected:raise ValueError('Cache differs from frozen study')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    data=load_processed_split(a.cache)['molecule_data']
    reports=[]
    # Examples chosen from prior geometry only, not labels or predictions.
    examples=[2438,732,705,1288,1337]
    for source in sorted((root/'retention').glob('pooled_*/eval_edge0.5.json')):
        rows=json.loads(source.read_text())['rows']
        for r in rows:
            d=data[r['index']]; atoms=set(r['atom_ids']);all_bonds=d.bond_index.T.tolist()
            selected=[all_bonds[i] for i in r['bond_ids']]
            induced=[b for b in all_bonds if b[0] in atoms and b[1] in atoms]
            original_c=nx.number_connected_components(graph(range(d.x.size(0)),all_bonds))
            selected_c=nx.number_connected_components(graph(atoms,selected))
            induced_c=nx.number_connected_components(graph(atoms,induced))
            if selected_c!=r['components']:raise ValueError('Saved component count differs')
            reports.append(dict(seed=source.parent.name,index=r['index'],view=r['view'],
                                original_atoms=d.x.size(0),original_components=original_c,
                                selected_components=selected_c,induced_components=induced_c,
                                missing_bonds=len(induced)-len(selected),
                                component_excess_from_missing_bonds=selected_c-induced_c,
                                component_excess_remaining=induced_c-original_c))
            if source.parent.name!='pooled_8301' or r['index'] not in examples:continue
            mol=Chem.MolFromSmiles(d.smiles);AllChem.Compute2DCoords(mol)
            if mol.GetNumAtoms()!=d.x.size(0):raise ValueError('Atom indexing differs')
            conf=mol.GetConformer();pos={i:(conf.GetAtomPosition(i).x,conf.GetAtomPosition(i).y) for i in range(mol.GetNumAtoms())}
            fig,axes=plt.subplots(1,3,figsize=(18,6),layout='constrained')
            variants=[('Original molecule',set(range(mol.GetNumAtoms())),all_bonds),
                      ('Actual selected graph',atoms,selected),('Same atoms, all original bonds',atoms,induced)]
            xs=[v[0] for v in pos.values()];ys=[v[1] for v in pos.values()]
            for ax,(name,ns,edges) in zip(axes,variants):
                g=graph(ns,edges);components=list(nx.connected_components(g));colors={n:j%20 for j,comp in enumerate(components) for n in comp}
                ordered=sorted(ns)
                nx.draw_networkx_edges(g,pos,ax=ax,width=.9,edge_color='#334155')
                nx.draw_networkx_nodes(g,pos,ax=ax,nodelist=ordered,node_color=[colors[n] for n in ordered],cmap=plt.cm.tab20,vmin=0,vmax=19,node_size=18 if mol.GetNumAtoms()>100 else 60)
                if mol.GetNumAtoms()<=60:
                    nx.draw_networkx_labels(g,pos,labels={i:mol.GetAtomWithIdx(i).GetSymbol()+str(i) for i in ns},ax=ax,font_size=5)
                ax.set_title(f'{name}\n{len(ns)} atoms, {len(edges)} bonds, {len(components)} components',fontsize=11)
                ax.set_xlim(min(xs)-2,max(xs)+2);ax.set_ylim(min(ys)-2,max(ys)+2);ax.set_aspect('equal');ax.axis('off')
            fig.suptitle(f'Validation molecule {r["index"]}, seed 8301, view {r["view"]+1}\nSaved forward trace; shared 2-D atom coordinates. Colors distinguish components, not chemical elements.',fontsize=12)
            fig.savefig(a.output_dir/f'molecule_{r["index"]}_view{r["view"]+1}.png',dpi=160);plt.close(fig)
    (a.output_dir/'connectivity_decomposition.json').write_text(json.dumps(reports,indent=2))
    (a.output_dir/'manifest.json').write_text(json.dumps(dict(source=str(root.resolve()),cache_sha256=expected,
        example_indices=examples,seed=8301,note='Examples chosen using geometry, not labels. Drawings are graph schematics, not bond-order/stereochemistry depictions.'),indent=2))
    body=['<html><head><meta charset="utf-8"><title>Connectivity inspection</title></head><body style="font-family:Arial;max-width:1500px;margin:auto">',
          '<h1>What fragmentation keeps and disconnects</h1><p>Actual saved selections from validation, seed 8301. Left: original graph. Middle: actual model input. Right: the same atoms with every original bond between them restored. No atoms or nonexistent bonds are added. Colors distinguish connected components. These are graph schematics; bond order and stereochemistry are not depicted.</p>']
    for index in examples:
        body.append(f'<h2>Molecule {index}: {data[index].x.size(0)} atoms</h2><p>{html.escape(data[index].smiles)}</p>')
        for view in range(1,6):body.append(f'<img style="width:100%" src="molecule_{index}_view{view}.png" alt="Molecule {index}, view {view}">')
    body.append('</body></html>');(a.output_dir/'index.html').write_text(''.join(body),encoding='utf-8')
    print(a.output_dir/'index.html')


if __name__=='__main__':main()
