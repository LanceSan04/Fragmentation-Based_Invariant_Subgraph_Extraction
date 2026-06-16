#!/usr/bin/env python
"""Modified script for SuGAR pipeline integration focusing on the three specific required visualizations."""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# Pipeline specific imports
from src.data.dataloader import load_processed_split
from src.data.pyg_batch import batch_molecules
from src.models.fragment_graph import build_fragment_graph_batch
from src.models.sugar_ensemble import SuGAREnsemble

# =====================================================================
# RDKit & NETWORKX CONFIGURATIONS
# =====================================================================

FRAGMENT_COLORS_RGB = [
    [0.9, 0.3, 0.3],  # Soft Red
    [0.3, 0.4, 0.9],  # Soft Blue
    [0.3, 0.8, 0.4],  # Soft Green
    [0.9, 0.6, 0.2],  # Orange
    [0.7, 0.3, 0.8],  # Purple
    [0.2, 0.8, 0.8],  # Cyan
    [0.9, 0.9, 0.2],  # Yellow
    [0.9, 0.4, 0.7],  # Pink
]


def _get_clean_rdkit_mol(mol_data):
    """Safely extracts or generates a valid RDKit molecule with 2D coordinates."""
    mol = getattr(mol_data, "rdkit_mol", None)
    if mol is None and hasattr(mol_data, "smiles"):
        mol = Chem.MolFromSmiles(mol_data.smiles)
    
    if mol is not None:
        mol = Chem.Mol(mol)
        rdMolDraw2D.PrepareMolForDrawing(mol)
    return mol


# =====================================================================
# VISUALIZATION 1: INITIAL ATOM GRAPH WITH HIGHLIGHTED FRAGMENTS
# =====================================================================
def draw_initial_atom_graph_with_fragments(mol_data, save_path):
    """Visualizes the initial molecule structure with fragments fully highlighted and explicitly labeled."""
    mol = _get_clean_rdkit_mol(mol_data)
    if mol is None:
        return

    atom_to_fragment = mol_data.atom_to_fragment.cpu()
    highlight_atoms = []
    highlight_atom_colors = {}
    
    for atom_idx in range(mol.GetNumAtoms()):
        frag_id = atom_to_fragment[atom_idx].item()
        color = tuple(FRAGMENT_COLORS_RGB[frag_id % len(FRAGMENT_COLORS_RGB)])
        
        highlight_atoms.append(atom_idx)
        highlight_atom_colors[atom_idx] = color
        mol.GetAtomWithIdx(atom_idx).SetProp("atomNote", f"F{frag_id}")

    highlight_bonds = []
    highlight_bond_colors = {}
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        frag_u, frag_v = atom_to_fragment[u].item(), atom_to_fragment[v].item()
        if frag_u == frag_v:
            b_idx = bond.GetIdx()
            highlight_bonds.append(b_idx)
            highlight_bond_colors[b_idx] = tuple(FRAGMENT_COLORS_RGB[frag_u % len(FRAGMENT_COLORS_RGB)])

    d = rdMolDraw2D.MolDraw2DCairo(600, 600)
    d.drawOptions().atomHighlightsAreCircles = False
    d.drawOptions().fillHighlights = True
    
    d.DrawMolecule(
        mol, legend="1. Initial Atom Graph with Fragments Highlighted",
        highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors,
        highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors
    )
    d.FinishDrawing()
    with open(save_path, "wb") as f:
        f.write(d.GetDrawingText())


# =====================================================================
# VISUALIZATION 2: FRAGMENT GRAPH WITH HIGHLIGHTED GNN EDGES
# =====================================================================
def draw_fragment_graph_with_extracted_edges(mol_data, save_path, selected_mask):
    """Visualizes the fragment-level topology graph highlighting edges extracted by the GNN."""
    G = nx.Graph()
    num_fragments = mol_data.fragment_x_static.size(0)

    for f in range(num_fragments):
        G.add_node(f)

    fragment_edge_index = mol_data.fragment_edge_index.cpu()
    
    # Track which fragment edges are selected/active
    active_edges = set()
    all_edges = []
    
    for e in range(fragment_edge_index.shape[1]):
        u, v = fragment_edge_index[0, e].item(), fragment_edge_index[1, e].item()
        if u < v:  # Avoid duplicate undirected processing
            G.add_edge(u, v)
            all_edges.append((u, v))
            if bool(selected_mask[e]):
                active_edges.add((u, v))

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 6))

    node_colors = [tuple(FRAGMENT_COLORS_RGB[f % len(FRAGMENT_COLORS_RGB)]) for f in G.nodes()]
    labels = {f: f"F{f}" for f in G.nodes()}

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1400, node_color=node_colors, edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=11, font_weight="bold", font_color="white")
    
    # Draw non-extracted/pruned edges as thin dashed lines
    inactive_edges = [edge for edge in all_edges if edge not in active_edges]
    nx.draw_networkx_edges(G, pos, edgelist=inactive_edges, width=1.5, edge_color="#ced4da", style="dashed")
    
    # Draw extracted/active edges explicitly highlighted
    nx.draw_networkx_edges(G, pos, edgelist=list(active_edges), width=4.0, edge_color="#28a745")

    plt.title("2. Fragment Graph (GNN Extracted Edges Highlighted Green)", fontsize=11, fontweight="bold", pad=15)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


# =====================================================================
# VISUALIZATION 3: RECONSTRUCTED SUBGRAPH FROM EXTRACTED FRAGMENTS
# =====================================================================
def draw_reconstructed_subgraph_atoms(mol_data, save_path, selected_mask):
    """Renders original atom topology, ensuring that fragments whose 
    inter-fragment edge was pruned by the GNN are physically disconnected.
    """
    mol = _get_clean_rdkit_mol(mol_data)
    if mol is None:
        return

    fragment_edge_index = mol_data.fragment_edge_index.cpu()
    atom_to_fragment = mol_data.atom_to_fragment.cpu()

    # 1. Map active fragment-level edges to a set of tuples for O(1) lookups
    active_fragment_edges = set()
    kept_fragments = set()
    
    for edge_idx in range(fragment_edge_index.size(1)):
        if bool(selected_mask[edge_idx]):
            u_frag = fragment_edge_index[0, edge_idx].item()
            v_frag = fragment_edge_index[1, edge_idx].item()
            active_fragment_edges.add((min(u_frag, v_frag), max(u_frag, v_frag)))
            kept_fragments.add(u_frag)
            kept_fragments.add(v_frag)

    if not kept_fragments:
        kept_fragments.add(0)

    # Tag every atom with its original index before structural alteration
    for idx in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(idx).SetIntProp("original_idx", idx)

    # 2. Use EditableMol to drop unselected atoms AND prune inactive cross-fragment bonds
    emol = Chem.EditableMol(mol)
    
    # First: Remove unselected atoms (reverse order to safely preserve indices during loop)
    for atom_idx in reversed(range(mol.GetNumAtoms())):
        frag_id = atom_to_fragment[atom_idx].item()
        if frag_id not in kept_fragments:
            emol.RemoveAtom(atom_idx)

    # Generate intermediate mol to assess bonds based on original mappings
    temp_mol = emol.GetMol()
    
    # Second: Break bonds between fragments if their macro edge was pruned
    # Process in reverse order because removing bonds changes bond indices
    for bond_idx in reversed(range(temp_mol.GetNumBonds())):
        bond = temp_mol.GetBondWithIdx(bond_idx)
        u_idx = bond.GetBeginAtomIdx()
        v_idx = bond.GetEndAtomIdx()
        
        orig_u = temp_mol.GetAtomWithIdx(u_idx).GetIntProp("original_idx")
        orig_v = temp_mol.GetAtomWithIdx(v_idx).GetIntProp("original_idx")
        
        frag_u = atom_to_fragment[orig_u].item()
        frag_v = atom_to_fragment[orig_v].item()
        
        # If it's a cross-fragment bond, check if that specific macro edge is active
        if frag_u != frag_v:
            edge_key = (min(frag_u, frag_v), max(frag_u, frag_v))
            if edge_key not in active_fragment_edges:
                # BREAK THE BOND: The GNN did not extract this connection!
                emol.RemoveBond(u_idx, v_idx)

    # Finalize structural transformations
    filtered_mol = emol.GetMol()

    highlight_atoms = []
    highlight_atom_colors = {}
    highlight_bonds = []
    highlight_bond_colors = {}

    for atom in filtered_mol.GetAtoms():
        new_idx = atom.GetIdx()
        orig_idx = atom.GetIntProp("original_idx")
        frag_id = atom_to_fragment[orig_idx].item() 
        
        highlight_atoms.append(new_idx)
        highlight_atom_colors[new_idx] = tuple(FRAGMENT_COLORS_RGB[frag_id % len(FRAGMENT_COLORS_RGB)])
        atom.SetProp("atomNote", f"F{frag_id}")

    for bond in filtered_mol.GetBonds():
        u_new, v_new = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        u_orig = filtered_mol.GetAtomWithIdx(u_new).GetIntProp("original_idx")
        v_orig = filtered_mol.GetAtomWithIdx(v_new).GetIntProp("original_idx")
        
        frag_u = atom_to_fragment[u_orig].item()
        frag_v = atom_to_fragment[v_orig].item()
        
        if frag_u == frag_v:
            b_idx = bond.GetIdx()
            highlight_bonds.append(b_idx)
            highlight_bond_colors[b_idx] = tuple(FRAGMENT_COLORS_RGB[frag_u % len(FRAGMENT_COLORS_RGB)])

    d = rdMolDraw2D.MolDraw2DCairo(600, 600)
    d.drawOptions().atomHighlightsAreCircles = False
    d.drawOptions().fillHighlights = True
    
    d.DrawMolecule(
        filtered_mol, legend="3. Reconstructed Atomic Subgraph (Same Topology)",
        highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors,
        highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors
    )
    d.FinishDrawing()
    with open(save_path, "wb") as f:
        f.write(d.GetDrawingText())


# =====================================================================
# RUNTIME PIPELINE EXECUTION
# =====================================================================
def test_pipeline(data_path, num_samples=2):
    """Runs data parsing, model evaluations, and serializes the 3 required visual assets."""
    print("=" * 80)
    print("SuGAR PIPELINE VISUALIZATION ROUTINE")
    print("=" * 80)

    viz_dir = Path("debug_viz")
    viz_dir.mkdir(exist_ok=True)

    # 1. Load data
    print("\n1. Loading processed molecular data...")
    payload = load_processed_split(data_path)
    molecules = payload["molecule_data"][:num_samples]

    # 2. Pre-batch step 1 visuals
    print("2. Generating Initial Fragment Assignments (Visualization 1)...")
    for idx, mol_data in enumerate(molecules):
        draw_initial_atom_graph_with_fragments(mol_data, viz_dir / f"mol_{idx}_viz1_initial_atoms.png")

    batch_data = batch_molecules(molecules)
    fragment_batch = build_fragment_graph_batch(batch_data)

    # 3. Model Build & Execution
    print("3. Executing Forward Pass through SuGAR Ensemble...")
    model = SuGAREnsemble(
        atom_feature_dim=batch_data.x.size(1),
        fragment_feature_dim=fragment_batch.x.size(1),
        num_classes=2, num_gnn_layers=2, gnn_hidden_dim=64, edge_ratio=0.5, dropout=0.0,
    )

    device = torch.device("cpu")
    batch_data = batch_data.to(device)
    model = model.to(device)

    with torch.no_grad():
        _, intermediates = model(batch_data, return_intermediates=True)

    fragment_edge_mask = intermediates["fragment_edge_masks"][0]
    split_molecules = batch_data.to_data_list()
    
    # 4. Generate remaining required visualization steps
    print("4. Generating Edge Selection and Subgraph Reconstructions (Visualizations 2 & 3)...")
    offset = 0
    for idx, mol in enumerate(split_molecules):
        if idx >= num_samples:
            break
        num_edges = mol.fragment_edge_index.size(1)
        local_mask = fragment_edge_mask[offset : offset + num_edges].cpu()
        offset += num_edges

        # Visualization 2: Pure fragment network showing GNN selection matrix highlights
        draw_fragment_graph_with_extracted_edges(
            mol, viz_dir / f"mol_{idx}_viz2_fragment_graph.png", local_mask
        )

        # Visualization 3: Backmapped atom space removing everything outside GNN selection
        draw_reconstructed_subgraph_atoms(
            mol, viz_dir / f"mol_{idx}_viz3_reconstructed_atoms.png", local_mask
        )

    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED - Visual assets stored in: {viz_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SuGAR pipeline integration")
    parser.add_argument("--data_path", type=str, default="data/processed/train.pt")
    parser.add_argument("--num_samples", type=int, default=2)
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        print(f"Data file not found: {args.data_path}")
    else:
        test_pipeline(args.data_path, args.num_samples)