"""Reconstruct atom-level invariant subgraphs from fragment-level edge selections."""

import torch


def reconstruct_from_fragment_edges(molecule_data, fragment_edge_mask):
    """Reconstruct atom-level graph from selected fragment edges.
    
    This takes the output from the fragment-level GNN (which edge masks to keep)
    and expands it back to the atom level using the fragment-to-atom mappings.
    
    Args:
        molecule_data: MoleculeData with atom/fragment provenance
        fragment_edge_mask: Binary mask indicating which fragment edges to keep
        
    Returns:
        Data: PyG Data with only atoms from selected fragments and their internal bonds
    """
    from torch_geometric.data import Data
    from src.models.atom_reconstructor import reconstruct_atom_graph
    
    # Use existing atom reconstructor which expects a mask over fragment edges
    reconstructed = reconstruct_atom_graph(molecule_data, fragment_edge_mask)
    
    return reconstructed


def batch_reconstruct_from_fragment_edges(batch_data, batch_fragment_edge_masks):
    """Reconstruct atoms for each molecule in a batch.
    
    Args:
        batch_data: Batched PyG Batch of MoleculeData
        batch_fragment_edge_masks: Concatenated mask for all fragment edges in batch
        
    Returns:
        list: Reconstructed Data objects, one per molecule
    """
    from src.models.atom_reconstructor import reconstruct_batch
    
    # Use batch reconstructor
    reconstructed_batch = reconstruct_batch(batch_data, batch_fragment_edge_masks)
    
    return reconstructed_batch.to_data_list()


def get_fragment_edges_for_molecule(batch_data, molecule_idx):
    """Get the fragment edges and their masks for a specific molecule in batch.
    
    Args:
        batch_data: Batched MoleculeData
        molecule_idx: Index of molecule in batch
        
    Returns:
        tuple: (fragment_edge_indices, fragment_edge_mask)
    """
    molecule_data = batch_data.to_data_list()[molecule_idx]
    num_frag_edges = molecule_data.fragment_edge_index.size(1)
    
    return molecule_data.fragment_edge_index, torch.ones(num_frag_edges, dtype=torch.bool)
