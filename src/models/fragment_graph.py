"""Convert molecule fragment-level provenance into fragment graph PyG format."""

import torch
from torch_geometric.data import Data


def build_fragment_graph(molecule_data, fragmentation_result):
    """Build a fragment-level graph from molecule data and fragmentation.
    
    Args:
        molecule_data: MoleculeData with atom-level features and bonds
        fragmentation_result: FragmentationResult with fragment provenance
        
    Returns:
        Data: PyG Data object representing the fragment-level graph
              - x: fragment features (from fragment_x_static)
              - edge_index: fragment-to-fragment edges
              - edge_attr: fragment edge features
    """
    # Fragment features are already pooled in molecule_data.fragment_x_static
    fragment_x = molecule_data.fragment_x_static
    
    # Fragment edges connect fragments that have inter-fragment bonds
    fragment_edge_index = molecule_data.fragment_edge_index
    fragment_edge_attr = molecule_data.fragment_edge_attr
    
    # Create PyG Data object for fragment graph
    fragment_graph = Data(
        x=fragment_x,
        edge_index=fragment_edge_index,
        edge_attr=fragment_edge_attr,
        num_nodes=fragment_x.size(0),
    )
    
    return fragment_graph


def build_fragment_graph_batch(molecule_batch):
    """Build a batched fragment-level graph from a batch of molecules.
    
    Args:
        molecule_batch: PyG Batch of MoleculeData objects
        
    Returns:
        Data: Batched PyG Data for fragment-level graph with batch assignment
    """
    from torch_geometric.data import Batch
    
    fragment_graphs = []
    for molecule_data in molecule_batch.to_data_list():
        # Build fragment graph for each molecule
        frag_graph = Data(
            x=molecule_data.fragment_x_static,
            edge_index=molecule_data.fragment_edge_index,
            edge_attr=molecule_data.fragment_edge_attr,
            y=molecule_data.y,
        )
        fragment_graphs.append(frag_graph)
    
    # Batch all fragment graphs together. The edge order produced here matches
    # the order used by reconstruct_batch(batch, full_edge_mask), so these ids
    # are the bridge from sampled fragment graphs back to the full molecule
    # provenance tensors.
    batched_frag_graph = Batch.from_data_list(fragment_graphs)
    batched_frag_graph.full_fragment_edge_id = torch.arange(
        batched_frag_graph.edge_index.size(1),
        dtype=torch.long,
        device=batched_frag_graph.edge_index.device,
    )
    
    return batched_frag_graph
