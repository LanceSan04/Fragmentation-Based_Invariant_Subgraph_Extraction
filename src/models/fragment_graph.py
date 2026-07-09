"""Convert molecule fragment-level provenance into fragment graph PyG format."""

import torch
from torch_geometric.data import Data

from src.data.pyg_batch import fragment_view_field


def get_fragment_view_value(molecule_data, field, method=None):
    """Read a default or method-specific fragment field from a molecule batch."""
    name = fragment_view_field(method, field)
    if hasattr(molecule_data, name):
        return getattr(molecule_data, name)
    if method is None:
        raise AttributeError(f"molecule data is missing fragment field {field!r}")
    raise AttributeError(f"molecule data is missing {method!r} fragment field {field!r}")


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


def build_fragment_graph_batch(molecule_batch, method=None):
    """Build a batched fragment-level graph directly from DataLoader batch tensors.

    Args:
        molecule_batch: PyG Batch of MoleculeData objects

    Returns:
        Data: Batched PyG Data for fragment-level graph with batch assignment
    """
    batched = Data(
        x=get_fragment_view_value(molecule_batch, "fragment_x_static", method),
        edge_index=get_fragment_view_value(molecule_batch, "fragment_edge_index", method),
        edge_attr=get_fragment_view_value(molecule_batch, "fragment_edge_attr", method),
        batch=get_fragment_view_value(molecule_batch, "fragment_x_static_batch", method),
        y=molecule_batch.y,
        num_nodes=get_fragment_view_value(molecule_batch, "fragment_x_static", method).size(0),
    )
    batched.full_fragment_edge_id = torch.arange(
        batched.edge_index.size(1),
        dtype=torch.long,
        device=batched.edge_index.device,
    )
    return batched
