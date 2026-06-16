"""Reconstruct atom-level invariant graphs from selected fragment edges."""


def selected_fragment_nodes(fragment_edge_index, selected_edge_mask):
    """Collect endpoint fragments for every selected fragment edge."""
    raise NotImplementedError


def selected_atom_nodes(fragment_to_atoms, selected_fragments):
    """Collect every original atom contained by selected fragments."""
    raise NotImplementedError


def reconstruct_atom_graph(
    molecule_data,
    selected_fragment_edge_mask,
):
    """Restore selected fragments, internal bonds, and selected attachments."""
    raise NotImplementedError


def relabel_reconstructed_atoms(atom_ids, edge_index):
    """Relabel selected atoms while retaining original atom provenance."""
    raise NotImplementedError


def reconstruct_batch(batch, selected_fragment_edge_mask):
    """Reconstruct each molecule independently and return a PyG batch."""
    raise NotImplementedError
