"""Reconstruct atom-level invariant graphs from selected fragment edges."""

import torch
from torch_geometric.data import Batch, Data


def selected_fragment_nodes(fragment_edge_index, selected_edge_mask):
    """Collect endpoint fragments for every selected fragment edge."""
    if fragment_edge_index.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=fragment_edge_index.device)
    selected_edges = fragment_edge_index[:, selected_edge_mask]
    if selected_edges.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=fragment_edge_index.device)
    return torch.unique(selected_edges.reshape(-1), sorted=True)


def reconstruct_atom_graph(
    molecule_data,
    selected_fragment_edge_mask,
):
    """Restore selected fragments, internal bonds, and selected attachments."""
    selected_fragment_edge_mask = torch.as_tensor(
        selected_fragment_edge_mask,
        dtype=torch.bool,
        device=molecule_data.fragment_edge_index.device,
    )
    num_fragment_edges = molecule_data.fragment_edge_index.size(1)
    if selected_fragment_edge_mask.numel() != num_fragment_edges:
        raise ValueError("selected_fragment_edge_mask must contain one value per fragment edge")

    if num_fragment_edges == 0:
        selected_fragments = torch.arange(
            molecule_data.fragment_x_static.size(0),
            dtype=torch.long,
            device=molecule_data.fragment_x_static.device,
        )
        selected_bond_indices = torch.empty((0,), dtype=torch.long, device=molecule_data.x.device)
    else:
        selected_fragments = selected_fragment_nodes(
            molecule_data.fragment_edge_index,
            selected_fragment_edge_mask,
        )
        if selected_fragments.numel() == 0:
            raise ValueError("At least one fragment edge must be selected when fragment edges exist")
        selected_bond_indices = molecule_data.fragment_edge_to_bond_index[selected_fragment_edge_mask]

    selected_atom_mask = torch.isin(molecule_data.atom_to_fragment, selected_fragments)
    selected_atoms = torch.nonzero(selected_atom_mask, as_tuple=False).view(-1)

    if selected_atoms.numel() == 0:
        raise ValueError("Reconstruction selected no atoms")

    bond_src = molecule_data.bond_index[0] if molecule_data.bond_index.numel() else torch.empty(0, dtype=torch.long)
    bond_dst = molecule_data.bond_index[1] if molecule_data.bond_index.numel() else torch.empty(0, dtype=torch.long)

    if molecule_data.bond_index.numel() == 0:
        selected_unique_bonds = torch.empty((0,), dtype=torch.long, device=selected_atoms.device)
    else:
        endpoint_mask = selected_atom_mask[bond_src] & selected_atom_mask[bond_dst]
        internal_mask = endpoint_mask & (
            molecule_data.atom_to_fragment[bond_src] == molecule_data.atom_to_fragment[bond_dst]
        )
        internal_bond_indices = torch.nonzero(internal_mask, as_tuple=False).view(-1)
        selected_unique_bonds = torch.unique(
            torch.cat([internal_bond_indices, selected_bond_indices.to(internal_bond_indices.device)]),
            sorted=True,
        )

    relabeled_edge_index, relabeled_atom_ids = relabel_reconstructed_atoms(
        selected_atoms,
        molecule_data.bond_index[:, selected_unique_bonds]
        if selected_unique_bonds.numel()
        else torch.empty((2, 0), dtype=torch.long, device=selected_atoms.device),
    )

    directed_edges = []
    directed_edge_attrs = []
    directed_edge_to_bond = []
    for new_bond_idx, old_bond_idx in enumerate(selected_unique_bonds.tolist()):
        src, dst = relabeled_edge_index[:, new_bond_idx].tolist()
        attr = molecule_data.bond_attr[old_bond_idx]
        directed_edges.extend([[src, dst], [dst, src]])
        directed_edge_attrs.extend([attr, attr])
        directed_edge_to_bond.extend([new_bond_idx, new_bond_idx])

    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long, device=selected_atoms.device).t().contiguous()
        edge_attr = torch.stack(directed_edge_attrs, dim=0)
        directed_edge_to_bond = torch.tensor(
            directed_edge_to_bond,
            dtype=torch.long,
            device=selected_atoms.device,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=selected_atoms.device)
        edge_attr = molecule_data.bond_attr.new_empty((0, molecule_data.bond_attr.size(1)))
        directed_edge_to_bond = torch.empty((0,), dtype=torch.long, device=selected_atoms.device)

    return Data(
        x=molecule_data.x[selected_atoms],
        edge_index=edge_index,
        edge_attr=edge_attr,
        bond_index=relabeled_edge_index,
        bond_attr=molecule_data.bond_attr[selected_unique_bonds],
        atom_ids=molecule_data.atom_ids[selected_atoms],
        bond_ids=molecule_data.bond_ids[selected_unique_bonds],
        directed_edge_to_bond=directed_edge_to_bond,
        selected_fragment_ids=selected_fragments,
        selected_original_atom_indices=relabeled_atom_ids,
    )


def relabel_reconstructed_atoms(atom_ids, edge_index):
    """Relabel selected atoms while retaining original atom provenance."""
    atom_ids = torch.as_tensor(atom_ids, dtype=torch.long, device=edge_index.device)
    old_to_new = {int(old_id): new_id for new_id, old_id in enumerate(atom_ids.tolist())}

    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device), atom_ids

    relabeled = torch.empty_like(edge_index)
    for col in range(edge_index.size(1)):
        relabeled[0, col] = old_to_new[int(edge_index[0, col])]
        relabeled[1, col] = old_to_new[int(edge_index[1, col])]
    return relabeled, atom_ids


def reconstruct_batch(batch, selected_fragment_edge_mask):
    """Reconstruct each molecule independently and return a PyG batch."""
    if not hasattr(batch, "to_data_list"):
        return reconstruct_atom_graph(batch, selected_fragment_edge_mask)

    selected_fragment_edge_mask = torch.as_tensor(selected_fragment_edge_mask, dtype=torch.bool)
    reconstructed = []
    cursor = 0
    for molecule_data in batch.to_data_list():
        num_edges = molecule_data.fragment_edge_index.size(1)
        mask = selected_fragment_edge_mask[cursor: cursor + num_edges]
        reconstructed.append(reconstruct_atom_graph(molecule_data, mask))
        cursor += num_edges

    if cursor != selected_fragment_edge_mask.numel():
        raise ValueError("selected_fragment_edge_mask has extra values after batch reconstruction")

    return Batch.from_data_list(reconstructed)
