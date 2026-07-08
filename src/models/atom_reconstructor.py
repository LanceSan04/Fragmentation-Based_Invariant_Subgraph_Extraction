"""Reconstruct atom-level invariant graphs from selected fragment edges."""

import torch
from torch_geometric.data import Batch, Data


class ReconstructedAtomData(Data):
    """Atom-level graph reconstructed from selected fragment evidence."""

    @property
    def num_bonds(self):
        if hasattr(self, "bond_attr") and self.bond_attr is not None:
            return int(self.bond_attr.size(0))
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        if key == "bond_index":
            return self.num_nodes
        if key == "directed_edge_to_bond":
            return self.num_bonds
        if key == "selected_attachment_node_index":
            return self.num_nodes
        if key in {
            "atom_ids",
            "bond_ids",
            "selected_fragment_ids",
            "selected_fragment_edge_ids",
            "selected_attachment_bond_ids",
            "selected_attachment_bond_indices",
            "selected_attachment_atoms",
            "reconstructed_original_bond_indices",
            "selected_original_atom_indices",
        }:
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {"bond_index", "selected_attachment_atoms", "selected_attachment_node_index"}:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


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
    selected_fragment_edge_weight=None,
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
    if selected_fragment_edge_weight is not None:
        selected_fragment_edge_weight = torch.as_tensor(
            selected_fragment_edge_weight,
            dtype=molecule_data.x.dtype,
            device=molecule_data.fragment_edge_index.device,
        )
        if selected_fragment_edge_weight.numel() != num_fragment_edges:
            raise ValueError(
                "selected_fragment_edge_weight must contain one value per fragment edge"
            )

    if num_fragment_edges == 0:
        selected_fragments = torch.arange(
            molecule_data.fragment_x_static.size(0),
            dtype=torch.long,
            device=molecule_data.fragment_x_static.device,
        )
        selected_fragment_edge_ids = torch.empty(
            (0,),
            dtype=torch.long,
            device=molecule_data.fragment_edge_index.device,
        )
        selected_bond_indices = torch.empty((0,), dtype=torch.long, device=molecule_data.x.device)
        selected_attachment_bond_ids = torch.empty((0,), dtype=torch.long, device=molecule_data.x.device)
        selected_attachment_atoms = torch.empty((2, 0), dtype=torch.long, device=molecule_data.x.device)
        selected_edge_weights = molecule_data.x.new_empty((0,))
    else:
        selected_fragment_edge_ids = torch.nonzero(
            selected_fragment_edge_mask,
            as_tuple=False,
        ).view(-1)
        selected_fragments = selected_fragment_nodes(
            molecule_data.fragment_edge_index,
            selected_fragment_edge_mask,
        )
        if selected_fragments.numel() == 0:
            raise ValueError("At least one fragment edge must be selected when fragment edges exist")
        selected_bond_indices = molecule_data.fragment_edge_to_bond_index[selected_fragment_edge_mask]
        selected_attachment_bond_ids = molecule_data.fragment_edge_to_bond[selected_fragment_edge_mask]
        selected_attachment_atoms = molecule_data.fragment_edge_to_attachment_atoms[
            :,
            selected_fragment_edge_mask,
        ]
        if selected_fragment_edge_weight is None:
            selected_edge_weights = molecule_data.x.new_ones((selected_fragment_edge_ids.numel(),))
        else:
            selected_edge_weights = selected_fragment_edge_weight[selected_fragment_edge_mask].to(
                molecule_data.x.device
            )

    selected_atom_mask = torch.isin(molecule_data.atom_to_fragment, selected_fragments)
    selected_atoms = torch.nonzero(selected_atom_mask, as_tuple=False).view(-1)

    if selected_atoms.numel() == 0:
        raise ValueError("Reconstruction selected no atoms")

    bond_src = (
        molecule_data.bond_index[0]
        if molecule_data.bond_index.numel()
        else torch.empty(0, dtype=torch.long, device=selected_atoms.device)
    )
    bond_dst = (
        molecule_data.bond_index[1]
        if molecule_data.bond_index.numel()
        else torch.empty(0, dtype=torch.long, device=selected_atoms.device)
    )

    if molecule_data.bond_index.numel() == 0:
        selected_unique_bonds = torch.empty((0,), dtype=torch.long, device=selected_atoms.device)
        internal_bond_indices = torch.empty((0,), dtype=torch.long, device=selected_atoms.device)
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

    if selected_unique_bonds.numel():
        reconstructed_bond_is_internal = torch.isin(selected_unique_bonds, internal_bond_indices)
        reconstructed_bond_is_selected_boundary = torch.isin(
            selected_unique_bonds,
            selected_bond_indices.to(selected_unique_bonds.device),
        )
    else:
        reconstructed_bond_is_internal = torch.empty(
            (0,),
            dtype=torch.bool,
            device=selected_atoms.device,
        )
        reconstructed_bond_is_selected_boundary = torch.empty(
            (0,),
            dtype=torch.bool,
            device=selected_atoms.device,
        )

    reconstructed_bond_weight = build_reconstructed_bond_weight(
        selected_unique_bonds,
        selected_bond_indices,
        selected_edge_weights,
        dtype=molecule_data.x.dtype,
        device=selected_atoms.device,
    )

    relabeled_edge_index, relabeled_atom_ids = relabel_reconstructed_atoms(
        selected_atoms,
        molecule_data.bond_index[:, selected_unique_bonds]
        if selected_unique_bonds.numel()
        else torch.empty((2, 0), dtype=torch.long, device=selected_atoms.device),
    )
    selected_attachment_node_index = relabel_attachment_atoms(
        selected_atoms,
        selected_attachment_atoms,
    )

    if selected_unique_bonds.numel():
        num_bonds = selected_unique_bonds.numel()
        src = relabeled_edge_index[0]  # [num_bonds]
        dst = relabeled_edge_index[1]  # [num_bonds]
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ], dim=0)
        bond_attrs = molecule_data.bond_attr[selected_unique_bonds]
        edge_attr = torch.cat([bond_attrs, bond_attrs], dim=0)
        bond_range = torch.arange(num_bonds, dtype=torch.long, device=selected_atoms.device)
        directed_edge_to_bond = torch.cat([bond_range, bond_range])
        edge_weight = torch.cat([reconstructed_bond_weight, reconstructed_bond_weight])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=selected_atoms.device)
        edge_attr = molecule_data.bond_attr.new_empty((0, molecule_data.bond_attr.size(1)))
        directed_edge_to_bond = torch.empty((0,), dtype=torch.long, device=selected_atoms.device)
        edge_weight = molecule_data.x.new_empty((0,))

    return ReconstructedAtomData(
        x=molecule_data.x[selected_atoms],
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
        bond_index=relabeled_edge_index,
        bond_attr=molecule_data.bond_attr[selected_unique_bonds],
        atom_ids=molecule_data.atom_ids[selected_atoms],
        bond_ids=molecule_data.bond_ids[selected_unique_bonds],
        directed_edge_to_bond=directed_edge_to_bond,
        selected_fragment_ids=selected_fragments,
        selected_fragment_edge_ids=selected_fragment_edge_ids,
        selected_fragment_edge_weight=selected_edge_weights,
        selected_attachment_bond_ids=selected_attachment_bond_ids,
        selected_attachment_bond_indices=selected_bond_indices,
        selected_attachment_atoms=selected_attachment_atoms,
        selected_attachment_node_index=selected_attachment_node_index,
        reconstructed_bond_is_internal=reconstructed_bond_is_internal,
        reconstructed_bond_is_selected_boundary=reconstructed_bond_is_selected_boundary,
        reconstructed_bond_weight=reconstructed_bond_weight,
        reconstructed_original_bond_indices=selected_unique_bonds,
        selected_original_atom_indices=relabeled_atom_ids,
    )


def relabel_reconstructed_atoms(atom_ids, edge_index):
    """Relabel selected atoms while retaining original atom provenance using fast vectorization."""
    atom_ids = torch.as_tensor(atom_ids, dtype=torch.long, device=edge_index.device)

    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device), atom_ids

    max_id = int(edge_index.max())
    
    mapping = torch.zeros(max_id + 1, dtype=torch.long, device=edge_index.device)
    mapping[atom_ids] = torch.arange(atom_ids.numel(), device=edge_index.device)

    relabeled = mapping[edge_index]
    
    return relabeled, atom_ids


def relabel_attachment_atoms(selected_atoms, selected_attachment_atoms):
    """Map original attachment atom IDs into reconstructed node indices."""
    if selected_attachment_atoms.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=selected_attachment_atoms.device)

    selected_atoms = torch.as_tensor(
        selected_atoms,
        dtype=torch.long,
        device=selected_attachment_atoms.device,
    )
    max_id = int(selected_atoms.max())
    mapping = torch.full((max_id + 1,), -1, dtype=torch.long, device=selected_attachment_atoms.device)
    mapping[selected_atoms] = torch.arange(selected_atoms.numel(), device=selected_attachment_atoms.device)
    relabeled = mapping[selected_attachment_atoms]
    if (relabeled < 0).any():
        raise ValueError("selected attachment atom is not in the reconstructed atom set")
    return relabeled


def build_reconstructed_bond_weight(
    selected_unique_bonds,
    selected_bond_indices,
    selected_edge_weights,
    dtype,
    device,
):
    """Assign selected fragment-edge scores to reconstructed boundary bonds."""
    selected_unique_bonds = torch.as_tensor(selected_unique_bonds, dtype=torch.long, device=device)
    weights = torch.ones((selected_unique_bonds.numel(),), dtype=dtype, device=device)
    if selected_unique_bonds.numel() == 0 or selected_bond_indices.numel() == 0:
        return weights

    selected_bond_indices = torch.as_tensor(selected_bond_indices, dtype=torch.long, device=device)
    selected_edge_weights = torch.as_tensor(selected_edge_weights, dtype=dtype, device=device)
    # selected_unique_bonds is sorted (from torch.unique); searchsorted gives each bond's position
    positions = torch.searchsorted(selected_unique_bonds, selected_bond_indices)
    weights[positions] = selected_edge_weights + 1
    return weights


def reconstruct_batch(batch, selected_fragment_edge_mask, selected_fragment_edge_weight=None):
    """Reconstruct each molecule independently and return a PyG batch."""
    if not hasattr(batch, "to_data_list"):
        return reconstruct_atom_graph(
            batch,
            selected_fragment_edge_mask,
            selected_fragment_edge_weight=selected_fragment_edge_weight,
        )

    selected_fragment_edge_mask = torch.as_tensor(selected_fragment_edge_mask, dtype=torch.bool)
    if selected_fragment_edge_weight is not None:
        selected_fragment_edge_weight = torch.as_tensor(selected_fragment_edge_weight)
        if selected_fragment_edge_weight.numel() != selected_fragment_edge_mask.numel():
            raise ValueError(
                "selected_fragment_edge_weight must contain one value per selected mask entry"
            )
    reconstructed = []
    cursor = 0
    for molecule_data in batch.to_data_list():
        num_edges = molecule_data.fragment_edge_index.size(1)
        mask = selected_fragment_edge_mask[cursor: cursor + num_edges]
        if selected_fragment_edge_weight is None:
            edge_weight = None
        else:
            edge_weight = selected_fragment_edge_weight[cursor: cursor + num_edges]
        reconstructed.append(
            reconstruct_atom_graph(
                molecule_data,
                mask,
                selected_fragment_edge_weight=edge_weight,
            )
        )
        cursor += num_edges

    if cursor != selected_fragment_edge_mask.numel():
        raise ValueError("selected_fragment_edge_mask has extra values after batch reconstruction")

    return Batch.from_data_list(reconstructed)


def reconstruct_batch_fast(batch, selected_fragment_edge_mask, selected_fragment_edge_weight=None):
    """Vectorized classifier-only atom-subgraph reconstruction for a batch.

    Produces exactly the tensors :class:`AtomClassifier` consumes (``x``,
    ``edge_index``, ``edge_attr``, ``edge_weight``, ``batch``) by operating on
    the globally-offset batch tensors directly, avoiding the per-molecule
    ``to_data_list`` / ``from_data_list`` round-trip in :func:`reconstruct_batch`.

    The classifier (GATv2 + mean pool) is order-invariant over edges, so edges
    are emitted in a different order than :func:`reconstruct_batch`, but ``x`` and
    ``batch`` share the same molecule-major node order and ``edge_attr`` /
    ``edge_weight`` stay aligned column-for-column with ``edge_index`` — giving
    identical GNN output. Provenance fields (atom_ids, selected_fragment_ids, …)
    are intentionally omitted; use :func:`reconstruct_batch` when those are needed
    (analysis / visualization paths).
    """
    device = batch.x.device
    mask = torch.as_tensor(selected_fragment_edge_mask, dtype=torch.bool, device=device)

    frag_edge_index = batch.fragment_edge_index
    if frag_edge_index.size(1) != mask.numel():
        raise ValueError("selected_fragment_edge_mask must contain one value per fragment edge")

    atom_to_fragment = batch.atom_to_fragment
    frag_node_batch = batch.fragment_x_static_batch

    # Molecules with no fragment edges keep all their fragments — matches the
    # num_fragment_edges == 0 branch of reconstruct_atom_graph.
    if getattr(batch, "ptr", None) is not None:
        num_mols = batch.ptr.numel() - 1
    else:
        num_mols = int(batch.batch.max().item()) + 1 if batch.batch.numel() else 0
    mol_has_edge = torch.zeros(num_mols, dtype=torch.bool, device=device)
    if frag_edge_index.numel():
        mol_has_edge[frag_node_batch[frag_edge_index[0]]] = True
    edgeless_fragments = (~mol_has_edge[frag_node_batch]).nonzero(as_tuple=True)[0]

    selected_endpoints = frag_edge_index[:, mask].reshape(-1)
    selected_fragments = torch.unique(torch.cat([selected_endpoints, edgeless_fragments]))

    selected_atom_mask = torch.isin(atom_to_fragment, selected_fragments)
    selected_atoms = selected_atom_mask.nonzero(as_tuple=True)[0]

    num_atoms = batch.x.size(0)
    atom_map = torch.full((num_atoms,), -1, dtype=torch.long, device=device)
    atom_map[selected_atoms] = torch.arange(selected_atoms.numel(), device=device)

    bond_index = batch.bond_index
    boundary_bonds = batch.fragment_edge_to_bond_index[mask]
    if bond_index.numel():
        bsrc, bdst = bond_index[0], bond_index[1]
        endpoint_mask = selected_atom_mask[bsrc] & selected_atom_mask[bdst]
        internal_mask = endpoint_mask & (atom_to_fragment[bsrc] == atom_to_fragment[bdst])
        internal_bonds = internal_mask.nonzero(as_tuple=True)[0]
        selected_bonds = torch.unique(torch.cat([internal_bonds, boundary_bonds]))
    else:
        selected_bonds = torch.empty((0,), dtype=torch.long, device=device)

    # Per-bond weight: internal bonds 1.0, selected boundary bonds score + 1
    # (matches build_reconstructed_bond_weight). selected_bonds is sorted.
    bond_weight = torch.ones(selected_bonds.numel(), dtype=batch.x.dtype, device=device)
    if selected_bonds.numel() and boundary_bonds.numel():
        if selected_fragment_edge_weight is None:
            boundary_scores = torch.ones(boundary_bonds.numel(), dtype=batch.x.dtype, device=device)
        else:
            weight = torch.as_tensor(selected_fragment_edge_weight, dtype=batch.x.dtype, device=device)
            boundary_scores = weight[mask]
        positions = torch.searchsorted(selected_bonds, boundary_bonds)
        bond_weight[positions] = boundary_scores + 1

    if selected_bonds.numel():
        src_local = atom_map[bond_index[0, selected_bonds]]
        dst_local = atom_map[bond_index[1, selected_bonds]]
        edge_index = torch.stack(
            [torch.cat([src_local, dst_local]), torch.cat([dst_local, src_local])],
            dim=0,
        )
        bond_attrs = batch.bond_attr[selected_bonds]
        edge_attr = torch.cat([bond_attrs, bond_attrs], dim=0)
        edge_weight = torch.cat([bond_weight, bond_weight])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = batch.bond_attr.new_empty((0, batch.bond_attr.size(1)))
        edge_weight = batch.x.new_empty((0,))

    return Data(
        x=batch.x[selected_atoms],
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
        batch=batch.batch[selected_atoms],
    )
