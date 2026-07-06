"""Pure interpretability metrics over extracted invariant subgraphs.

These functions operate on unbatched ``Data``/``ReconstructedAtomData`` objects and
plain Python/tensor inputs. They intentionally avoid any model or device coupling so
they can be reused by analysis scripts and tests.
"""

import statistics

import torch


def _num_atoms(data):
    """Number of atoms (nodes) in a molecule/subgraph Data object."""
    if getattr(data, "x", None) is None:
        return 0
    return int(data.x.size(0))


def _num_bonds(data):
    """Number of undirected bonds.

    Uses ``bond_attr`` (one row per undirected bond), not ``edge_index`` — the latter
    is the doubled/directed representation and would double-count every bond.
    """
    bond_attr = getattr(data, "bond_attr", None)
    if bond_attr is not None:
        return int(bond_attr.size(0))
    num_bonds = getattr(data, "num_bonds", None)
    if num_bonds is not None:
        return int(num_bonds)
    return 0


def _retention_pct(part, whole):
    """Percentage ``100 * part / whole``, or 0.0 when ``whole`` is 0."""
    if whole <= 0:
        return 0.0
    return 100.0 * part / whole


def subgraph_size_stats(full_data, sub_data):
    """Atom/bond counts and retention percentages for one molecule.

    Args:
        full_data: The original molecule Data (all atoms and bonds).
        sub_data: An extracted ReconstructedAtomData subgraph of that molecule.

    Returns:
        dict with atom/bond counts for both graphs and their retention percentages.
    """
    full_atoms = _num_atoms(full_data)
    sub_atoms = _num_atoms(sub_data)
    full_bonds = _num_bonds(full_data)
    sub_bonds = _num_bonds(sub_data)
    return {
        "full_atoms": full_atoms,
        "subgraph_atoms": sub_atoms,
        "atom_retention_pct": _retention_pct(sub_atoms, full_atoms),
        "full_bonds": full_bonds,
        "subgraph_bonds": sub_bonds,
        "bond_retention_pct": _retention_pct(sub_bonds, full_bonds),
    }


def _to_index_set(atoms):
    """Coerce a tensor / iterable of atom indices into a set of ints."""
    if atoms is None:
        return set()
    if isinstance(atoms, torch.Tensor):
        if atoms.numel() == 0:
            return set()
        return set(int(a) for a in atoms.reshape(-1).tolist())
    return set(int(a) for a in atoms)


def atom_jaccard(atoms_a, atoms_b):
    """Jaccard IoU between two sets of selected original atom indices.

    Accepts tensors or iterables. Two empty selections are defined as fully in
    agreement (IoU = 1.0).
    """
    set_a = _to_index_set(atoms_a)
    set_b = _to_index_set(atoms_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def pairwise_view_agreement(per_view_atom_sets):
    """Symmetric IoU matrix across all view pairs for one molecule.

    Args:
        per_view_atom_sets: list of per-view selected atom-index tensors/iterables.

    Returns:
        list[list[float]] of shape [num_views, num_views]; diagonal is 1.0.
    """
    num_views = len(per_view_atom_sets)
    matrix = [[1.0] * num_views for _ in range(num_views)]
    for i in range(num_views):
        for j in range(i + 1, num_views):
            iou = atom_jaccard(per_view_atom_sets[i], per_view_atom_sets[j])
            matrix[i][j] = iou
            matrix[j][i] = iou
    return matrix


def summarize(values):
    """Mean/median/min/max/stdev of a sequence of numbers via ``statistics``.

    Returns zeros for an empty input; stdev is 0.0 for a single value.
    """
    values = list(values)
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }
