"""Common interfaces for molecular fragmentation strategies."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FragmentationResult:
    """Tensor-friendly result required by the hierarchical model."""

    fragment_to_atoms: Any = None
    atom_to_fragment: Any = None
    fragment_edge_index: Any = None
    fragment_edge_to_bond: Any = None
    fragment_edge_to_attachment_atoms: Any = None
    fragment_types: Optional[Any] = None


class Fragmenter:
    """Base interface implemented by Group Graph and future ablations."""

    def fragment(self, mol) -> FragmentationResult:
        raise NotImplementedError


def validate_fragmentation(result: FragmentationResult, mol):
    """Validate coverage, non-overlap, indices, and original-bond mappings."""
    raise NotImplementedError
