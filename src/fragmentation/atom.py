"""Singleton control: atoms are fragments and bonds are connections."""

from src.fragmentation.base import Fragmenter, validate_fragmentation
from src.fragmentation.group_graph import build_fragmentation_result


class AtomFragmenter(Fragmenter):
    def fragment(self, mol):
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("Atom fragmentation requires a nonempty molecule")
        # Reuse canonical bond provenance and the standard feature layout.
        clusters = {(i,): "atom" for i in range(mol.GetNumAtoms())}
        return validate_fragmentation(build_fragmentation_result(mol, clusters), mol)
