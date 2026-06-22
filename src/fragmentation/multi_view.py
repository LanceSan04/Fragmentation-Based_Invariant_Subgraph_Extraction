"""Multi-view fragment sampling using different fragmentation methods."""

import torch
from src.fragmentation.group_graph import GroupGraphFragmenter
from src.fragmentation.base import Fragmenter


class MultiViewFragmenter:
    """Samples multiple views of a molecule using different fragmentation methods."""

    def __init__(self, methods=None):
        """Initialize with list of fragmentation method names.
        
        Args:
            methods: List of method names like ["group_graph", "brics", "recap"].
                    Defaults to ["group_graph"] for now.
        """
        self.methods = methods or ["group_graph"]
        self.fragmenters = {}
        for method in self.methods:
            self.fragmenters[method] = self._build_fragmenter(method)

    def _build_fragmenter(self, method_name):
        if method_name == "group_graph":
            return GroupGraphFragmenter()
        else:
            raise ValueError(f"Unsupported fragmentation method: {method_name}")

    def fragment_molecule(self, mol):
        """Fragment molecule using all methods.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            dict: {method_name -> FragmentationResult}
        """
        results = {}
        for method, fragmenter in self.fragmenters.items():
            results[method] = fragmenter.fragment(mol)
        return results

    def fragment_batch(self, molecule_data_list):
        """Fragment a batch of molecules.
        
        Args:
            molecule_data_list: List of MoleculeData objects with smiles attribute
            
        Returns:
            list of dicts: [{method_name -> FragmentationResult}, ...]
        """
        from rdkit import Chem
        results = []
        for data in molecule_data_list:
            mol = Chem.MolFromSmiles(data.smiles)
            results.append(self.fragment_molecule(mol))
        return results
