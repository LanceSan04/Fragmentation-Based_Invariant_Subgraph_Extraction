import torch

from src.data.preprocess import build_molecule_data
from src.models.atom_reconstructor import reconstruct_atom_graph

SMILES_SAMPLES = [
    "CCO",
    "CC(=O)N",
    "c1ccccc1O",
]


def check_fragmentation(smiles):
    data = build_molecule_data(smiles, label=1.0)

    assert data.x.shape[0] == data.atom_to_fragment.numel(), "Atom count and fragment mapping length differ"
    num_fragments = int(data.fragment_x_static.size(0))
    assert num_fragments > 0, "Expected at least one fragment"
    assert torch.unique(data.atom_to_fragment).numel() == num_fragments, "Fragment IDs should cover all fragments"

    if data.fragment_edge_index.numel() > 0:
        assert int(data.fragment_edge_index.max()) < num_fragments, "Fragment edge references a missing fragment"

    if data.fragment_edge_to_attachment_atoms.numel() > 0:
        assert data.fragment_edge_to_attachment_atoms.shape[0] == 2, "Attachment mapping must have shape [2, num_edges]"
        assert int(data.fragment_edge_to_attachment_atoms.max()) < data.x.shape[0], "Attachment atom index out of range"

    selected_mask = torch.ones((data.fragment_edge_index.size(1),), dtype=torch.bool)
    reconstructed = reconstruct_atom_graph(data, selected_mask)

    assert reconstructed.x.shape[0] <= data.x.shape[0], "Reconstructed graph has too many atoms"
    print(
        f"{smiles}: atoms={data.x.shape[0]} fragments={num_fragments} "
        f"frag_edges={data.fragment_edge_index.size(1)} reconstructed_atoms={reconstructed.x.shape[0]}"
    )


def main():
    for smiles in SMILES_SAMPLES:
        check_fragmentation(smiles)
    print("Fragmentation test passed.")


if __name__ == "__main__":
    main()
