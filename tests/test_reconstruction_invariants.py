"""Regression tests for invariants that silently broke and cost OOD AUC.

Each test here pins a property that was violated in the code that produced the
40-epoch EC-Assay results, and whose violation is invisible from training logs:

  * the selected subgraph must be the same size in training and evaluation,
  * the fast and slow reconstructors must gate bonds identically,
  * the atom featurizer must stay at its documented width.

Run from the repository root with:

    .\\.venv\\Scripts\\python.exe -m unittest tests.test_reconstruction_invariants -v
"""

import unittest

import torch
from torch_geometric.data import Batch

from src.data.atom_graph import ATOM_FDIM, BOND_FDIM
from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data
from src.fragmentation import build_fragmenter
from src.fragmentation.group_graph import FRAGMENT_EDGE_FEATURE_DIM
from src.models.atom_reconstructor import reconstruct_batch, reconstruct_batch_fast
from src.models.fragment_masker import (
    FRAGMENT_EDGE_DIM,
    reverse_fragment_edge_attr,
    segment_topk_mask,
)
from src.models.fragment_pooler import FRAGMENT_FEATURE_DIM


SMILES = ["CCOC(=O)C", "c1ccccc1CC(=O)NC", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]


def build_batch(smiles=SMILES, fragmentation="group_graph"):
    examples = [
        molecule_example_to_data(
            build_molecule_example(
                smi,
                label=idx % 2,
                molecule_id=f"test_{idx}",
                fragmenter=build_fragmenter(fragmentation),
            )
        )
        for idx, smi in enumerate(smiles)
    ]
    return Batch.from_data_list(examples, follow_batch=["fragment_x_static"])


class TopKSizeInvarianceTests(unittest.TestCase):
    """k must be sized from the full edge count, not the eligible subset.

    Sizing k from the eligible subset made a training subgraph
    ratio * keep_ratio * E edges while evaluation built ratio * E, so the
    classifier's BatchNorm statistics were fitted to graphs ~43% smaller than
    the ones it was scored on.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.edge_batch = torch.tensor([0] * 10 + [1] * 20)
        self.scores = torch.rand(self.edge_batch.numel())

    def selected_per_graph(self, mask):
        return torch.bincount(self.edge_batch[mask], minlength=2).tolist()

    def test_eligible_mask_does_not_shrink_selection(self):
        full = segment_topk_mask(self.scores, self.edge_batch, ratio=0.3)
        self.assertEqual(self.selected_per_graph(full), [3, 6])

        eligible = torch.rand(self.edge_batch.numel()) < 0.7
        eligible[:3] = True  # guarantee graph 0 has enough candidates
        eligible[10:17] = True
        sampled = segment_topk_mask(self.scores, self.edge_batch, ratio=0.3, eligible_mask=eligible)

        self.assertEqual(self.selected_per_graph(sampled), self.selected_per_graph(full))

    def test_selection_stays_within_eligible_edges(self):
        eligible = torch.zeros(self.edge_batch.numel(), dtype=torch.bool)
        eligible[:4] = True
        eligible[10:14] = True
        mask = segment_topk_mask(self.scores, self.edge_batch, ratio=0.3, eligible_mask=eligible)
        self.assertTrue(bool((mask & ~eligible).sum() == 0))

    def test_k_is_clipped_to_available_eligible_edges(self):
        eligible = torch.zeros(self.edge_batch.numel(), dtype=torch.bool)
        eligible[0] = True  # graph 0 has 1 eligible edge but k would be 3
        mask = segment_topk_mask(self.scores, self.edge_batch, ratio=0.3, eligible_mask=eligible)
        self.assertEqual(self.selected_per_graph(mask), [1, 0])

    def test_empty_eligible_mask_selects_nothing(self):
        eligible = torch.zeros(self.edge_batch.numel(), dtype=torch.bool)
        mask = segment_topk_mask(self.scores, self.edge_batch, ratio=0.3, eligible_mask=eligible)
        self.assertFalse(bool(mask.any()))


class ReconstructorAgreementTests(unittest.TestCase):
    """reconstruct_batch and reconstruct_batch_fast must gate bonds identically.

    Training and evaluation use the fast path; the analysis and visualization
    paths use the slow one. When they disagreed, published rationale figures
    showed a subgraph the model had never actually been run on.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.batch = build_batch()
        num_edges = self.batch.fragment_edge_index.size(1)
        self.assertGreater(num_edges, 0, "test molecules must produce fragment edges")
        self.scores = torch.rand(num_edges)
        edge_batch = self.batch.fragment_x_static_batch[self.batch.fragment_edge_index[0]]
        self.mask = segment_topk_mask(self.scores, edge_batch, ratio=0.5)

    def test_edge_weights_match(self):
        slow = reconstruct_batch(self.batch, self.mask, selected_fragment_edge_weight=self.scores)
        fast = reconstruct_batch_fast(self.batch, self.mask, selected_fragment_edge_weight=self.scores)

        self.assertEqual(fast.x.size(0), slow.x.size(0))
        self.assertEqual(fast.edge_index.size(1), slow.edge_index.size(1))
        # Edge emission order differs by design, so compare as multisets.
        torch.testing.assert_close(
            torch.sort(fast.edge_weight).values,
            torch.sort(slow.edge_weight).values,
        )

    def test_internal_bonds_are_gated_not_left_at_one(self):
        fast = reconstruct_batch_fast(self.batch, self.mask, selected_fragment_edge_weight=self.scores)
        # Scores are drawn from U(0,1); an all-ones gate would mean internal
        # bonds were never gated at all, the pre-fix slow-path behavior.
        self.assertLess(float(fast.edge_weight.max()), 1.0)

    def test_uniform_weights_give_all_ones_gate(self):
        ones = torch.ones_like(self.scores)
        fast = reconstruct_batch_fast(self.batch, self.mask, selected_fragment_edge_weight=ones)
        slow = reconstruct_batch(self.batch, self.mask, selected_fragment_edge_weight=ones)
        torch.testing.assert_close(fast.edge_weight, torch.ones_like(fast.edge_weight))
        torch.testing.assert_close(slow.edge_weight, torch.ones_like(slow.edge_weight))


class FeatureWidthTests(unittest.TestCase):
    """Pin the featurizer widths.

    Every other test builds synthetic tensors at hardcoded dimensions, so a
    regression in the real featurizer would not fail any of them.
    """

    def test_atom_and_bond_widths(self):
        self.assertEqual(ATOM_FDIM, 46)
        self.assertEqual(BOND_FDIM, 14)

    def test_fragment_widths_follow_atom_and_bond_widths(self):
        self.assertEqual(FRAGMENT_FEATURE_DIM, ATOM_FDIM + BOND_FDIM + 2)
        self.assertEqual(FRAGMENT_FEATURE_DIM, 62)
        self.assertEqual(FRAGMENT_EDGE_FEATURE_DIM, (2 * ATOM_FDIM) + BOND_FDIM + 2)
        self.assertEqual(FRAGMENT_EDGE_FEATURE_DIM, 108)
        self.assertEqual(FRAGMENT_EDGE_DIM, FRAGMENT_EDGE_FEATURE_DIM)

    def test_built_tensors_have_the_pinned_widths(self):
        batch = build_batch(["CCOC(=O)C"])
        self.assertEqual(batch.x.size(1), ATOM_FDIM)
        self.assertEqual(batch.bond_attr.size(1), BOND_FDIM)
        self.assertEqual(batch.fragment_x_static.size(1), FRAGMENT_FEATURE_DIM)
        self.assertEqual(batch.fragment_edge_attr.size(1), FRAGMENT_EDGE_FEATURE_DIM)


class ReverseFragmentEdgeAttrTests(unittest.TestCase):
    """The masker symmetrizes the fragment graph; the reversed direction must
    carry src/dst-swapped features, not a copy of the forward ones."""

    def test_swap_is_an_involution(self):
        attr = torch.rand(4, FRAGMENT_EDGE_DIM)
        torch.testing.assert_close(
            reverse_fragment_edge_attr(reverse_fragment_edge_attr(attr)), attr
        )

    def test_swap_moves_the_expected_blocks(self):
        attr = torch.rand(3, FRAGMENT_EDGE_DIM)
        flipped = reverse_fragment_edge_attr(attr)
        torch.testing.assert_close(flipped[:, :ATOM_FDIM], attr[:, ATOM_FDIM : 2 * ATOM_FDIM])
        torch.testing.assert_close(flipped[:, ATOM_FDIM : 2 * ATOM_FDIM], attr[:, :ATOM_FDIM])
        # bond block is direction-agnostic and must be untouched
        bond = slice(2 * ATOM_FDIM, 2 * ATOM_FDIM + BOND_FDIM)
        torch.testing.assert_close(flipped[:, bond], attr[:, bond])
        # the two fragment-size scalars swap
        torch.testing.assert_close(flipped[:, -2], attr[:, -1])
        torch.testing.assert_close(flipped[:, -1], attr[:, -2])

    def test_unknown_width_passes_through(self):
        attr = torch.rand(2, 7)
        torch.testing.assert_close(reverse_fragment_edge_attr(attr), attr)
        self.assertIsNone(reverse_fragment_edge_attr(None))


if __name__ == "__main__":
    unittest.main()
