"""Guard the planned discovery/confirmation split and selection rule."""

import unittest

from src.experiments.fragment_ratio_sweep import (
    BASELINE, CONFIRMATION_SEEDS, DISCOVERY_SEEDS, choose_winner, grid, label,
)


class FragmentRatioSweepTests(unittest.TestCase):
    def test_budget_and_fresh_confirmation_seeds(self):
        candidates = grid()
        self.assertEqual(len(candidates), 6)
        self.assertIn(BASELINE, candidates)
        self.assertTrue(all(sample < 1 for _, sample in candidates))
        self.assertFalse(set(DISCOVERY_SEEDS) & set(CONFIRMATION_SEEDS))
        self.assertEqual(len(candidates) * len(DISCOVERY_SEEDS)
                         + 2 * len(CONFIRMATION_SEEDS), 22)

    def test_selection_uses_mean_and_fixed_tie_order(self):
        candidates = grid()
        scores = {label(*c):[.75,.75] for c in candidates}
        self.assertEqual(choose_winner(scores,candidates), candidates[0])
        scores[label(*candidates[-1])] = [.73,.79]
        self.assertEqual(choose_winner(scores,candidates), candidates[-1])
        scores[label(*candidates[-1])] = [.7,float('nan')]
        with self.assertRaises(ValueError):choose_winner(scores,candidates)


if __name__ == "__main__":
    unittest.main()
