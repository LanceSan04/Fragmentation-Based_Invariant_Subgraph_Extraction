"""Check fixed transfer settings and validation-before-test evaluation gate."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from src.experiments.cross_domain import (
    ARMS, DATASETS, POLICY, SEEDS, arm_config, dataset_config, evaluate_frozen, summary,
)


class CrossDomainTests(unittest.TestCase):
    def test_frozen_configuration_has_no_historical_size_overrides(self):
        for dataset in DATASETS:
            base = dataset_config(dataset, "raw", "cache", "cpu")
            self.assertEqual(base["learning_rate"], .0003)
            self.assertEqual(base["dropout"], .5)
            self.assertEqual(base["edge_ratio"], .5)
            self.assertEqual(base["fragment_sample_ratio"], .7)
            self.assertEqual(base["num_views"], 5)
            self.assertFalse(base["early_stopping"])
            self.assertEqual(base["loader_split_names"], ["train", "ood_val"])
            self.assertIsNone(base["test_split"])
            self.assertTrue(base["strict"])
        self.assertEqual(len(SEEDS) * len(ARMS), 15)

    def test_random_differs_only_in_selection_and_erm_is_prediction_only(self):
        base = dataset_config("ec50_size", "raw", "cache", "cpu")
        learned, random = arm_config(base, "learned"), arm_config(base, "random")
        self.assertEqual({k for k in learned if learned[k] != random[k]}, {"selection_mode"})
        erm = arm_config(base, "erm")
        self.assertTrue(erm["erm_baseline"])
        self.assertEqual([erm[k] for k in ("lambda_contrast", "lambda_diversity", "lambda_irm")], [0., 0., 0.])

    def entries(self, output):
        entries = []
        for arm in ARMS:
            path = output / f"{arm}.pt"
            path.write_bytes(arm.encode())
            entries.append(dict(dataset="ec50_size", arm=arm, seed=8701,
                                policy=POLICY[arm], checkpoint=str(path), cache="cache",
                                selected_epoch=7, selected_auc=.8))
        return entries

    def checkpoint(self, path, **kwargs):
        arm = Path(path).stem
        return dict(epoch=7, config=dict(seed=8701, inference_policy=POLICY[arm]))

    def test_failed_validation_never_loads_test(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            calls = []
            def evaluate(checkpoint, split, *args):
                calls.append(split)
                return dict(auc=.6, acc=.5, loss=.7)
            with patch("src.experiments.cross_domain.load_checkpoint", side_effect=self.checkpoint), \
                 patch("src.experiments.cross_domain.evaluate_saved", side_effect=evaluate):
                with self.assertRaisesRegex(RuntimeError, "no test"):
                    evaluate_frozen(self.entries(out), out, "cpu", True)
            self.assertEqual(calls, ["ood_val"])

    def test_all_validation_precedes_any_test(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            calls = []
            def evaluate(checkpoint, split, *args):
                calls.append(split)
                return dict(auc=np.float64(.8), acc=.5, loss=.7)
            with patch("src.experiments.cross_domain.load_checkpoint", side_effect=self.checkpoint), \
                 patch("src.experiments.cross_domain.evaluate_saved", side_effect=evaluate):
                result = evaluate_frozen(self.entries(out), out, "cpu", True)
            self.assertEqual(calls, ["ood_val"] * 3 + ["ood_test"] * 3)
            self.assertTrue(result["test_evaluated"])
            self.assertEqual(result["datasets"]["ec50_size"]["summary"]["ood_test"]["arms"]["erm"]["n"], 1)

    def test_paired_differences_are_not_best_seed_selection(self):
        rows = [{arm: {"ood_val": {"auc": values[i]}} for i, arm in enumerate(ARMS)}
                for values in ((.8, .7, .9), (.6, .7, .5))]
        result = summary(rows, "ood_val")
        self.assertAlmostEqual(result["arms"]["learned"]["mean"], .7)
        self.assertAlmostEqual(result["paired_differences"]["learned_minus_random"]["mean"], 0)
        self.assertEqual(result["paired_differences"]["learned_minus_random"]["positive_seeds"], 1)


if __name__ == "__main__":
    unittest.main()
