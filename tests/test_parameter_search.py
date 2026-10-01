from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from src.experiments.parameter_search import plan
from src.experiments.search import run_search


class ParameterSearchTests(unittest.TestCase):
    def test_size_uses_own_cache_and_current_reference(self):
        spec, base = plan(Path("unused"), "cpu", dataset="size")
        self.assertEqual(base["dataset"], "drugood_lbap_core_ec50_size")
        self.assertTrue(base["processed_dir"].endswith("size"))
        self.assertEqual(spec["base_seed"], 401)
        self.assertEqual(spec["enqueue_trials"][0]["learning_rate"], 0.0003)
        self.assertEqual(spec["enqueue_trials"][0]["edge_ratio"], 0.5)
        self.assertEqual(spec["grid"]["fragment_sample_ratio"], [0.7, 0.9])
        self.assertEqual(base["loader_split_names"], ["train", "ood_val"])
        self.assertIsNone(base["test_split"])

    def test_fixed_protocol_and_reference(self):
        spec, base = plan(Path("unused"), "cpu")
        self.assertEqual(spec["n_trials"] * spec["search_repeats"], 48)
        self.assertEqual(spec["enqueue_trials"][0], {k: base[k] for k in spec["grid"]})
        self.assertEqual(
            (base["lambda_contrast"], base["lambda_diversity"], base["lambda_irm"]), (0.1, 0, 0)
        )
        self.assertEqual(base["loader_split_names"], ["train", "ood_val"])
        self.assertTrue(base["early_stopping"])
        self.assertEqual(base["selection_ma_window"], 1)

    def test_reference_is_first_and_seeds_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, base = plan(Path(directory), "cpu", True)
            calls = []

            def fake(*args, **kwargs):
                calls.append((args[2], args[3], kwargs["evaluate_test"]))
                return {}, {"val_auc": [0.6, 0.7]}

            args = SimpleNamespace(
                epochs=None, device="cpu", dry_run=False, trials=None, repeats=None
            )
            with patch("src.experiments.search.run_once", side_effect=fake):
                run_search(spec, base, args)
            self.assertEqual(calls[0][0], spec["enqueue_trials"][0])
            self.assertEqual([r[1] for r in calls], [201, 202, 201, 202])
            self.assertTrue(all(not r[2] for r in calls))


if __name__ == "__main__":
    unittest.main()
