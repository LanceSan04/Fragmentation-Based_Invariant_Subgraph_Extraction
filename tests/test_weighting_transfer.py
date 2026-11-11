import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.experiments.scorer_paths import main, TRANSFER_ARMS
from src.evaluation.paired_uncertainty import cluster_difference


class WeightingTransferTests(unittest.TestCase):
    def test_gating_mechanism_has_full_graph_and_explicit_erm_contrast(self):
        with tempfile.TemporaryDirectory() as root:
            plan = main(["--profile", "gating_mechanism", "--cache-root", root,
                         "--output-dir", str(Path(root) / "results"), "--dry-run"])
        self.assertEqual(plan["training_runs"], 50)
        self.assertEqual(plan["seeds"], list(range(9401, 9406)))
        full = plan["arm_settings"]["full_graph_gated"]
        self.assertEqual(full["edge_ratio"], 1.)
        self.assertEqual(full["fragment_sample_ratio"], 1.)
        self.assertEqual(full["scorer_context"], "full")
        self.assertTrue(plan["arm_settings"]["erm_contrast"]["erm_contrast_control"])
        self.assertEqual(plan["arm_settings"]["erm_contrast"]["lambda_contrast"], .1)

    def test_good_weighting_has_matched_controls_and_no_assay_metric(self):
        with tempfile.TemporaryDirectory() as root:
            plan = main(["--profile", "good_weighting", "--processed-dir", root,
                         "--output-dir", str(Path(root) / "results"), "--dry-run"])
        self.assertEqual(plan["training_runs"], 15)
        self.assertEqual(plan["seeds"], list(range(9301, 9306)))
        self.assertEqual(plan["datasets"], ["good_hiv_scaffold"])
        self.assertFalse(plan["configs"]["good_hiv_scaffold"]["report_within_assay"])
        self.assertEqual(set(plan["arms"]), {"random_learned_gates", "random_unit_gates", "erm"})

    def test_atom_ranking_profile_is_two_arms_at_fixed_target(self):
        with tempfile.TemporaryDirectory() as root:
            plan = main(["--profile", "atom_ranking", "--cache-root", root,
                         "--output-dir", str(Path(root) / "results"), "--dry-run"])
        self.assertEqual(plan["training_runs"], 20)
        self.assertEqual(plan["seeds"], list(range(9201, 9206)))
        self.assertEqual(plan["primary_comparison"], "learned_vs_random_ranking")
        settings = plan["arm_settings"]
        for arm in settings.values():
            self.assertEqual(arm["selection_budget"], "atoms")
            self.assertEqual(arm["atom_retention_ratio"], .65)
            self.assertEqual(arm["message_gate_mode"], "learned")
            self.assertEqual(arm["selection_estimator"], "hard")
        a, b = (dict(settings[name]) for name in ("hard_learned_gates", "random_learned_gates"))
        self.assertEqual(a.pop("ranking_mode"), "learned")
        self.assertEqual(b.pop("ranking_mode"), "random")
        self.assertEqual(a, b)

    def test_fixed_recipe_and_fresh_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            plan = main(["--profile", "weighting_transfer", "--cache-root", root,
                         "--output-dir", str(Path(root) / "results"), "--dry-run"])
        self.assertEqual(plan["training_runs"], 40)
        self.assertEqual(plan["seeds"], list(range(9101, 9106)))
        self.assertEqual(len(plan["arms"]), 4)
        self.assertEqual(plan["primary_rule"], "best_raw")
        for config in plan["configs"].values():
            self.assertEqual(config["view_fragmentations"], {"group_graph": 5})
            self.assertEqual(config["lambda_contrast"], .1)
            self.assertFalse(config["early_stopping"])
        self.assertEqual(TRANSFER_ARMS["erm"]["lambda_contrast"], 0)

    def test_bootstrap_identical_predictions_and_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            a, b = Path(root) / "a.npz", Path(root) / "b.npz"
            values = dict(labels=np.array([0, 1, 0, 1]), probabilities=np.array([.1, .9, .2, .8]),
                          assays=np.array(["a", "a", "b", "b"]), row_index=np.arange(4))
            np.savez(a, **values)
            np.savez(b, **values)
            result = cluster_difference([a], [b], repeats=20)
            self.assertEqual(result["percentile_95"], [0., 0.])
            values["row_index"] = np.arange(4)[::-1]
            np.savez(b, **values)
            with self.assertRaisesRegex(ValueError, "membership differs"):
                cluster_difference([a], [b], repeats=20)
