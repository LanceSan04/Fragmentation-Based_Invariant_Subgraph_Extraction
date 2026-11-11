import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import torch
from src.evaluation.within_assay import within_assay_report
from src.experiments.granularity import interaction
from src.experiments.plans import make_plan
from src.experiments.selection import run_seed
from src.data.preprocess import run_preprocessing
from src.training.pipeline import Config
from src.training.checkpoint import load_checkpoint


class CheckpointRulesTests(unittest.TestCase):
    def test_report_excludes_single_class_and_small_groups(self):
        y = [0] * 5 + [1] * 5 + [1] * 10 + [0, 1]
        p = [0.1] * 5 + [0.9] * 5 + [0.8] * 10 + [0.2, 0.8]
        groups = ["mixed"] * 10 + ["single"] * 10 + ["small"] * 2
        result = within_assay_report(y, p, groups)
        self.assertEqual(result["eligible_assays"], 1)
        self.assertEqual(result["macro_auc"], 1)
        self.assertAlmostEqual(result["molecule_coverage"], 10 / 22)
        self.assertTrue(result["reported_only"])
        self.assertIsNone(within_assay_report([0, 1], [0.1, 0.9], [None, None])["macro_auc"])

    def test_three_rules_match_saved_epochs_and_replayed_validation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            _, base = make_plan(root, "cpu", True)
            base.update(epochs=6, compare_checkpoint_rules=True, report_within_assay=True,
                num_views=2, view_fragmentations={"atom": 2}, lambda_irm=0.0, lambda_diversity=0.0)
            run_preprocessing(Config(base))
            result = run_seed(base, root / "run", 17, policies=("independent",))
            rules = result["checkpoint_rules"]["records"]["independent"]
            self.assertEqual(set(rules), {"best_raw", "best_ma5", "final"})
            self.assertEqual(rules["final"]["epoch"], 6)
            self.assertGreaterEqual(rules["best_ma5"]["epoch"], 5)
            for rule, record in rules.items():
                payload = load_checkpoint(record["checkpoint"])
                self.assertEqual(payload["config"]["checkpoint_selection_rule"], rule)
                self.assertEqual(payload["config"]["inference_policy"], "independent")
                self.assertEqual(payload["epoch"], record["epoch"])
                replay = result["checkpoint_rules"]["results"]["independent"][rule]["ood_val"]["auc"]
                self.assertAlmostEqual(replay, record["validation_auc_at_selection"])

    def test_interaction_is_difference_of_paired_advantages(self):
        result = interaction([dict(atom_learned=.8, atom_random=.7,
                                   fragment_learned=.75, fragment_random=.72)])
        self.assertAlmostEqual(result["mean"], .07)


if __name__ == "__main__":
    unittest.main()
