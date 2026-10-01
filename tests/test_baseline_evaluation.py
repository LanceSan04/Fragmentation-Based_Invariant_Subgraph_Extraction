"""Validation-before-test and saved-checkpoint evaluation checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.experiments import baseline_eval, selection


class BaselineEvaluationTests(unittest.TestCase):
    def test_parity_failure_never_loads_test(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.json"
            raw.write_text("{}")
            output = root / "evaluation"
            records = {arm: {1: {"checkpoint": root / f"{arm}.pt",
                                 "checkpoint_sha256": "dummy",
                                 "selected_epoch": 1,
                                 "expected_ood_val_auc": 0.75}}
                       for arm in baseline_eval.POLICIES}
            calls = []

            def evaluate(checkpoint, split, cache, device, policy, seed):
                calls.append(split)
                return {"loss": 0.1, "acc": 0.5, "auc": 0.70}

            with patch.object(baseline_eval, "load_sources", return_value=(records, {})), \
                    patch.object(baseline_eval, "evaluate_saved", side_effect=evaluate):
                with self.assertRaisesRegex(RuntimeError, "test was not loaded"):
                    baseline_eval.run({arm: root / arm for arm in records}, root / "tune",
                                      root / "cache", raw, [1], output)
            self.assertEqual(calls, ["ood_val"])
            self.assertTrue((output / "validation_parity.json").is_file())
            self.assertFalse((output / "comparison.json").exists())

    def test_saved_checkpoints_replay_validation_and_load_test(self):
        with TemporaryDirectory() as temporary:
            for arm, policy in baseline_eval.POLICIES.items():
                with self.subTest(arm=arm):
                    output = Path(temporary) / arm
                    model = "fragment" if arm == "learned" else arm
                    selection.main(["--smoke", "--model", model, "--device", "cpu",
                                    "--output-dir", str(output)])
                    summary = json.loads((output / "summary.json").read_text())
                    expected = summary["runs"][0]["results"][policy][policy]["ood_val"]["auc"]
                    checkpoint = output / "seed5001" / f"best_{policy}.pt"
                    cache = output / "processed"
                    validation = baseline_eval.evaluate_saved(checkpoint, "ood_val", cache,
                                                              "cpu", policy, 5001)
                    test = baseline_eval.evaluate_saved(checkpoint, "ood_test", cache,
                                                        "cpu", policy, 5001)
                    self.assertAlmostEqual(validation["auc"], expected,
                                           delta=baseline_eval.PARITY_TOLERANCE)
                    self.assertGreaterEqual(test["auc"], 0.0)
                    self.assertLessEqual(test["auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
