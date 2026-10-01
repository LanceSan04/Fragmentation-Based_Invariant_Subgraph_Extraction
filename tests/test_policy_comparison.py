import contextlib
import io
import tempfile
import unittest

import torch
from tests.test_selection_and_erm import ScriptedTrainer
from src.training.policy_comparison import PolicyComparison
from src.training.checkpoint import load_checkpoint
from src.training.pipeline import Config, run_training


class PolicyComparisonTests(unittest.TestCase):
    def test_two_actual_snapshots_on_fixed_trajectory(self):
        scores = [0.72, 0.76, 0.75, 0.73, 0.67, 0.6, 0.6, 0.6]
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config(
                dict(
                    run_dir=directory,
                    epochs=8,
                    seed=11,
                    validation_split="valid",
                    selection_ma_window=5,
                )
            )
            recorder = PolicyComparison(cfg, {})
            trainer = ScriptedTrainer(scores, selection_ma_window=5)
            with contextlib.redirect_stdout(io.StringIO()):
                history = trainer.train(
                    [], [None], num_epochs=8, patience=None, epoch_callback=recorder.observe
                )
                report = recorder.finish(history)
            self.assertEqual(len(history["train_loss"]), 8)
            for name, epoch in [("best_raw", 2), ("best_ma_endpoint", 5)]:
                record = report["policies"][name]
                self.assertEqual(record["epoch"], epoch)
                checkpoint = load_checkpoint(record["checkpoint"])
                self.assertEqual(checkpoint["view_state_dicts"][0]["weight"].item(), epoch)
                self.assertNotIn("optimizer_state_dicts", checkpoint)
            self.assertAlmostEqual(
                report["policies"]["best_ma_endpoint"]["validation_auc_at_selection"], 0.67
            )

    def test_saving_does_not_consume_rng_or_change_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = PolicyComparison(
                Config(dict(run_dir=directory, validation_split="valid")), {}
            )
            trainer = ScriptedTrainer([0.7])
            before = trainer.views[0].weight.detach().clone()
            rng = torch.get_rng_state().clone()
            recorder.observe(trainer, 1, 0.7, None, "auc")
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            torch.testing.assert_close(before, trainer.views[0].weight)

    def test_guard_rejects_test_evaluation_before_loading_data(self):
        with self.assertRaisesRegex(ValueError, "fresh validation-only"):
            run_training(Config(dict(compare_selection_policies=True, test_split="ood_test")))


if __name__ == "__main__":
    unittest.main()
