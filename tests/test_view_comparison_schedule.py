import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import torch
from src.experiments.views import compare, correlation
from src.experiments.plans import make_plan
from src.training.diversity_schedule import diversity_weight
from src.data.preprocess import run_preprocessing
from src.training.pipeline import Config, run_training


class ViewComparisonScheduleTests(unittest.TestCase):
    def test_schedule_boundaries(self):
        self.assertEqual(
            [diversity_weight(0.2, e, 2, 2) for e in range(1, 6)], [0, 0, 0.1, 0.2, 0.2]
        )
        self.assertEqual(diversity_weight(0.2, 1), 0.2)
        self.assertEqual(diversity_weight(0.2, 3, 2, 0), 0.2)
        with self.assertRaises(ValueError):
            diversity_weight(0.2, 0)
        self.assertIsNone(correlation([1, 1], [2, 3]))

    def test_real_schedule_and_identical_view_control(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            _, values = make_plan(root, "cpu", True)
            values.update(
                epochs=3,
                lambda_diversity=0.1,
                diversity_delay_epochs=1,
                diversity_ramp_epochs=2,
                lambda_contrast=0.1,
                lambda_irm=0.0,
                num_views=2,
                view_fragmentations={"group_graph": 2},
                save_model=str(root / "checkpoint.pt"),
                loss_log_path=str(root / "loss.csv"),
                early_stopping=True,
                patience=1,
                early_stopping_min_epochs=3,
            )
            cfg = Config(values)
            run_preprocessing(cfg)
            result = run_training(cfg)
            h = result["history"]
            self.assertEqual(h["effective_diversity_weight"], [0.0, 0.05, 0.1])
            for raw, weighted, weight in zip(
                h["train_diversity_loss"],
                h["train_diversity_loss_weighted"],
                h["effective_diversity_weight"],
            ):
                self.assertAlmostEqual(raw * weight, weighted, places=5)
            payload = torch.load(root / "checkpoint.pt", weights_only=False)
            payload["view_state_dicts"][1] = payload["view_state_dicts"][0]
            torch.save(payload, root / "identical.pt")
            report = compare(root / "identical.pt", root / "processed")
            self.assertEqual(report["pairs"][0]["edge_jaccard_mean"], 1.0)
            self.assertEqual(report["pairs"][0]["own"]["disagreement"], 0.0)
            self.assertAlmostEqual(
                report["metrics"]["full_ensemble"]["auc"], report["metrics"]["own_view_0"]["auc"]
            )
            self.assertEqual(report["split"], "ood_val")


if __name__ == "__main__":
    unittest.main()
