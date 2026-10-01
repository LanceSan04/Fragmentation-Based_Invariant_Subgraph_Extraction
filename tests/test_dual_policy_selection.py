import contextlib, io, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import torch
from src.experiments.selection import run_seed
from src.experiments.plans import make_plan
from src.data.preprocess import run_preprocessing
from src.training.pipeline import Config


class DualSelectionTests(unittest.TestCase):
    def test_separate_selection_and_no_test_loader(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d)
            _, base = make_plan(root, "cpu", True)
            base.update(epochs=3, lambda_diversity=0.0, lambda_irm=0.0, lambda_contrast=0.1)
            run_preprocessing(Config(base))
            # Shared best at epoch2, independent best at epoch3.
            aucs = [0.6, 0.5, 0.8, 0.6, 0.7, 0.9, 0.8, 0.6, 0.7, 0.9]
            with patch(
                "src.experiments.selection._evaluate_ensemble",
                side_effect=[(0.5, 0.5, a) for a in aucs],
            ) as evaluate:
                result = run_seed(base, root / "run", 5001, False)
            self.assertEqual(result["best"]["shared"]["epoch"], 2)
            self.assertEqual(result["best"]["independent"]["epoch"], 3)
            self.assertEqual(evaluate.call_count, 10)
            for policy, epoch in [("shared", 2), ("independent", 3)]:
                payload = torch.load(root / "run" / f"best_{policy}.pt", weights_only=False)
                self.assertEqual(payload["epoch"], epoch)
                self.assertEqual(payload["config"]["inference_policy"], policy)
                self.assertEqual(payload["config"]["loader_split_names"], ["train", "ood_val"])
                self.assertNotIn("ood_test", result["results"][policy][policy])
            self.assertEqual(len(json.loads((root / "run/history.json").read_text())), 3)


if __name__ == "__main__":
    unittest.main()
