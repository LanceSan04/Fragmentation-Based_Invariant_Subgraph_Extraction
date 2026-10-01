import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from src.experiments.inference import compare, checkpoint_selection_metadata
from src.experiments.plans import make_plan
from src.data.preprocess import run_preprocessing
from src.training.pipeline import Config, run_training


class FollowupTests(unittest.TestCase):
    def test_selection_metadata_uses_saved_policy(self):
        for policy in ("shared", "independent"):
            with self.subTest(policy=policy):
                result = checkpoint_selection_metadata({"config": {"inference_policy": policy}})
                self.assertEqual(result["checkpoint_selection_policy"], policy)
                self.assertEqual(
                    result["checkpoint_selection"], f"original {policy}-graph validation selection"
                )
                self.assertEqual(
                    result["checkpoint_selection_source"], "checkpoint.config.inference_policy"
                )

    def test_selection_metadata_does_not_guess_for_legacy_or_invalid_policy(self):
        for payload in ({}, {"config": {}}, {"config": {"inference_policy": "invalid"}}):
            with self.subTest(payload=payload):
                result = checkpoint_selection_metadata(payload)
                self.assertEqual(result["checkpoint_selection_policy"], "unknown")
                self.assertIsNone(result["checkpoint_selection_source"])

    def test_identical_views_vote_equivalence_and_read_only(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            _, base = make_plan(root, "cpu", True)
            base.update(
                epochs=1,
                lambda_irm=0.0,
                lambda_diversity=0.0,
                num_views=2,
                view_fragmentations={"group_graph": 2},
                save_model=str(root / "model.pt"),
                inference_policy="independent",
            )
            run_preprocessing(Config(base))
            run_training(Config(base))
            payload = torch.load(root / "model.pt", weights_only=False)
            payload["view_state_dicts"][1] = payload["view_state_dicts"][0]
            torch.save(payload, root / "model.pt")
            before = hashlib.sha256((root / "model.pt").read_bytes()).hexdigest()
            report = compare(
                root / "model.pt", root / "comparison", processed_dir=root / "processed"
            )
            self.assertEqual(before, hashlib.sha256((root / "model.pt").read_bytes()).hexdigest())
            self.assertEqual(report["checkpoint_selection_policy"], "independent")
            for split in ["ood_val", "ood_test"]:
                self.assertEqual(report["splits"][split]["n"], 4)
                with np.load(root / "comparison" / f"{split}_predictions.npz") as arrays:
                    np.testing.assert_allclose(arrays["shared"], arrays["independent"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
