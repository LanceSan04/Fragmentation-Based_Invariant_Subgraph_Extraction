import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.experiments import granularity as g


class FrozenEvaluationTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "training"
        source.mkdir()
        plan = dict(datasets=["ec50_size"], seeds=[1], primary_rule="best_raw",
                    inference_policy="independent", test_selects=False, synthetic=True)
        entries = []
        for arm in g.ARMS:
            for rule in g.RULES:
                path = source / (arm + rule + ".pt")
                path.write_bytes(b"checkpoint")
                entries.append(dict(dataset="ec50_size", seed=1, arm=arm, rule=rule,
                    checkpoint=str(path), checkpoint_sha256=g.digest(path), epoch=5,
                    cache=str(root / "cache"), selected_auc=0.7))
        (source / "plan.json").write_text(json.dumps(plan))
        (source / "frozen_checkpoints.json").write_text(json.dumps(entries))
        return source

    def payload(self, path):
        rule = next(r for r in g.RULES if Path(path).stem.endswith(r))
        return dict(epoch=5, config=dict(seed=1, inference_policy="independent",
                                         checkpoint_selection_rule=rule))

    def test_all_validation_precedes_test_without_training_or_source_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = self.fixture(root)
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            calls = []
            def evaluate(checkpoint, split, *args):
                calls.append(split)
                return dict(auc=0.7, loss=0.5, acc=0.6)
            with patch.object(g, "load_checkpoint", side_effect=self.payload), \
                 patch.object(g, "evaluate_saved", side_effect=evaluate), \
                 patch.object(g, "run_seed", side_effect=AssertionError("Must not train")):
                g.evaluate_frozen(source, root / "evaluation", "cpu")
            self.assertEqual(calls, ["ood_val"] * 12 + ["ood_test"] * 12)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_validation_failure_blocks_every_test_call(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = self.fixture(root)
            with patch.object(g, "load_checkpoint", side_effect=self.payload), \
                 patch.object(g, "evaluate_saved", return_value=dict(auc=0.2)) as evaluate:
                with self.assertRaisesRegex(ValueError, "parity failed"):
                    g.evaluate_frozen(source, root / "evaluation", "cpu")
                self.assertEqual(evaluate.call_args.args[1], "ood_val")
                self.assertEqual(evaluate.call_count, 1)

    def test_changed_checkpoint_is_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = self.fixture(root)
            next(source.glob("*.pt")).write_bytes(b"changed")
            with patch.object(g, "load_checkpoint", side_effect=self.payload), \
                 patch.object(g, "evaluate_saved") as evaluate:
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    g.evaluate_frozen(source, root / "evaluation", "cpu")
                evaluate.assert_not_called()
