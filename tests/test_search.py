import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.experiments import search as grid_search
from src.experiments.plans import make_plan


def arguments():
    return SimpleNamespace(
        epochs=None, device=None, dry_run=False, trials=None, repeats=None, runs=None, params=None
    )


class SearchTests(unittest.TestCase):
    def test_full_factorial_budget(self):
        spec, base = make_plan(Path("unused"), "cpu")
        self.assertEqual(
            spec["grid"],
            {
                "lambda_contrast": [0.0, 0.1],
                "lambda_diversity": [0.0, 0.5],
                "lambda_irm": [0.0, 0.5, 1.0, 2.0],
            },
        )
        self.assertEqual(spec["n_trials"] * spec["search_repeats"] + spec["eval_runs"], 35)
        self.assertEqual(base["num_views"], 5)
        self.assertFalse(base["early_stopping"])

    def test_grid_common_seeds_and_completion_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, base = make_plan(Path(directory), "cpu", smoke=True)
            calls = []

            def run(*args, **kwargs):
                calls.append((args[2], args[3], kwargs))
                return {}, {"val_auc": [0.6, 0.7]}

            with patch.object(grid_search, "run_once", side_effect=run):
                grid_search.run_search(spec, base, arguments())
            self.assertEqual([c[1] for c in calls], [101, 102] * 8)
            self.assertTrue(all(c[2]["evaluate_test"] is False for c in calls))

    def test_missing_repeat_cannot_win(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, base = make_plan(Path(directory), "cpu", smoke=True)
            spec.update(grid={"lambda_irm": [0.0]}, n_trials=1)
            with patch.object(
                grid_search,
                "run_once",
                side_effect=[({}, {"val_auc": [0.9]}), ({}, {"val_auc": [None]})],
            ):
                grid_search.run_search(spec, base, arguments())
            self.assertFalse((Path(directory) / "search/best_params.json").exists())

    def test_eval_enables_test_and_rejects_missing_auc(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, base = make_plan(Path(directory), "cpu", smoke=True)
            path = Path(directory) / "best.json"
            path.write_text(json.dumps({"params": {"lambda_irm": 0}}))
            args = arguments()
            args.params = str(path)
            args.runs = 1
            with patch.object(
                grid_search, "run_once", return_value=({"ood_test": {"auc": 0.7}}, {})
            ) as run:
                grid_search.run_eval(spec, base, args)
            self.assertEqual(run.call_args.args[1]["test_split"], "ood_test")
            self.assertFalse(run.call_args.args[1]["log_test_curve"])
            self.assertTrue(run.call_args.kwargs["retain_artifacts"])
            with patch.object(grid_search, "run_once", return_value=({}, {})):
                with self.assertRaisesRegex(RuntimeError, "Missing/nonfinite"):
                    grid_search.run_eval(spec, base, args)

    def test_nonfinite_summary(self):
        self.assertEqual(grid_search._summarize([None, float("nan"), float("inf"), 0.7])["n"], 1)


if __name__ == "__main__":
    unittest.main()
