import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.experiments import cli


class ConsolidatedCLITests(unittest.TestCase):
    def test_main_routes_to_module_and_indexes_output(self):
        import main

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "new_run"

            def fake(args):
                self.assertIn("--dataset", args)
                out.mkdir()
                (out / "summary.json").write_text("{}")

            with patch("src.experiments.selection.main", side_effect=fake):
                main.main(["experiment", "select", "--dataset", "assay", "--output-dir", str(out)])
            index = json.loads((out / "experiment.json").read_text())
            self.assertEqual(index["workflow"], "select")
            self.assertEqual(index["result_files"], ["summary.json"])

    def test_dry_run_routes_under_outputs_without_creating(self):
        calls = []
        with patch(
            "src.experiments.parameter_search.main", side_effect=lambda args: calls.append(args)
        ):
            cli.main(["tune", "--dataset", "size", "--dry-run"])
        out = Path(cli.option(calls[0], "--output-dir"))
        self.assertEqual(out.parent, cli.ROOT / "outputs/size/tune")
        self.assertFalse(out.exists())

    def test_existing_metadata_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "experiment.json").write_text("original")
            with patch("src.experiments.selection.main", side_effect=FileExistsError):
                with self.assertRaises(FileExistsError):
                    cli.main(["select", "--output-dir", str(out)])
            self.assertEqual((out / "experiment.json").read_text(), "original")


if __name__ == "__main__":
    unittest.main()
