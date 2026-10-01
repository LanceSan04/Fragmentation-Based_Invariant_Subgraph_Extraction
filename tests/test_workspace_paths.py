import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from src.data import paths


class WorkspacePathTests(unittest.TestCase):
    def test_old_cache_redirect_only_when_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            new = root / paths.NEW_ASSAY_CACHE
            new.mkdir(parents=True)
            with patch.object(paths, "ROOT", root):
                with self.assertWarns(UserWarning):
                    self.assertEqual(
                        paths.resolve_processed_path("/old/server/" + paths.OLD_ASSAY_CACHE),
                        str(new),
                    )
                self.assertEqual(paths.resolve_processed_path("/other/cache"), "/other/cache")
                original = root / paths.OLD_ASSAY_CACHE
                original.mkdir(parents=True)
                self.assertEqual(paths.resolve_processed_path(str(original)), str(original))


if __name__ == "__main__":
    unittest.main()
