import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from src.data import paths
from scripts.migrate_workspace_paths import plan

class WorkspacePathTests(unittest.TestCase):
    def test_old_cache_redirect_only_when_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);new=root/paths.NEW_ASSAY_CACHE;new.mkdir(parents=True)
            with patch.object(paths,'ROOT',root):
                with self.assertWarns(UserWarning):self.assertEqual(paths.resolve_processed_path('/old/server/'+paths.OLD_ASSAY_CACHE),str(new))
                self.assertEqual(paths.resolve_processed_path('/other/cache'),'/other/cache')
                original=root/paths.OLD_ASSAY_CACHE;original.mkdir(parents=True)
                self.assertEqual(paths.resolve_processed_path(str(original)),str(original))
    def test_migration_plan_cache_first_and_collision_safe(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/paths.OLD_ASSAY_CACHE;cache.mkdir(parents=True)
            pairs=plan(root);self.assertEqual(pairs[0],(cache,root/paths.NEW_ASSAY_CACHE))
            (root/paths.NEW_ASSAY_CACHE).mkdir(parents=True)
            with self.assertRaises(FileExistsError):plan(root)
            self.assertTrue(cache.exists())
    def test_migration_does_not_touch_canonical_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'outputs/assay').mkdir(parents=True)
            self.assertEqual(plan(root),[])

if __name__=='__main__':unittest.main()
