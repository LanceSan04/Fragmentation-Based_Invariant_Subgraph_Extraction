import argparse
import importlib.util
import json
from pathlib import Path
import unittest

from src.experiments.research_batch import commands


class ResearchBatchTests(unittest.TestCase):
    def test_frozen_sequential_scope(self):
        args=argparse.Namespace(output_dir='fixture',reference_env='fixture_env',
            smoke=False,raw_root='data/raw',device='cuda',cache_root='fixture_cache',
            selection_source='fixture_status.json')
        steps=commands(args)
        self.assertEqual([s['name'] for s in steps],
                         ['reference_environment','upstream_pilot','ki','good'])
        self.assertEqual(steps[1]['depends_on'],'reference_environment')
        self.assertIn('--evaluate-primary-test',steps[2]['command'])
        self.assertIn('good_hiv_size',steps[3]['command'])
        self.assertNotIn('--smoke',steps[1]['command'])

    def test_release_contains_every_pinned_reference_file(self):
        root=Path(__file__).resolve().parents[1]
        release_path = root / "tools" / "remote" / "reu_release.py"
        if not release_path.is_file():
            self.skipTest("Local release builder is unavailable")
        spec = importlib.util.spec_from_file_location("release", release_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        bundled=module.files(root)
        for directory in ('upstream','drugood_reference'):
            provenance=root/'baselines/sugar'/('provenance.json' if directory=='upstream' else 'drugood_reference/provenance.json')
            for name in json.loads(provenance.read_text())['files']:
                self.assertIn('baselines/sugar/'+directory+'/'+name,bundled)
