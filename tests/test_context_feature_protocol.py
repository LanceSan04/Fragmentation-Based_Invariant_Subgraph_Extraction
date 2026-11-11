"""Plan-only checks for optional IC50 support and preserved EC50 scope."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
import hashlib
from unittest.mock import patch
from pathlib import Path

from src.experiments.context_features import main, validate
from src.experiments.cross_domain import DATASETS, dataset_config

ROOT = Path(__file__).resolve().parents[1]


class ContextProtocolTests(unittest.TestCase):
    def test_frozen_protocols_and_preparation_plans_do_not_execute(self):
        for filename, runs in [('context_features_pilot_20261028.json',6),
                               ('context_features_full_20261028.json',30),
                               ('ic50_context_features_pilot_20261029.json',6),
                               ('ic50_context_features_full_20261029.json',30),
                               ('official_ec50_context_pilot_20261029.json',6),
                               ('official_ec50_context_full_20261029.json',30)]:
            config = ROOT/'configs'/filename
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)/'must_not_exist'
                with contextlib.redirect_stdout(io.StringIO()):
                    plan = main(['--config',str(config),'--project-root',str(ROOT),
                                 '--output-dir',str(out)])
                    preparation = main(['--config',str(config),'--project-root',str(ROOT),
                                        '--output-dir',str(out),'--stage','prepare'])
                self.assertEqual(plan['training_runs'],runs)
                self.assertFalse(plan['test_evaluated'])
                self.assertEqual(preparation['training_runs'],0)
                self.assertFalse(out.exists())

    def test_mixed_dataset_families_and_changed_budget_rejected(self):
        spec = json.loads((ROOT/'configs/ic50_context_features_pilot_20261029.json').read_text())
        for field, value in [('epochs',6),('seeds',[11001])]:
            changed = copy.deepcopy(spec); changed[field] = value
            with self.assertRaises(ValueError): validate(changed)
        changed = copy.deepcopy(spec); changed['datasets'][1]['dataset']='ec50_scaffold'
        with self.assertRaises(ValueError): validate(changed)

    def test_ic50_support_does_not_expand_historical_defaults(self):
        self.assertNotIn('ic50_assay',DATASETS)
        for split in ('assay','scaffold','size'):
            cfg = dataset_config('ic50_'+split,ROOT/'data/raw',ROOT/'data/processed','cpu')
            self.assertEqual(cfg['raw_format'],'drugood_json')
            self.assertEqual(cfg['dataset'],'drugood_lbap_core_ic50_'+split)
            self.assertTrue(cfg['raw_path'].endswith('lbap_core_ic50_'+split+'.json'))

    def test_explicit_cache_stage_never_trains(self):
        spec = json.loads((ROOT/'configs/ic50_context_features_pilot_20261029.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            for row in spec['datasets']:
                raw = project/row['raw_path']; raw.parent.mkdir(parents=True,exist_ok=True)
                raw.write_text('{}'); row['raw_sha256']=hashlib.sha256(raw.read_bytes()).hexdigest()
            config = project/'spec.json'; config.write_text(json.dumps(spec))
            with patch('src.experiments.context_features.prepare_cache',return_value={'verified':True}) as prepare, \
                 patch('src.experiments.context_features.run_seed') as train:
                result = main(['--config',str(config),'--project-root',str(project),
                               '--output-dir',str(project/'prepared'),'--stage','prepare','--execute'])
            self.assertEqual(prepare.call_count,3)
            train.assert_not_called()
            self.assertEqual(result['training_runs'],0)
            self.assertFalse(result['test_evaluated'])


if __name__ == '__main__':
    unittest.main()
