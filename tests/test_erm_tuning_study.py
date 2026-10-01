"""Small checks for the predeclared validation-only baseline search."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.experiments.erm_tune import GRID, main, rank_conditions
from src.experiments.selection import verify_study_cache


class ErmTuningStudyTests(unittest.TestCase):
    def test_fixed_grid_and_earliest_tie_break(self):
        self.assertEqual(len(GRID),4)
        rows=[dict(index=0,mean_ood_val_auc=.72),
              dict(index=1,mean_ood_val_auc=.72),
              dict(index=2,mean_ood_val_auc=.71)]
        self.assertEqual(rank_conditions(rows)['index'],0)

    def test_dry_run_and_cache_source_check(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);raw=root/'assay.json';raw.write_text('{}')
            cache=root/'cache';cache.mkdir()
            for split in ('train','ood_val'):(cache/f'{split}.pt').touch()
            import hashlib
            (cache/'study_cache.json').write_text(json.dumps(dict(
                raw_path=str(raw.resolve()),raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())))
            verify_study_cache(cache,raw)
            main(['--processed-dir',str(cache),'--raw-path',str(raw),
                  '--device','cpu','--epochs','2','--seeds','5','6','--dry-run',
                  '--output-dir',str(root/'unused')])
            self.assertFalse((root/'unused').exists())
            raw.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'source does not match'):
                verify_study_cache(cache,raw)

    def test_grid_runs_validation_only_and_writes_winner(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);raw=root/'assay.json';raw.write_text('{}')
            cache=root/'cache';cache.mkdir()
            for split in ('train','ood_val'):(cache/f'{split}.pt').touch()
            calls=[]
            def fake_run(config,out,seed,evaluate_test):
                calls.append((config['learning_rate'],seed,evaluate_test))
                return {'best':{'shared':{'auc':.8 if config['dropout']==.1 else .7}}}
            with patch('src.experiments.erm_tune.run_seed',side_effect=fake_run):
                main(['--processed-dir',str(cache),'--raw-path',str(raw),
                      '--device','cpu','--epochs','2','--seeds','5','6',
                      '--output-dir',str(root/'output')])
            report=json.loads((root/'output'/'leaderboard.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['winner']['index'],1)
            self.assertEqual(len(calls),8)
            self.assertTrue(all(not evaluate_test for _,_,evaluate_test in calls))


if __name__=='__main__':unittest.main()
