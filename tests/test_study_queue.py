import copy
from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch
import yaml
from src.experiments.study_queue import validate, choose_diversity, choose_contrast, queue_lock


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.spec = yaml.safe_load(Path('configs/mentor_study_queue.yaml').read_text())

    def test_budget_and_seed_partition(self):
        self.assertEqual(sum(validate(self.spec).values()), 198)
        self.spec['max_training_runs'] = 197
        with self.assertRaisesRegex(ValueError, 'budget'):
            validate(self.spec)
        self.spec['max_training_runs'] = 198
        self.spec['tune_seeds'] = self.spec['screen_seeds'][:2]
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            validate(self.spec)

    def test_diversity_requires_paired_gain_and_complete_batch(self):
        records = {(f,d,s): 0.8 for f in self.spec['feature_modes']
                   for d in self.spec['diversity'] for s in self.spec['screen_seeds']}
        self.assertEqual(set(choose_diversity(self.spec, records).values()), {'off'})
        for seed in self.spec['screen_seeds']:
            records['internal_gnn','tiny',seed] += 0.01
        self.assertEqual(choose_diversity(self.spec,records)['internal_gnn'],'tiny')
        del records['internal_gnn','ramp',self.spec['screen_seeds'][0]]
        with self.assertRaises(KeyError):
            choose_diversity(self.spec,records)

    def test_contrast_prefers_lower_near_tie(self):
        records = {(f,c,s): 0.8 for f in self.spec['feature_modes']
                   for c in self.spec['contrast_coefficients'] for s in self.spec['tune_seeds']}
        for seed in self.spec['tune_seeds']:
            records['internal_gnn', 32.0, seed] = 0.801
        self.assertEqual(set(choose_contrast(self.spec, records).values()), {0.1})

    def test_exclusive_queue_lock(self):
        with tempfile.TemporaryDirectory() as d:
            with queue_lock(Path(d)/'lock'):
                with self.assertRaises(OSError):
                    with queue_lock(Path(d)/'lock'):
                        self.fail('second lock acquired')

    def test_fresh_launcher_can_point_to_existing_queue(self):
        from src.experiments.study_queue import main
        with tempfile.TemporaryDirectory() as d:
            previous=Path(d)/'previous'; previous.mkdir()
            (previous/'status.json').write_text('{}')
            fresh=Path(d)/'fresh'
            with patch('src.experiments.study_queue.run_queue') as run:
                main(['--resume','--resume-from',str(previous),'--output-dir',str(fresh)])
                self.assertEqual(run.call_args.args[1],str(previous.resolve()))
                self.assertEqual(json.loads((fresh/'resume_pointer.json').read_text())['queue'],str(previous.resolve()))

    def test_resume_rejects_source_or_manifest_change_before_any_training(self):
        from src.experiments.study_queue import run_queue, write
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'queue';out.mkdir()
            write(out/'status.json',{'identity':{'changed':True}})
            with self.assertRaisesRegex(ValueError,'Source, manifest'):
                run_queue(self.spec,out,'data/raw',Path(d)/'cache','cpu',resume=True)


if __name__ == '__main__':
    unittest.main()
