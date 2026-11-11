import copy
import json
from pathlib import Path
import unittest
from src.experiments.sugar_reference_replicate import validate,summarize


class ReferenceReplicationTests(unittest.TestCase):
    def test_recipe_and_fresh_initializations_are_frozen(self):
        spec=json.loads(Path('configs/sugar_winner_replication_20261017.json').read_text())
        import hashlib
        selection=json.dumps(dict(training_runs=18,test_evaluated=False,objective='all_mean',
            best=dict(contrast=32,diversity_penalty=.01,sample_rate=90))).encode()
        spec['selection_sha256']=hashlib.sha256(selection).hexdigest();validate(spec,selection)
        for key,value in [('initializations',[1,2,3,4,5]),('contrast',16),('views',5)]:
            bad=copy.deepcopy(spec);bad[key]=value
            with self.assertRaises(ValueError):validate(bad,selection)

    def test_summary_includes_all_runs_and_sample_standard_deviation(self):
        rows=[dict(initialization=i,status='completed',validation_auc=v,test_auc=v)
              for i,v in [(2,.6),(3,.8)]]
        report=summarize(rows)
        self.assertEqual(report['test_auc']['n'],2)
        self.assertAlmostEqual(report['test_auc']['mean'],.7)
        self.assertAlmostEqual(report['test_auc']['std'],.1414213562373095)
        rows[1]['status']='failed'
        with self.assertRaises(ValueError):summarize(rows)
