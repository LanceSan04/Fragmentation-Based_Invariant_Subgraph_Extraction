import copy
import json
from pathlib import Path
import unittest
from collections import Counter
from src.experiments.sugar_reference_search import validate


class ReferenceSearchTests(unittest.TestCase):
    def test_frozen_budget_and_balanced_coverage(self):
        spec=json.loads(Path('configs/sugar_overnight_20261017.json').read_text());validate(spec)
        self.assertEqual(len(spec['trials']),18)
        self.assertEqual(set(row[0] for row in spec['trials']),{.5,1,2,4,8,16,32})
        self.assertEqual(set(Counter(row[1] for row in spec['trials']).values()),{3})
        self.assertEqual(set(Counter(row[2] for row in spec['trials']).values()),{6})
        self.assertEqual(spec['objective'],'all_mean')
        self.assertEqual(spec['contrastive_indexing'],'released')

    def test_scope_expansion_and_out_of_range_values_fail(self):
        original=json.loads(Path('configs/sugar_overnight_20261017.json').read_text())
        for key,value in [('epochs',100),('views',5),('objective','all_max'),('contrastive_indexing','corrected')]:
            bad=copy.deepcopy(original);bad[key]=value
            with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(original);bad['trials'][0][2]=100
        with self.assertRaises(ValueError):validate(bad)
