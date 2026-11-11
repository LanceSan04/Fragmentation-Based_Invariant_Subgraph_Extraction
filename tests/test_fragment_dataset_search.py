from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.experiments.checkpoints import digest
from src.experiments.fragment_dataset_search import validate,diagnostic_summary,apply_screen_settings


class FragmentDatasetSearchTests(unittest.TestCase):
    def test_screened_search_budget_settings_and_expanded_coverage(self):
        spec=json.loads(Path('configs/fragment_scaffold_size_search_screened_20261018.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'reference.json';p.write_text('{}');spec['reference_plan_sha256']=digest(p)
            validate(spec,p)
            self.assertEqual(len(spec['recipes'])*len(spec['seeds'])*len(spec['datasets']),48)
            self.assertEqual({r[1] for r in spec['recipes']},{.3,.5,.7,.9})
            self.assertEqual({r[2] for r in spec['recipes']},{.5,.7,.9})
            for col in range(4):self.assertEqual(len(set(Counter(r[col] for r in spec['recipes']).values())),1)
            base=dict(num_views=5,lambda_diversity=0,lambda_irm=2,processed_dir='old')
            apply_screen_settings(base,spec,'ec50_size',d)
            self.assertEqual(base['num_views'],3)
            self.assertEqual(base['view_fragmentations'],{'group_graph':3})
            self.assertEqual(base['lambda_diversity'],.005)
            self.assertEqual(base['lambda_irm'],0)
            self.assertEqual(Path(base['processed_dir']),Path(d).resolve()/'ec50_size/views3')
            with self.assertRaises(ValueError):apply_screen_settings({},spec,'ec50_size',None)
            bad=copy.deepcopy(spec);bad['fixed']['lambda_diversity']=0
            with self.assertRaises(ValueError):validate(bad,p)

    def test_budget_and_recipe_coverage(self):
        spec=json.loads(Path('configs/fragment_scaffold_size_search_20261018.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'reference.json';p.write_text('{}');spec['reference_plan_sha256']=digest(p)
            validate(spec,p)
            self.assertEqual(len(spec['recipes'])*len(spec['seeds'])*len(spec['datasets']),48)
            for column in [0,1,3]:self.assertEqual(set(Counter(r[column] for r in spec['recipes']).values()),{4})
            bad=copy.deepcopy(spec);bad['inference_policy']='shared'
            with self.assertRaises(ValueError):validate(bad,p)
            bad=copy.deepcopy(spec);bad['recipes'][0][2]=1.
            with self.assertRaises(ValueError):validate(bad,p)

    def test_missing_chemical_groups_are_not_reported_as_zero_cut(self):
        report=dict(summary={'molecule_views':1},records=[dict(index=0,original_atoms=4,original_bonds=3,
            selected=dict(atoms=2,bonds=1,components=1,chemical_groups={},atom_ids=[0,1],bond_atom_pairs=[[0,1]]))])
        result=diagnostic_summary(report)
        self.assertIsNone(result['groups']['ring']['mean_cut_fraction'])
        self.assertEqual(result['mean_atom_fraction'],.5)
