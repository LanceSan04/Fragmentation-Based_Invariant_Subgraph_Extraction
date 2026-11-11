import copy
import json
from pathlib import Path
import unittest

from src.experiments.ec50_seed_check import validate,scopes,top_five,DATASETS


class SeedCheckTests(unittest.TestCase):
    def spec(self):
        return json.loads((Path(__file__).resolve().parents[1]/'configs/ec50_seed_check_20261102.json').read_text())

    def test_two_groups_have_exact_requested_budget(self):
        s=self.spec();validate(s);groups=scopes(s,False)
        self.assertEqual(sum(g['training_runs'] for g in groups),30)
        self.assertEqual(groups[0]['seeds'],[1,2,3,4,5])
        self.assertEqual(groups[0]['datasets'],DATASETS)
        for g in groups:
            self.assertEqual(g['feature_mode'],'static_pool')
            self.assertEqual(g['classifier_feature_mode'],'raw')
            self.assertEqual(g['epochs'],150)
        self.assertEqual(groups[2]['seeds'],[11413,11403,11409,11414,11401])
        self.assertEqual(sum(g['training_runs'] for g in scopes(s,True)),6)

    def test_top_five_use_each_datasets_own_test_results(self):
        rows=[dict(dataset=d,seed=i,metrics={'auc':.5+.01*i}) for d in DATASETS for i in range(6)]
        chosen=top_five({'test_evaluation':{'runs':rows}})
        self.assertTrue(all([r['seed'] for r in chosen[d]]==[5,4,3,2,1] for d in DATASETS))

    def test_selected_only_excludes_requested_initialization_cohort(self):
        s=self.spec();groups=scopes(s,False,True)
        self.assertEqual(sum(g['training_runs'] for g in groups),15)
        self.assertEqual(len(groups),3)
        self.assertTrue(all(g['study'].startswith('selected_') for g in groups))
        self.assertTrue(all(seed>5 for g in groups for seed in g['seeds']))
        self.assertEqual(sum(g['training_runs'] for g in scopes(s,True,True)),3)

    def test_budget_and_reproducibility_labels_cannot_be_changed(self):
        for key,value in [('epochs',100),('training_runs',15),('selected_reruns_are_reproducibility_only',False)]:
            s=copy.deepcopy(self.spec());s[key]=value
            with self.assertRaises(ValueError):validate(s)


if __name__=='__main__':unittest.main()
