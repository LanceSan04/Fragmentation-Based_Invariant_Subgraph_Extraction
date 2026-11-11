import copy
import json
from pathlib import Path
import unittest
import numpy as np

from src.experiments.ec50_replication import validate,study_scope,seed_summary,compare_reports,DATASETS


class EC50ReplicationTests(unittest.TestCase):
    def spec(self):
        return json.loads((Path(__file__).resolve().parents[1]/'configs/ec50_replication_20261031.json').read_text())

    def test_fixed_budget_and_paired_seed_prefix(self):
        spec=self.spec();validate(spec)
        a=study_scope(spec,'static',False);b=study_scope(spec,'shared-gnn',False)
        self.assertEqual((a['training_runs'],b['training_runs']),(90,30))
        self.assertEqual(a['seeds'][:10],b['seeds'])
        self.assertEqual(a['classifier_feature_mode'],'raw')
        self.assertEqual(b['classifier_feature_mode'],'encoder')
        self.assertEqual(a['epochs'],150)
        for key,value in [('epochs',100),('encoder',dict(spec['encoder'],classifier_feature_mode='raw'))]:
            bad=copy.deepcopy(spec);bad[key]=value
            with self.assertRaises(ValueError):validate(bad)

    def test_summary_includes_all_seeds_and_uncertainty(self):
        s=seed_summary([.6,.7,.8])
        self.assertEqual(s['n'],3);self.assertAlmostEqual(s['mean'],.7)
        self.assertAlmostEqual(s['sample_sd'],.1)
        self.assertLess(s['mean_seed_ci95'][0],s['mean'])
        self.assertGreater(s['mean_seed_ci95'][1],s['mean'])
        self.assertIsNone(seed_summary([.7])['mean_seed_ci95'])

    def test_paired_comparison_uses_seed_identity_not_row_order(self):
        def report(name,seeds):
            return dict(plan=dict(study=name,classifier_feature_mode='raw' if name=='static' else 'encoder',
                feature_mode='static_pool' if name=='static' else 'molecular_gnn'),test_evaluation=dict(runs=[
                dict(dataset=d,seed=s,recipe=1,metrics={'auc':.6+s*.001}) for d in DATASETS for s in seeds]))
        a=report('static',[1,2,3]);b=report('shared-gnn',[2,1])
        for row in b['test_evaluation']['runs']:row['metrics']['auc']+=.02
        for row in compare_reports(a,b).values():
            self.assertEqual(row['n'],2);self.assertAlmostEqual(row['mean'],.02)
        for row in b['test_evaluation']['runs']:row['metrics']['auc']=np.float64(row['metrics']['auc'])
        json.dumps(compare_reports(a,b),allow_nan=False)
        b['test_evaluation']['runs'][0]['seed']=999
        with self.assertRaisesRegex(ValueError,'Missing paired'):compare_reports(a,b)

    def test_mislabeled_scorer_only_report_cannot_be_feature_sharing(self):
        a=dict(plan=dict(study='static',classifier_feature_mode='raw'),test_evaluation={'runs':[]})
        b=dict(plan=dict(study='shared-gnn',classifier_feature_mode='raw',feature_mode='molecular_gnn'))
        with self.assertRaisesRegex(ValueError,'feature routing'):compare_reports(a,b)


if __name__=='__main__':unittest.main()
