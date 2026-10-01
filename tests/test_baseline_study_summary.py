"""Paired summary checks: seed alignment and frozen primary policies."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.summarize_baseline_study import summarize


class BaselineStudySummaryTests(unittest.TestCase):
    def make_arm(self,root,name,seeds):
        arm=root/name;arm.mkdir()
        runs=[]
        policy='shared' if name=='erm' else 'independent'
        for seed in seeds:
            (arm/f'seed{seed}').mkdir()
            (arm/f'seed{seed}'/'config.json').write_text(json.dumps(dict(
                seed=seed,edge_ratio=.5,selection_mode='random' if name=='random' else 'learned')))
            runs.append(dict(seed=seed,results={policy:{policy:{
                'ood_val':{'auc':.70},'ood_test':{'auc':.75 if name=='learned' else .70}}}}))
        (arm/'summary.json').write_text(json.dumps(dict(runs=runs)))
        return arm

    def test_paired_difference_and_seed_guard(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            learned=self.make_arm(root,'learned',[1,2])
            random=self.make_arm(root,'random',[1,2])
            erm=self.make_arm(root,'erm',[1,2])
            report=summarize(learned,random,erm)
            self.assertAlmostEqual(report['paired_differences']['learned_minus_random']['mean'],.05)
            self.assertEqual(report['primary_policy']['learned'],'independent/independent')
            self.make_arm(root,'mismatch',[1,3])
            with self.assertRaisesRegex(ValueError,'same nonempty seed set'):
                summarize(learned,root/'mismatch',erm)


if __name__=='__main__':unittest.main()
