import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.experiments.checkpoints import digest
from src.experiments.original_tuning import validate,winners,DATASETS,dataset_scope,require_complete
from src.experiments.plans import make_plan
from src.experiments.selection import run_seed,validation_execution
from src.data.preprocess import run_preprocessing
from src.training.pipeline import Config
from src.training.checkpoint import load_checkpoint


class OriginalTuningTests(unittest.TestCase):
    def test_family_scope_limits_search_and_winner_selection(self):
        datasets=dataset_scope('ec50')
        self.assertEqual(datasets,['ec50_assay','ec50_scaffold','ec50_size'])
        self.assertEqual(len(datasets)*6*2,36)
        self.assertEqual(dataset_scope('all'),DATASETS)
        self.assertTrue(all(d.startswith('ic50_') for d in dataset_scope('ic50')))
        report=dict(plan=dict(datasets=datasets),runs=[dict(dataset=d,recipe=1,validation={'auc':.7}) for d in datasets])
        self.assertEqual(list(winners(report)),datasets)

    def test_confirmation_rejects_different_family_search(self):
        report=dict(plan=dict(stage='search',config_sha256='x',synthetic=False,
            recipe_ids=list(range(1,7)),seeds=[11311,11312],epochs=150,datasets=dataset_scope('ec50')))
        with self.assertRaisesRegex(ValueError,'dataset scope'):
            require_complete(report,'search','x',False,Path('.'),dataset_scope('ic50'))

    def test_scope_and_sampler_limit(self):
        spec=json.loads((Path(__file__).resolve().parents[1]/'configs/official_original_tuning_20261030.json').read_text())
        validate(spec)
        for field,value in [('fragment_feature_mode','molecular_gnn'),('classifier_feature_mode','encoder')]:
            changed=copy.deepcopy(spec);changed['base'][field]=value
            with self.assertRaises(ValueError):validate(changed)
        self.assertTrue(all(recipe[2]<1 for recipe in spec['recipes']))

    def test_recipe_choice_uses_mean_over_seeds_and_stable_ties(self):
        rows=[]
        for dataset in DATASETS:
            for rid,values in [(1,[.9,.6]),(2,[.8,.8])]:
                rows.extend(dict(dataset=dataset,recipe=rid,seed=s,validation={'auc':v})
                            for s,v in zip([11311,11312],values))
        chosen=winners({'runs':rows})
        self.assertTrue(all(v['recipe']==2 for v in chosen.values()))
        for row in rows:row['validation']['auc']=.8
        self.assertTrue(all(v['recipe']==1 for v in winners({'runs':rows}).values()))

    def test_cpu_validation_scope_restores_training_flags(self):
        before=torch.are_deterministic_algorithms_enabled()
        try:
            with validation_execution('cpu'):
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                raise RuntimeError('scope exit')
        except RuntimeError:pass
        self.assertEqual(torch.are_deterministic_algorithms_enabled(),before)

    def test_cpu_clones_preserve_training_and_capture_selected_forward(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,base=make_plan(root/'fixture','cpu',True)
            base.update(epochs=2,num_workers=0,report_within_assay=False,
                compare_checkpoint_rules=False,fragment_feature_mode='static_pool',
                classifier_feature_mode='raw',early_stopping=False)
            run_preprocessing(Config(base))
            legacy=run_seed(base,root/'legacy',77,False,('independent',))
            paired=run_seed(dict(base,validation_device='cpu',save_selected_validation_predictions=True),
                            root/'cpu',77,False,('independent',))
            a=load_checkpoint(root/'legacy/best_independent.pt');b=load_checkpoint(root/'cpu/best_independent.pt')
            self.assertEqual(legacy['best'],paired['best'])
            for i,states in a['view_state_dicts'].items():
                for key,value in states.items():
                    torch.testing.assert_close(value,b['view_state_dicts'][i][key],rtol=0,atol=0)
            reference=json.loads((root/'cpu/best_independent_validation_reference.json').read_text())
            self.assertEqual(reference['checkpoint_sha256'],digest(root/'cpu/best_independent.pt'))
            self.assertEqual(reference['epoch'],paired['best']['independent']['epoch'])


if __name__=='__main__':unittest.main()
