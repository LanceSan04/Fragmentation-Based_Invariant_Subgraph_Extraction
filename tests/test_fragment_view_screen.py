import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from src.experiments.checkpoints import digest
from src.experiments.fragment_view_screen import validate


class ViewScreenTests(unittest.TestCase):
    def test_budget_and_frozen_fixed_recipe(self):
        spec=json.loads(Path('configs/fragment_view_diversity_screen_20261018.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'reference.json';p.write_text('{}');spec['reference_plan_sha256']=digest(p)
            validate(spec,p)
            self.assertEqual(len(spec['datasets'])*len(spec['seeds'])*len(spec['views'])*len(spec['diversity']),24)
            bad=copy.deepcopy(spec);bad['fixed']['inference_policy']='shared'
            with self.assertRaises(ValueError):validate(bad,p)

    def test_custom_selection_loop_applies_the_ramp(self):
        from src.experiments.plans import make_plan
        from src.data.preprocess import run_preprocessing
        from src.training.pipeline import Config
        from src.experiments.selection import run_seed
        threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as d:
                _,base=make_plan(Path(d),'cpu',True)
                base.update(epochs=5,num_views=2,view_fragmentations={'group_graph':2},
                    lambda_diversity=.005,diversity_delay_epochs=1,diversity_ramp_epochs=2)
                base['splits'].pop('ood_test',None);base['split_names']=['train','ood_val']
                run_preprocessing(Config(base));run_seed(base,Path(d)/'run',19,False,('independent',))
                history=json.loads((Path(d)/'run/history.json').read_text())
                self.assertEqual([r['effective_diversity_weight'] for r in history],[0.,.0025,.005,.005,.005])
        finally:torch.set_num_threads(threads)
