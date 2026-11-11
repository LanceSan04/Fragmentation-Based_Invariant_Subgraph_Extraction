import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml
from src.experiments.mentor_confirmation import evaluate_primary, verify_selection


class MentorConfirmationTests(unittest.TestCase):
    def test_two_dataset_plan_is_40_runs_without_retuning(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from src.experiments.mentor_confirmation import main
        with tempfile.TemporaryDirectory() as d,patch('src.experiments.mentor_confirmation.verify_selection'),redirect_stdout(StringIO()):
            plan=main(['--datasets','ec50_scaffold','ec50_size','--cache-root',str(Path(d)/'cache'),
                       '--output-dir',str(Path(d)/'results'),'--selection-source','fixture','--dry-run'])
        self.assertEqual(plan['training_runs'],40)
        self.assertEqual(plan['seeds'],list(range(9521,9526)))
        self.assertEqual(set(plan['configs']),{'ec50_scaffold','ec50_size'})
        self.assertEqual(plan['arms']['internal_gnn']['lambda_diversity'],.005)
        self.assertEqual(plan['arms']['static_pool']['lambda_contrast'],2.)

    def test_frozen_recipes_and_fresh_seeds(self):
        spec=yaml.safe_load(Path('configs/mentor_assay_confirmation.yaml').read_text())
        self.assertEqual(spec['seeds'],list(range(9521,9526)))
        self.assertEqual(spec['epochs'],100)
        self.assertEqual(spec['arms']['static_pool']['lambda_contrast'],2.)
        self.assertEqual(spec['arms']['internal_gnn']['lambda_contrast'],1.)
        self.assertEqual(spec['arms']['internal_gnn']['lambda_diversity'],.005)
        self.assertEqual(spec['arms']['random_learned_gates']['ranking_mode'],'random')
        self.assertTrue(spec['arms']['erm']['erm_baseline'])

    def test_selection_provenance_cannot_silently_change(self):
        spec=yaml.safe_load(Path('configs/mentor_assay_confirmation.yaml').read_text())
        status=dict(status='tune_completed',decisions=spec['selection_decisions'],
                    jobs={str(i):dict(status='completed') for i in range(58)})
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'status.json';path.write_text(json.dumps(status))
            spec['selection_source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            verify_selection(spec,path)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'hash changed'):
                verify_selection(spec,path)

    def test_every_validation_passes_before_any_test(self):
        from src.experiments.checkpoints import digest
        with tempfile.TemporaryDirectory() as d:
            entries=[]
            for i,arm in enumerate(('static_pool','internal_gnn')):
                path=Path(d)/f'{arm}.pt';path.write_bytes(b'fake checkpoint fixture')
                entries.append(dict(arm=arm,seed=9521,epoch=8,cache=d,checkpoint=str(path),
                                    checkpoint_sha256=digest(path),selected_auc=.8))
            payload=dict(epoch=8,config=dict(seed=9521,inference_policy='independent',checkpoint_selection_rule='best_raw'))
            calls=[]
            def fake(path,split,*args,**kwargs):
                import numpy as np
                calls.append(split)
                return dict(auc=np.float64(.8),acc=.5,loss=.7)
            with patch('src.experiments.mentor_confirmation.load_checkpoint',return_value=payload):
                evaluate_primary(entries,Path(d)/'ok','cpu',evaluator=fake)
            self.assertEqual(calls,['ood_val','ood_val','ood_test','ood_test'])
            calls=[]
            with patch('src.experiments.mentor_confirmation.load_checkpoint',return_value=payload):
                evaluate_primary(entries,Path(d)/'validation_only','cpu',evaluator=fake,validation_only=True)
            self.assertEqual(calls,['ood_val','ood_val'])
            calls=[]
            def bad(path,split,*args,**kwargs):
                calls.append(split)
                return dict(auc=.81,acc=.5,loss=.7)
            with patch('src.experiments.mentor_confirmation.load_checkpoint',return_value=payload):
                with self.assertRaisesRegex(ValueError,'Validation parity failed'):
                    evaluate_primary(entries,Path(d)/'bad','cpu',evaluator=bad)
            self.assertNotIn('ood_test',calls)
