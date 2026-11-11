"""Global validation gate and immutable test selection contracts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from unittest.mock import patch

from src.experiments.checkpoints import digest
from src.experiments.context_test import evaluate_frozen
from src.experiments.context_transition import main as transition_main


class FrozenContextTests(unittest.TestCase):
    def entries(self,root):
        rows=[]
        for arm in ('scorer_only','scorer_and_classifier'):
            path=root/(arm+'.pt');path.write_bytes(arm.encode())
            rows.append(dict(dataset='ec50_assay',arm=arm,seed=7,status='completed',
                checkpoint=str(path),checkpoint_sha256=digest(path),validation={'auc':.75}))
        return rows

    def test_all_validation_before_any_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=self.entries(root);calls=[]
            def evaluate(checkpoint,split,*args,**kwargs):
                calls.append(split);return dict(auc=np.float64(.75 if split=='ood_val' else .7),loss=.5,acc=.6)
            with patch('src.experiments.context_test.evaluate_saved',side_effect=evaluate):
                result=evaluate_frozen(rows,{'ec50_assay':root},root/'out','cpu')
            self.assertEqual(calls,['ood_val','ood_val','ood_test','ood_test'])
            self.assertTrue(result['test_evaluated'])
            self.assertNotIn('metrics',rows[0])

    def test_validation_failure_prevents_first_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=self.entries(root);calls=[]
            def evaluate(checkpoint,split,*args,**kwargs):
                calls.append(split);return dict(auc=.75 if len(calls)==1 else .6)
            with patch('src.experiments.context_test.evaluate_saved',side_effect=evaluate):
                with self.assertRaises(ValueError):
                    evaluate_frozen(rows,{'ec50_assay':root},root/'out','cpu')
            self.assertNotIn('ood_test',calls)

    def test_tampered_checkpoint_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=self.entries(root);Path(rows[0]['checkpoint']).write_bytes(b'changed')
            with patch('src.experiments.context_test.evaluate_saved') as evaluate:
                with self.assertRaises(ValueError):
                    evaluate_frozen(rows,{'ec50_assay':root},root/'out','cpu')
            evaluate.assert_not_called()

    def test_transition_plan_has_explicit_total_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                plan=transition_main(['--project-root',tmp,'--output-dir',str(Path(tmp)/'out'),'--stage','all'])
            self.assertEqual(plan['new_training_runs'],36)
            self.assertEqual(plan['historical_test_checkpoints'],30)
            self.assertEqual(plan['official_test_checkpoints'],30)
            self.assertFalse((Path(tmp)/'out').exists())


if __name__=='__main__':unittest.main()
