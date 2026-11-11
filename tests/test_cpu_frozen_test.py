import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.experiments.checkpoints import digest
from src.experiments.context_test import evaluate_frozen
from src.experiments.cpu_frozen_test import main,validate


class CpuFrozenTestTests(unittest.TestCase):
    def setup_entries(self,root):
        entries=[];refs={}
        for arm in ('scorer_only','scorer_and_classifier'):
            ck=root/(arm+'.pt');ck.write_bytes(arm.encode());ref=root/(arm+'.npz')
            np.savez(ref,labels=np.array([0,1]),probabilities=np.array([.2,.8],dtype=np.float32))
            entries.append(dict(dataset='ec50_assay',arm=arm,seed=1,status='completed',
                checkpoint=str(ck),checkpoint_sha256=digest(ck),validation={'auc':1.0},
                original_gpu_selection_auc=.9999,selected_epoch=10))
            refs[('ec50_assay',arm,1)]=ref
        return entries,refs

    def test_equal_auc_but_changed_probabilities_blocks_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,refs=self.setup_entries(root);calls=[]
            def evaluate(checkpoint,split,*args,**kwargs):
                calls.append(split)
                np.savez(kwargs['prediction_path'],labels=np.array([0,1]),
                         probabilities=np.array([.3,.7],dtype=np.float32))
                return dict(auc=1.0,acc=1.0,loss=.2)
            with patch('src.experiments.context_test.evaluate_saved',side_effect=evaluate):
                with self.assertRaisesRegex(ValueError,'reference mismatch'):
                    evaluate_frozen(rows,{'ec50_assay':root},root/'out','cpu',validation_references=refs)
            self.assertNotIn('ood_test',calls)

    def test_exact_references_globally_gate_tests_and_preserve_gpu_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,refs=self.setup_entries(root);calls=[]
            def evaluate(checkpoint,split,*args,**kwargs):
                calls.append(split)
                np.savez(kwargs['prediction_path'],labels=np.array([0,1]),
                         probabilities=np.array([.2,.8],dtype=np.float32))
                return dict(auc=1.0,acc=1.0,loss=.2)
            with patch('src.experiments.context_test.evaluate_saved',side_effect=evaluate):
                report=evaluate_frozen(rows,{'ec50_assay':root},root/'out','cpu',validation_references=refs)
            self.assertEqual(calls,['ood_val','ood_val','ood_test','ood_test'])
            self.assertEqual(report['verification_policy'],'exact_cpu_prediction_reference')
            self.assertEqual(report['runs'][0]['original_gpu_selection_auc'],.9999)
            self.assertEqual(report['runs'][0]['selected_epoch'],10)

    def test_frozen_budget_and_preview(self):
        config=Path(__file__).resolve().parents[1]/'configs/cpu_frozen_test_20261030.json'
        spec=json.loads(config.read_text());validate(spec)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'out'
            with contextlib.redirect_stdout(io.StringIO()):
                plan=main(['--config',str(config),'--project-root',tmp,'--output-dir',str(out)])
            self.assertEqual(plan['test_evaluations'],30)
            self.assertEqual(plan['training_runs'],0)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
