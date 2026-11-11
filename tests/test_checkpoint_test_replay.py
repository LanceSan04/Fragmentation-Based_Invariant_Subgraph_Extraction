import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from src.experiments.checkpoint_test_replay import replay,compare_predictions,validate
from src.experiments.checkpoints import digest


class CheckpointTestReplayTests(unittest.TestCase):
    def fixture(self,root):
        arrays=dict(labels=np.array([0,1,0,1]),probabilities=np.array([.2,.8,.3,.9],dtype=np.float32),
            assays=np.array(['a','a','b','b']),row_index=np.arange(4))
        reference=root/'reference.npz';np.savez(reference,**arrays)
        rows=[]
        for cohort in ['original','first_rerun']:
            checkpoint=root/(cohort+'.pt');checkpoint.write_bytes(b'fixture')
            rows.append(dict(cohort=cohort,dataset='ec50_assay',seed=11422,
                checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),cache=str(root),
                validation_reference=str(reference),test_reference=str(reference),validation_auc=1.,test_auc=1.))
        return rows,arrays

    def test_all_validation_precedes_two_fresh_test_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,arrays=self.fixture(root);calls=[]
            def evaluate(cp,split,*args,**kwargs):
                calls.append(split);np.savez(kwargs['prediction_path'],**arrays);return dict(auc=1.,loss=0.,acc=1.)
            with patch('src.experiments.checkpoint_test_replay.evaluate_saved',side_effect=evaluate):
                result=replay(rows,root/'out')
            self.assertEqual(calls,['ood_val']*2+['ood_test']*4)
            self.assertTrue(result['all_test_predictions_exact'])
            self.assertEqual(result['test_evaluations'],4)

    def test_same_auc_does_not_hide_probability_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,arrays=self.fixture(root)
            def evaluate(cp,split,*args,**kwargs):
                values=dict(arrays)
                if split=='ood_test':values['probabilities']=arrays['probabilities']+.0001
                np.savez(kwargs['prediction_path'],**values);return dict(auc=1.)
            with patch('src.experiments.checkpoint_test_replay.evaluate_saved',side_effect=evaluate):
                with self.assertRaisesRegex(ValueError,'Fresh test probabilities differ'):replay(rows,root/'out')
            report=json.loads((root/'out/audit_results.json').read_text())
            self.assertEqual(len(report['tests']),4)
            self.assertTrue(all(r['auc_matches_saved'] for r in report['tests']))
            self.assertFalse(report['all_test_predictions_exact'])

    def test_validation_mismatch_blocks_all_test_forwards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,arrays=self.fixture(root);calls=[]
            def evaluate(cp,split,*args,**kwargs):
                calls.append(split);values=dict(arrays,probabilities=arrays['probabilities']+.01)
                np.savez(kwargs['prediction_path'],**values);return dict(auc=1.)
            with patch('src.experiments.checkpoint_test_replay.evaluate_saved',side_effect=evaluate):
                with self.assertRaisesRegex(ValueError,'Validation replay changed'):replay(rows,root/'out')
            self.assertNotIn('ood_test',calls)

    def test_frozen_scope_and_metadata_identity(self):
        spec=json.loads((Path(__file__).resolve().parents[1]/'configs/checkpoint_test_replay_20261103.json').read_text())
        validate(spec);spec['passes_per_checkpoint']=3
        with self.assertRaises(ValueError):validate(spec)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,arrays=self.fixture(root);other=dict(arrays,row_index=np.array([1,0,2,3]))
            np.savez(root/'other.npz',**other)
            self.assertFalse(compare_predictions(root/'other.npz',rows[0]['test_reference'])['exact'])


if __name__=='__main__':unittest.main()
