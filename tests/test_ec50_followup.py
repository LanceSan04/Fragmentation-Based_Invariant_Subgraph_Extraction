import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from scripts.compare_inference import compare
from scripts.run_ec50_followup import audit_raw, summarize
from scripts.run_overnight import make_plan
from src.data.preprocess import run_preprocessing
from train_sugar import Config, run_training


class FollowupTests(unittest.TestCase):
    def test_identical_views_vote_equivalence_and_read_only(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            _, base = make_plan(root, 'cpu', True)
            base.update(epochs=1, lambda_irm=0., lambda_diversity=0., num_views=2,
                        view_fragmentations={'group_graph':2}, save_model=str(root/'model.pt'))
            run_preprocessing(Config(base))
            run_training(Config(base))
            payload = torch.load(root/'model.pt', weights_only=False)
            payload['view_state_dicts'][1] = payload['view_state_dicts'][0]
            torch.save(payload, root/'model.pt')
            before = hashlib.sha256((root/'model.pt').read_bytes()).hexdigest()
            report = compare(root/'model.pt', root/'comparison', processed_dir=root/'processed')
            self.assertEqual(before, hashlib.sha256((root/'model.pt').read_bytes()).hexdigest())
            for split in ['ood_val','ood_test']:
                self.assertEqual(report['splits'][split]['n'], 4)
                with np.load(root/'comparison'/f'{split}_predictions.npz') as arrays:
                    np.testing.assert_allclose(arrays['shared'], arrays['independent'], atol=1e-6)

    def test_raw_audit_detects_canonical_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.json'
            rows = [dict(smiles='CCO', cls_label=0, domain_id=1), dict(smiles='CCN', cls_label=1, domain_id=2)]
            path.write_text(json.dumps({'split': {s:rows for s in ['train','ood_val','ood_test']}}))
            report = audit_raw(path)
            self.assertEqual(report['canonical_overlap']['train__ood_test'], 2)

    def test_summary_rejects_duplicate_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'eval/runs').mkdir(parents=True)
            record = dict(seed=3001,run_name='one',ood_val={'auc':.8},ood_test={'auc':.7})
            (root/'eval/eval_results.json').write_text(json.dumps({'runs':[record]}))
            (root/'eval/runs/one.json').write_text(json.dumps({'config':{}}))
            with self.assertRaisesRegex(ValueError, 'Duplicate seed'):
                summarize([root,root], root/'summary.json')


if __name__ == '__main__':
    unittest.main()
