import contextlib,io,tempfile,unittest
from pathlib import Path
import torch
from scripts.run_overnight import make_plan
from src.data.preprocess import run_preprocessing
from train_sugar import Config,run_training,evaluate_checkpoint,build_trainer,apply_checkpoint_defaults
from src.models.sugar_ensemble import SuGAREnsemble

class InferencePolicyTests(unittest.TestCase):
    def test_independent_training_selection_and_checkpoint(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d);_,base=make_plan(root,'cpu',True)
            base.update(epochs=2,lambda_irm=0.,lambda_diversity=0.,inference_policy='independent',
                        save_model=str(root/'model.pt'),test_split='ood_test')
            cfg=Config(base);run_preprocessing(cfg);result=run_training(cfg)
            ensemble=result['ensemble'];self.assertEqual(ensemble.inference_policy,'independent')
            from src.data.dataloader import build_split_dataloaders
            from train_sugar import strip_non_tensor_batch_fields
            loaders=build_split_dataloaders(cfg.processed_dir,cfg);strip_non_tensor_batch_fields(loaders)
            batch=next(iter(loaders['ood_val']));ensemble.eval()
            with torch.no_grad():
                manual=torch.stack([v(batch).exp() for v in ensemble.views]).mean(0)
                torch.testing.assert_close(ensemble(batch).exp(),manual)
                torch.testing.assert_close(ensemble(batch,view_indices=[1]).exp(),ensemble.views[1](batch).exp())
            trainer=build_trainer(cfg,ensemble.views,'cpu')
            self.assertEqual(trainer.ensemble.inference_policy,'independent')
            payload=torch.load(root/'model.pt',weights_only=False)
            self.assertEqual(payload['config']['inference_policy'],'independent')
            restored=apply_checkpoint_defaults(Config({}),payload)
            self.assertEqual(restored.inference_policy,'independent')
            evaluated=evaluate_checkpoint(restored,root/'model.pt',split_names=['ood_val','ood_test'])
            self.assertIsNotNone(evaluated)
    def test_default_and_invalid_policy(self):
        self.assertEqual(SuGAREnsemble([]).inference_policy,'shared')
        with self.assertRaises(ValueError):SuGAREnsemble([],inference_policy='typo')

if __name__=='__main__':unittest.main()
