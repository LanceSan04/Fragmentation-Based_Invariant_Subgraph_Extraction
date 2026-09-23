"""Controlled graph, checkpoint and search-isolation regression checks."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from tests.test_reconstruction_invariants import build_batch
from tests import test_sugar_scorer as scorer_tests
from src.training.sugar_trainer import SuGARTrainer
from src.training.selection_metrics import trailing_mean, best_moving_average
from scripts.grid_search import resolve_overrides, run_once


class ScriptedTrainer(SuGARTrainer):
    """Encode epoch in a parameter so restoration is directly observable."""
    def __init__(self, scores, **kwargs):
        super().__init__([torch.nn.Linear(1, 1, bias=False)], **kwargs)
        self.scores = scores
        self.epoch = 0

    def train_epoch(self, loader, return_loss_components=False):
        self.epoch += 1
        with torch.no_grad():
            self.views[0].weight.fill_(self.epoch)
        return 0.5, 0.5, dict.fromkeys(self.loss_component_keys(), 0.0)

    def evaluate_all(self, loader, return_loss_components=False):
        score = self.scores[self.epoch - 1]
        result = (0.5, 0.5, score, dict.fromkeys(self.loss_component_keys(), 0.0),
                  [{"acc": 0.5, "auc": score}])
        return result if return_loss_components else result[:3] + (result[4],)


class SelectionAndERMTests(unittest.TestCase):
    def test_diversity_schedule_stopping_and_early_checkpoint(self):
        trainer = ScriptedTrainer([.9]+[.5]*9, lambda_diversity=.1,
                                  diversity_delay_epochs=2, diversity_ramp_epochs=2,
                                  early_stopping_min_epochs=5)
        with contextlib.redirect_stdout(io.StringIO()):
            history = trainer.train([], [None], num_epochs=10, patience=2)
        self.assertEqual(len(history['train_loss']), 7)
        self.assertEqual(history['effective_diversity_weight'], [0.,0.,.05,.1,.1,.1,.1])
        self.assertEqual(history['selected_checkpoint_epoch'], 1)
        self.assertEqual(trainer.views[0].weight.item(), 1)

    def run_scripted(self, scores, window, **kwargs):
        trainer = ScriptedTrainer(scores, selection_ma_window=window, **kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            history = trainer.train([], val_loader=[None], num_epochs=len(scores), patience=20)
        return trainer, history

    def test_early_spike_cannot_beat_complete_window(self):
        scores = [.99, .1, .1, .1, .1, .8, .8, .8, .8, .8]
        trainer, history = self.run_scripted(scores, 5)
        self.assertEqual(trainer.views[0].weight.item(), 10)
        self.assertEqual(history['selected_checkpoint_epoch'], 10)
        self.assertEqual(history['val_selection_metric_smoothed'][:4], [None] * 4)
        self.assertAlmostEqual(history['best_selection_metric_smoothed'], best_moving_average(scores, 5))

    def test_top_k_includes_non_record_breaker(self):
        trainer, _ = self.run_scripted([.9, .9, .7, .5], 2, selection_weight_avg=True)
        # Complete MA scores: epoch 2=.9, epoch 3=.8, epoch 4=.6.
        self.assertEqual(trainer.views[0].weight.item(), 2.5)

    def test_warmup_does_not_spend_patience(self):
        trainer = ScriptedTrainer([.9, .1, .1, .1, .1], selection_ma_window=5)
        with contextlib.redirect_stdout(io.StringIO()):
            history = trainer.train([], [None], num_epochs=5, patience=1)
        self.assertEqual(len(history['train_loss']), 5)
        self.assertEqual(history['selected_checkpoint_epoch'], 5)

    def test_erm_full_skips_fragment_analysis(self):
        import yaml
        from main import build_parser, run_full_command
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / 'config.yaml'
            config_path.write_text(yaml.safe_dump(dict(
                dataset='unit', output_root=directory, processed_dir=directory,
                run_name='erm', num_views=1, view_fragmentations={'group_graph': 1},
                erm_baseline=True, save_outputs={'checkpoint': True, 'subgraph_analysis': True,
                'training_curves': False, 'preprocessing_fragmentation_diagnostics': False})))
            args = build_parser().parse_args(['full', '--config', str(config_path)])
            with patch('src.data.preprocess.run_preprocessing'), patch('main.run_training', return_value={}), patch('main.run_subgraph_diagnostics') as analyze:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_full_command(args)
            analyze.assert_not_called()

    def test_seed_and_selection_cli_overrides(self):
        from main import build_parser, apply_common_overrides
        args = build_parser().parse_args(['train', '--seed', '11', '--sampler-seed', '12', '--selection-ma-window', '1'])
        config = apply_common_overrides({'processed_dir': 'unused'}, args)
        self.assertEqual((config['seed'], config['sampler_seed'], config['selection_ma_window']), (11, 12, 1))

    def test_control_configs_are_matched_and_valid(self):
        import yaml
        from src.fragmentation.multi_view import expand_view_fragmentations
        root = Path(__file__).resolve().parents[1]
        configs = [yaml.safe_load((root / 'configs' / name).read_text())
                   for name in ('erm_baseline.yaml', 'fragment_control.yaml')]
        for config in configs:
            self.assertEqual(expand_view_fragmentations(config), ['group_graph'])
            self.assertIsNone(config['test_split'])
            self.assertEqual([config[k] for k in ('lambda_contrast', 'lambda_diversity', 'lambda_irm')], [0, 0, 0])
        configs[0]['erm_baseline'] = False
        self.assertEqual(configs[0], configs[1])

    def test_raw_selection_remains_supported(self):
        trainer, _ = self.run_scripted([.8, .7, .6], 1)
        self.assertEqual(trainer.views[0].weight.item(), 1)

    def test_invalid_and_incomplete_windows(self):
        self.assertIsNone(trailing_mean([.8], 2))
        self.assertIsNone(trailing_mean([.8, float('nan')], 2))
        self.assertIsNone(best_moving_average([None, .8], 2))
        with self.assertRaises(ValueError):
            trailing_mean([.8], 0)
        with self.assertRaisesRegex(ValueError, 'epochs must be'):
            self.run_scripted([.8], 5)

    def test_erm_matches_full_bidirectional_classifier_and_skips_masker(self):
        batch = build_batch()
        view = scorer_tests.SuGArScorerTests()._make_view(batch, erm_baseline=True)
        view.eval()
        with patch.object(view.fragment_masker, 'forward', side_effect=AssertionError('masker called')):
            with torch.no_grad():
                expected = F.log_softmax(view.atom_classifier(batch), dim=-1)
                actual = view(batch)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(batch.edge_index.size(1), 2 * batch.bond_index.size(1))
        with self.assertRaisesRegex(ValueError, 'not applicable'):
            view.get_invariant_subgraph(batch)

    def test_batchnorm_weight_averaging_is_explicitly_guarded(self):
        view = scorer_tests.SuGArScorerTests()._make_view(build_batch())
        with self.assertRaisesRegex(ValueError, 'recalibration'):
            SuGARTrainer([view], selection_weight_avg=True)

    def test_search_window_is_explicit_and_conflicts_fail(self):
        args = SimpleNamespace(epochs=None, device=None)
        self.assertEqual(resolve_overrides({'auc_ma_window': 3}, args)['selection_ma_window'], 3)
        with self.assertRaisesRegex(ValueError, 'must match'):
            resolve_overrides({'auc_ma_window': 3, 'overrides': {'selection_ma_window': 5}}, args)

    def test_search_suppresses_test_even_when_combo_requests_it(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.grid_search.run_training', return_value={'metrics': {}, 'history': {}}) as train:
                run_once({}, {}, {'test_split': 'ood_test', 'report_splits': ['iid_test'],
                                  'log_test_curve': True}, 7, 'unit', Path(directory), evaluate_test=False)
            config = train.call_args[0][0]
            self.assertIsNone(config.test_split)
            self.assertEqual(config.report_splits, [])
            self.assertFalse(config.log_test_curve)


if __name__ == '__main__':
    unittest.main()
