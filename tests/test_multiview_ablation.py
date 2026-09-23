import unittest
from pathlib import Path
import yaml
from src.data.config_hash import preprocessing_config_hash
from src.fragmentation.multi_view import expand_view_fragmentations
from scripts.run_multiview_ablation import build_commands

ROOT = Path(__file__).resolve().parents[1]


class MultiviewAblationTests(unittest.TestCase):
    def test_stages_and_seed_pairing(self):
        for stage, count in [('views', 3), ('auxiliary', 9)]:
            commands = build_commands(stage)
            self.assertEqual(len(commands), count)
            self.assertEqual(len(set(str(p) for p, _ in commands)), count)
            self.assertEqual([c[c.index('--seed') + 1] for _, c in commands], ['11', '22', '33'] * (count // 3))

    def test_auxiliary_configs_change_only_intended_loss(self):
        def read(name):
            return yaml.safe_load((ROOT / 'configs' / f'multiview_{name}.yaml').read_text())
        base = read('prediction')
        self.assertEqual(expand_view_fragmentations(base), ['group_graph'] * 5)
        for name, allowed in [('diversity', {'lambda_diversity'}), ('contrastive', {'lambda_contrast'}), ('irm', {'lambda_irm', 'num_envs'})]:
            cfg = read(name)
            self.assertEqual({k for k in cfg if cfg[k] != base[k]}, allowed)
            self.assertEqual(preprocessing_config_hash(cfg), preprocessing_config_hash(base))
        self.assertEqual([base[k] for k in ['lambda_contrast', 'lambda_diversity', 'lambda_irm']], [0, 0, 0])

    def test_primary_and_search_policy_agree(self):
        main = yaml.safe_load((ROOT / 'configs/experiment.yaml').read_text())
        search = yaml.safe_load((ROOT / 'configs/grid_search.yaml').read_text())
        self.assertEqual(main['selection_ma_window'], 1)
        self.assertEqual(search['auc_ma_window'], 1)
        self.assertFalse(main['early_stopping'])


if __name__ == '__main__':
    unittest.main()
