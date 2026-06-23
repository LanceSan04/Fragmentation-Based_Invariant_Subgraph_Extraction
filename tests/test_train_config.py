"""Tests for train_sugar.py config/CLI merging.

Run this file from the repository root with:

    .\.venv\Scripts\python.exe -m unittest tests.test_train_config -v
"""

import argparse
import unittest

from src.data.config_hash import preprocessing_config_hash
from train_sugar import merge_config_and_cli


class TrainConfigTests(unittest.TestCase):
    def test_cli_overrides_only_explicit_values(self):
        config = {
            "processed_dir": "data/processed/bbbp/group_graph",
            "batch_size": 32,
            "hidden_dim": 128,
            "lambda_irm": 0.0,
            "num_envs": 1,
        }
        args = argparse.Namespace(
            config="configs/experiment.yaml",
            processed_dir=None,
            batch_size=4,
            hidden_dim=None,
            lambda_irm=1.0,
            num_envs=None,
        )

        merged = merge_config_and_cli(config, args)

        self.assertEqual(merged["processed_dir"], "data/processed/bbbp/group_graph")
        self.assertEqual(merged["batch_size"], 4)
        self.assertEqual(merged["hidden_dim"], 128)
        self.assertEqual(merged["lambda_irm"], 1.0)
        self.assertEqual(merged["num_envs"], 1)
        self.assertNotIn("config", merged)

    def test_training_only_overrides_do_not_change_preprocessing_hash(self):
        base = {
            "preprocess_version": "0.1.0",
            "dataset": "bbbp",
            "raw_path": "data/raw/bbbp/BBBP.csv",
            "smiles_key": "smiles",
            "label_key": "p_np",
            "fragmentation": "group_graph",
            "pooling": "mean_atom_and_internal_bond",
            "atom_feature_dim": 133,
            "bond_feature_dim": 14,
            "fragment_feature_dim": 149,
        }
        changed_training = dict(base)
        changed_training.update(
            {
                "hidden_dim": 256,
                "num_views": 3,
                "lambda_contrast": 0.0,
                "lambda_diversity": 0.2,
                "lambda_irm": 0.0,
                "num_envs": 1,
            }
        )

        self.assertEqual(
            preprocessing_config_hash(base),
            preprocessing_config_hash(changed_training),
        )


if __name__ == "__main__":
    unittest.main()
