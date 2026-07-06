"""Tests for train_sugar.py config/CLI merging.

Run this file from the repository root with:

    .\.venv\Scripts\python.exe -m unittest tests.test_train_config -v
"""

import argparse
import json
import os.path as osp
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch.optim import Adam

from src.data.config_hash import preprocessing_config_hash
from train_sugar import (
    Config,
    build_shared_initialized_views,
    merge_config_and_cli,
    write_final_metrics_report,
)
from src.training.checkpoint import (
    checkpoint_feature_dims,
    load_checkpoint,
    load_view_state_dicts,
    save_training_checkpoint,
    view_state_dicts_from_checkpoint,
)


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
                "aggregation_mode": "max",
                "greedy_model_selection": True,
            }
        )

        self.assertEqual(
            preprocessing_config_hash(base),
            preprocessing_config_hash(changed_training),
        )

    def test_split_selection_overrides_do_not_change_preprocessing_hash(self):
        base = {
            "preprocess_version": "0.1.0",
            "dataset": "drugood_lbap_core_ec50_assay",
            "raw_format": "drugood_json",
            "raw_path": "data/raw/drugood/lbap_core_ec50_assay.json",
            "split_names": ["train", "iid_val", "ood_val", "iid_test", "ood_test"],
            "smiles_key": "smiles",
            "label_key": "cls_label",
            "environment_key": "domain_id",
            "metadata_keys": ["assay_id", "reg_label", "domain_id"],
            "fragmentation": "group_graph",
            "pooling": "mean_atom_and_internal_bond",
            "atom_feature_dim": 133,
            "bond_feature_dim": 14,
            "fragment_feature_dim": 149,
        }
        changed_selection = dict(base)
        changed_selection.update(
            {
                "train_split": "train",
                "validation_split": "ood_val",
                "test_split": "ood_test",
                "report_splits": ["iid_val", "iid_test"],
            }
        )

        self.assertEqual(
            preprocessing_config_hash(base),
            preprocessing_config_hash(changed_selection),
        )

    def test_fragment_ciga_views_share_initial_weights_but_have_distinct_sampler_seeds(self):
        torch.manual_seed(123)
        config = Config(
            {
                "num_views": 3,
                "num_classes": 2,
                "gnn_layers": 1,
                "hidden_dim": 16,
                "edge_ratio": 0.25,
                "dropout": 0.0,
                "fragment_sample_ratio": 0.95,
                "sampler_seed": 7,
            }
        )

        views = build_shared_initialized_views(
            config,
            atom_feat_dim=133,
            fragment_feat_dim=149,
            atom_edge_dim=14,
            fragment_edge_dim=14,
        )

        self.assertEqual(len(views), 3)
        first_state = views[0].state_dict()
        for view in views[1:]:
            for key, value in first_state.items():
                self.assertTrue(torch.equal(value, view.state_dict()[key]), key)

        self.assertEqual(
            [view.fragment_edge_sampler.seed for view in views],
            [7, 8, 9],
        )

    def test_full_checkpoint_saves_resume_and_eval_metadata(self):
        config = Config(
            {
                "num_views": 2,
                "num_classes": 2,
                "gnn_layers": 1,
                "hidden_dim": 8,
                "edge_ratio": 0.25,
                "dropout": 0.0,
                "fragment_sample_ratio": 1.0,
                "sampler_seed": 11,
                "processed_dir": "data/processed/example/group_graph",
                "aggregation_mode": "mean",
            }
        )
        views = build_shared_initialized_views(
            config,
            atom_feat_dim=133,
            fragment_feat_dim=149,
            atom_edge_dim=14,
            fragment_edge_dim=14,
        )
        trainer = SimpleNamespace(
            views=views,
            optimizers=[Adam(view.parameters(), lr=0.001) for view in views],
        )
        feature_dims = {
            "atom_feature_dim": 133,
            "fragment_feature_dim": 149,
            "atom_edge_dim": 14,
            "fragment_edge_dim": 14,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = osp.join(tmp_dir, "checkpoint.pt")
            save_training_checkpoint(
                path,
                trainer,
                config,
                feature_dims,
                history={"train_loss": [1.0]},
                selected_view_indices=[0],
                metrics={"ood_test": {"auc": 0.75}},
                epoch=1,
            )
            payload = load_checkpoint(path)

        self.assertEqual(payload["checkpoint_version"], "0.1")
        self.assertEqual(payload["config"]["processed_dir"], "data/processed/example/group_graph")
        self.assertEqual(payload["feature_dims"], feature_dims)
        self.assertEqual(payload["selected_view_indices"], [0])
        self.assertEqual(payload["epoch"], 1)
        self.assertEqual(len(payload["view_state_dicts"]), 2)
        self.assertEqual(len(payload["optimizer_state_dicts"]), 2)
        self.assertEqual(checkpoint_feature_dims(payload), feature_dims)

    def test_final_metrics_report_writes_metrics_and_metadata(self):
        config = Config(
            {
                "dataset": "drugood_lbap_core_ec50_assay",
                "fragmentation": "group_graph",
                "processed_dir": "data/processed/drugood_lbap_core_ec50_assay/group_graph",
                "train_split": "train",
                "validation_split": "ood_val",
                "test_split": "ood_test",
                "report_splits": ["iid_test"],
                "num_views": 2,
                "aggregation_mode": "mean",
                "save_model": "checkpoints/example.pt",
                "loss_log_path": "logs/example.csv",
            }
        )
        metrics = {
            "ood_test": {"loss": 0.5, "acc": 0.75, "auc": 0.82},
            "iid_test": {"loss": 0.4, "acc": 0.8, "auc": 0.9},
        }
        feature_dims = {
            "atom_feature_dim": 133,
            "fragment_feature_dim": 149,
            "atom_edge_dim": 14,
            "fragment_edge_dim": 14,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = osp.join(tmp_dir, "final_eval_metrics.json")
            write_final_metrics_report(
                path,
                config,
                metrics,
                selected_view_indices=[0, 1],
                feature_dims=feature_dims,
                epoch=3,
            )
            with open(path, "r", encoding="utf-8") as handle:
                report = json.load(handle)

        self.assertEqual(report["metrics"]["ood_test"]["auc"], 0.82)
        self.assertEqual(report["metrics"]["iid_test"]["acc"], 0.8)
        self.assertEqual(report["metadata"]["dataset"], "drugood_lbap_core_ec50_assay")
        self.assertEqual(report["metadata"]["fragmentation"], "group_graph")
        self.assertEqual(report["metadata"]["selected_view_indices"], [0, 1])
        self.assertEqual(report["metadata"]["epoch"], 3)
        self.assertEqual(report["feature_dims"], feature_dims)

    def test_checkpoint_loader_accepts_legacy_view_state_dict_payload(self):
        config = Config(
            {
                "num_views": 1,
                "num_classes": 2,
                "gnn_layers": 1,
                "hidden_dim": 8,
                "edge_ratio": 0.25,
                "dropout": 0.0,
                "fragment_sample_ratio": 1.0,
                "sampler_seed": 3,
            }
        )
        source = build_shared_initialized_views(
            config,
            atom_feat_dim=133,
            fragment_feat_dim=149,
            atom_edge_dim=14,
            fragment_edge_dim=14,
        )
        target = build_shared_initialized_views(
            config,
            atom_feat_dim=133,
            fragment_feat_dim=149,
            atom_edge_dim=14,
            fragment_edge_dim=14,
        )
        legacy_payload = {0: source[0].state_dict()}

        self.assertEqual(set(view_state_dicts_from_checkpoint(legacy_payload)), {0})
        load_view_state_dicts(target, legacy_payload)

        for key, value in source[0].state_dict().items():
            self.assertTrue(torch.equal(value, target[0].state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
