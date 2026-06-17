"""Regression tests for the fragmentation/preprocessing foundation.

Run this file from the repository root with:

    .\.venv\Scripts\python.exe -m unittest tests.test_fragment_pipeline -v

Run the full test suite with:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

import os.path as osp
import tempfile
import unittest
import csv

import torch
from torch_geometric.data import Batch

from src.data.dataloader import build_split_dataloaders, validate_dataloader_batch
from src.data.molecule_example import build_molecule_example
from src.data.pyg_batch import molecule_example_to_data
from src.data.preprocess import run_preprocessing
from src.fragmentation import build_fragmenter
from src.models.atom_reconstructor import reconstruct_atom_graph, reconstruct_batch


class FragmentPipelineTests(unittest.TestCase):
    def build_data(self, smiles="CCOC(=O)C", fragmentation="group_graph"):
        example = build_molecule_example(
            smiles,
            label=1,
            molecule_id=f"test_{fragmentation}",
            fragmenter=build_fragmenter(fragmentation),
        )
        return molecule_example_to_data(example)

    def test_fragment_identity_survives_example_data_and_batch_roundtrip(self):
        data = self.build_data("CCOC(=O)C", "group_graph")

        self.assertTrue(hasattr(data, "fragment_types"))
        self.assertTrue(hasattr(data, "fragment_smiles"))
        self.assertEqual(len(data.fragment_types), data.fragment_x_static.size(0))
        self.assertEqual(len(data.fragment_smiles), data.fragment_x_static.size(0))
        self.assertTrue(all(isinstance(value, str) for value in data.fragment_types))
        self.assertTrue(all(isinstance(value, str) for value in data.fragment_smiles))

        batch = Batch.from_data_list([data])
        recovered = batch.to_data_list()[0]
        self.assertEqual(recovered.fragment_types, data.fragment_types)
        self.assertEqual(recovered.fragment_smiles, data.fragment_smiles)

    def test_batched_molecule_mappings_remain_local(self):
        data_list = [
            self.build_data("CCOC(=O)C", "group_graph"),
            self.build_data("c1ccccc1", "group_graph"),
        ]
        batch = Batch.from_data_list(data_list, follow_batch=["fragment_x_static"])
        validate_dataloader_batch(batch)

        recovered = batch.to_data_list()
        self.assertEqual(len(recovered), 2)
        for original, restored in zip(data_list, recovered):
            self.assertTrue(torch.equal(original.atom_to_fragment, restored.atom_to_fragment))
            self.assertTrue(torch.equal(original.fragment_to_atom_index, restored.fragment_to_atom_index))
            self.assertTrue(torch.equal(original.fragment_edge_index, restored.fragment_edge_index))
            self.assertTrue(
                torch.equal(
                    original.fragment_edge_to_attachment_atoms,
                    restored.fragment_edge_to_attachment_atoms,
                )
            )

    def test_all_fragment_edges_reconstruct_original_atom_graph(self):
        self.assert_all_fragment_edges_reconstruct_original_atom_graph("group_graph")

    def test_all_brics_fragment_edges_reconstruct_original_atom_graph(self):
        self.assert_all_fragment_edges_reconstruct_original_atom_graph("brics")

    def test_group_graph_resolves_overlapping_split_fragments(self):
        data = self.build_data(
            "N[S](=O)(=O)c1cc2c(NC=N[S]2(=O)=O)cc1Cl",
            "group_graph",
        )

        self.assertEqual(data.atom_to_fragment.numel(), data.x.size(0))
        for fragment_id in range(data.fragment_x_static.size(0)):
            atoms = data.fragment_to_atom_index[1][
                data.fragment_to_atom_index[0] == fragment_id
            ]
            self.assertTrue(torch.all(data.atom_to_fragment[atoms] == fragment_id))

    def assert_all_fragment_edges_reconstruct_original_atom_graph(self, fragmentation):
        data = self.build_data("CCOC(=O)C", fragmentation)
        mask = torch.ones(data.fragment_edge_index.size(1), dtype=torch.bool)

        reconstructed = reconstruct_atom_graph(data, mask)

        self.assertEqual(reconstructed.x.size(0), data.x.size(0))
        self.assertEqual(reconstructed.bond_attr.size(0), data.bond_attr.size(0))
        self.assertEqual(reconstructed.edge_index.size(1), data.edge_index.size(1))
        self.assertEqual(
            sorted(reconstructed.atom_ids.tolist()),
            sorted(data.atom_ids.tolist()),
        )
        self.assertEqual(
            sorted(reconstructed.bond_ids.tolist()),
            sorted(data.bond_ids.tolist()),
        )
        self.assertTrue(torch.equal(reconstructed.selected_fragment_edge_ids, torch.arange(mask.numel())))
        self.assertTrue(
            torch.equal(
                reconstructed.selected_attachment_bond_ids,
                data.fragment_edge_to_bond,
            )
        )
        self.assertTrue(
            torch.all(
                reconstructed.reconstructed_bond_is_internal
                | reconstructed.reconstructed_bond_is_selected_boundary
            )
        )
        self.assertTrue(torch.any(reconstructed.reconstructed_bond_is_internal))
        self.assertTrue(torch.any(reconstructed.reconstructed_bond_is_selected_boundary))

    def test_dataloader_rejects_stale_config_hash(self):
        config = {
            "preprocess_version": "0.1.0",
            "dataset": "hash_test",
            "fragmentation": "group_graph",
            "pooling": "mean_atom_and_internal_bond",
            "atom_feature_dim": 133,
            "bond_feature_dim": 14,
            "fragment_feature_dim": 149,
            "batch_size": 1,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config["processed_dir"] = tmp_dir
            torch.save(
                {
                    "metadata": {
                        "preprocess_version": "0.1.0",
                        "dataset": "hash_test",
                        "fragmentation": "group_graph",
                        "pooling": "mean_atom_and_internal_bond",
                        "atom_feature_dim": 133,
                        "bond_feature_dim": 14,
                        "fragment_feature_dim": 149,
                        "config_hash": "stale",
                    },
                    "molecule_data": [],
                },
                osp.join(tmp_dir, "train.pt"),
            )

            with self.assertRaisesRegex(ValueError, "config_hash"):
                build_split_dataloaders(tmp_dir, config)

    def test_preprocessing_writes_failure_report_for_skipped_records(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "dataset": "failure_report_test",
                "preprocess_version": "0.1.0",
                "fragmentation": "group_graph",
                "pooling": "mean_atom_and_internal_bond",
                "processed_dir": tmp_dir,
                "smiles_key": "smiles",
                "label_key": "label",
                "molecule_id_key": "molecule_id",
                "strict": False,
                "debug": False,
                "splits": {
                    "train": [
                        {"molecule_id": "good", "smiles": "CC", "label": 1},
                        {"molecule_id": "bad", "smiles": "not_a_smiles", "label": 0},
                    ]
                },
            }

            result = run_preprocessing(config)

            self.assertIsNotNone(result["failure_report"])
            self.assertTrue(osp.exists(result["failure_report"]))
            with open(result["failure_report"], newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["split"], "train")
            self.assertEqual(rows[0]["molecule_id"], "bad")
            self.assertIn("RDKit could not parse SMILES", rows[0]["error"])

    def test_single_fragment_reconstruction_keeps_full_graph_without_edges(self):
        data = self.build_data("c1ccccc1", "group_graph")
        self.assertEqual(data.fragment_edge_index.size(1), 0)

        reconstructed = reconstruct_atom_graph(data, torch.empty((0,), dtype=torch.bool))

        self.assertEqual(reconstructed.x.size(0), data.x.size(0))
        self.assertEqual(reconstructed.bond_attr.size(0), data.bond_attr.size(0))
        self.assertEqual(reconstructed.selected_fragment_ids.tolist(), [0])
        self.assertEqual(reconstructed.selected_fragment_edge_ids.numel(), 0)
        self.assertEqual(reconstructed.selected_attachment_bond_ids.numel(), 0)
        self.assertTrue(torch.all(reconstructed.reconstructed_bond_is_internal))
        self.assertFalse(torch.any(reconstructed.reconstructed_bond_is_selected_boundary))

    def test_reconstruct_batch_splits_selected_edges_per_molecule(self):
        data_list = [
            self.build_data("CCOC(=O)C", "group_graph"),
            self.build_data("c1ccccc1", "group_graph"),
        ]
        batch = Batch.from_data_list(data_list, follow_batch=["fragment_x_static"])
        mask = torch.cat(
            [
                torch.ones(data_list[0].fragment_edge_index.size(1), dtype=torch.bool),
                torch.empty((0,), dtype=torch.bool),
            ]
        )

        reconstructed_batch = reconstruct_batch(batch, mask)
        reconstructed = reconstructed_batch.to_data_list()

        self.assertEqual(len(reconstructed), 2)
        self.assertEqual(reconstructed[0].x.size(0), data_list[0].x.size(0))
        self.assertEqual(reconstructed[1].x.size(0), data_list[1].x.size(0))
        self.assertEqual(
            sorted(reconstructed[0].bond_ids.tolist()),
            sorted(data_list[0].bond_ids.tolist()),
        )
        self.assertEqual(
            sorted(reconstructed[1].bond_ids.tolist()),
            sorted(data_list[1].bond_ids.tolist()),
        )


if __name__ == "__main__":
    unittest.main()
