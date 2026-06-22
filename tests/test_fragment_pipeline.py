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
            self.assert_fragment_atoms_connected(data, atoms)

    def test_group_graph_splits_disconnected_repaired_fragments(self):
        data = self.build_data("[Na+].[Na+].[Na+]", "group_graph")

        self.assertEqual(data.fragment_x_static.size(0), 3)
        self.assertEqual(data.fragment_smiles, ["[Na+]", "[Na+]", "[Na+]"])
        for fragment_id in range(data.fragment_x_static.size(0)):
            atoms = data.fragment_to_atom_index[1][
                data.fragment_to_atom_index[0] == fragment_id
            ]
            self.assertEqual(atoms.numel(), 1)
            self.assert_fragment_atoms_connected(data, atoms)

    def test_molecule_example_validates_fragment_edge_attr_shape(self):
        example = build_molecule_example(
            "CCOC(=O)C",
            label=1,
            molecule_id="bad_fragment_edge_attr",
            fragmenter=build_fragmenter("group_graph"),
        )
        example.fragment_edge_attr = example.fragment_edge_attr[:, :-1]

        with self.assertRaisesRegex(ValueError, "fragment_edge_attr"):
            example.validate()

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

    def test_dataloader_allows_non_preprocessing_config_changes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "dataset": "hash_scope_test",
                "preprocess_version": "0.1.0",
                "fragmentation": "group_graph",
                "pooling": "mean_atom_and_internal_bond",
                "processed_dir": tmp_dir,
                "smiles_key": "smiles",
                "label_key": "label",
                "molecule_id_key": "molecule_id",
                "strict": False,
                "debug": False,
                "batch_size": 1,
                "splits": {
                    "train": [
                        {"molecule_id": "mol1", "smiles": "CC", "label": 1},
                        {"molecule_id": "mol2", "smiles": "CCC", "label": 0},
                    ]
                },
            }
            run_preprocessing(config)

            changed_loader_config = dict(config)
            changed_loader_config["batch_size"] = 2
            loaders = build_split_dataloaders(tmp_dir, changed_loader_config)

            self.assertIn("train", loaders)
            batch = next(iter(loaders["train"]))
            validate_dataloader_batch(batch)

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

    def test_reconstructed_batch_offsets_directed_edge_to_bond(self):
        data_list = [
            self.build_data("CCOC(=O)C", "group_graph"),
            self.build_data("CCOC(=O)C", "group_graph"),
        ]
        reconstructed = []
        for data in data_list:
            mask = torch.ones(data.fragment_edge_index.size(1), dtype=torch.bool)
            reconstructed.append(reconstruct_atom_graph(data, mask))

        batch = Batch.from_data_list(reconstructed)
        first_num_directed_edges = reconstructed[0].edge_index.size(1)
        first_num_bonds = reconstructed[0].bond_attr.size(0)

        self.assertTrue(
            torch.all(batch.directed_edge_to_bond[:first_num_directed_edges] < first_num_bonds)
        )
        self.assertTrue(
            torch.all(batch.directed_edge_to_bond[first_num_directed_edges:] >= first_num_bonds)
        )
        self.assertLess(int(batch.directed_edge_to_bond.max()), batch.bond_attr.size(0))

    def test_partial_reconstruction_keeps_attachment_provenance_unoffset(self):
        data = self.build_data("CCOC(=O)C", "group_graph")
        mask = torch.zeros(data.fragment_edge_index.size(1), dtype=torch.bool)
        mask[1] = True

        reconstructed = reconstruct_atom_graph(data, mask)
        self.assertEqual(reconstructed.selected_original_atom_indices.tolist(), [2, 3, 4])
        self.assertEqual(reconstructed.selected_attachment_atoms.tolist(), [[2], [3]])
        self.assertEqual(reconstructed.selected_attachment_node_index.tolist(), [[0], [1]])
        self.assertGreater(
            int(reconstructed.selected_attachment_atoms.max()),
            reconstructed.x.size(0) - 1,
        )
        self.assertLess(
            int(reconstructed.selected_attachment_node_index.max()),
            reconstructed.x.size(0),
        )

        batch = Batch.from_data_list([reconstructed, reconstructed])
        self.assertEqual(batch.selected_attachment_atoms.tolist(), [[2, 2], [3, 3]])
        self.assertEqual(batch.selected_attachment_node_index.tolist(), [[0, 3], [1, 4]])
        self.assertLess(int(batch.selected_attachment_node_index.max()), batch.x.size(0))

    def assert_fragment_atoms_connected(self, data, atoms):
        atoms = set(int(atom_id) for atom_id in atoms.tolist())
        if len(atoms) <= 1:
            return

        adjacency = {atom_id: set() for atom_id in atoms}
        for src, dst in data.bond_index.t().tolist():
            if src in atoms and dst in atoms:
                adjacency[src].add(dst)
                adjacency[dst].add(src)

        start = min(atoms)
        seen = {start}
        stack = [start]
        while stack:
            atom_id = stack.pop()
            for neighbor in adjacency[atom_id]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)

        self.assertEqual(seen, atoms)


if __name__ == "__main__":
    unittest.main()
