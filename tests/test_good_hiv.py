"""Protect published split identity, label meaning, and graph/SMILES alignment."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles

from src.data.good_hiv import file_sha256, molecule_record, read_good_hiv_archive
from src.data.preprocess import load_raw_dataset


def graph(smiles, label, index):
    data = from_smiles(smiles)
    data.y = torch.tensor([[float(label)]])
    data.idx = index
    data.domain_id = torch.tensor([index // 2])
    return data


class GoodHIVTests(unittest.TestCase):
    def archive(self, root, duplicate=False, bad_label=False):
        path = root / "fixture.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for offset, name in enumerate(("train", "val", "test")):
                start = 0 if duplicate else offset * 2
                datas = [graph("CCO", 0, start), graph("CCN", 2 if bad_label else 1, start + 1)]
                stream = io.BytesIO()
                torch.save(InMemoryDataset.collate(datas), stream)
                archive.writestr(
                    f"GOODHIV/scaffold/processed/covariate_{name}.pt", stream.getvalue()
                )
        return path

    def test_official_val_and_test_mapping_and_source_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.archive(Path(temp))
            rows, meta = read_good_hiv_archive(
                path, "scaffold", ["train", "ood_val", "ood_test"], file_sha256(path)
            )
            self.assertEqual(rows["ood_val"][0]["good_source_id"], 2)
            self.assertEqual(rows["ood_test"][0]["good_source_id"], 4)
            self.assertEqual(rows["ood_val"][0]["good_official_split"], "val")
            self.assertEqual([r["cls_label"] for r in rows["train"]], [0, 1])
            self.assertIsNone(rows["ood_val"][0]["good_env_id"])
            self.assertEqual(meta["canonical_smiles_overlap"]["train:ood_val"], 2)

    def test_rejects_hash_before_deserializing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.archive(Path(temp))
            with patch("src.data.good_hiv.torch.load", side_effect=AssertionError("must not load")):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    read_good_hiv_archive(path, "scaffold", ["train"])

    def test_rejects_cross_split_source_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.archive(Path(temp), duplicate=True)
            with self.assertRaisesRegex(ValueError, "both train and ood_val"):
                read_good_hiv_archive(path, "scaffold", ["train", "ood_val"], file_sha256(path))

    def test_rejects_nonbinary_and_graph_mismatch(self):
        data = graph("CCO", 2, 1)
        with self.assertRaisesRegex(ValueError, "binary label"):
            molecule_record(data, "scaffold", "train", 0)
        data = graph("CCO", 1, 1)
        data.smiles = "CCN"
        with self.assertRaisesRegex(ValueError, "elements"):
            molecule_record(data, "scaffold", "train", 0)
        data = graph("CCO", 1, 1)
        data.edge_index = data.edge_index[:, :2]
        with self.assertRaisesRegex(ValueError, "bonds"):
            molecule_record(data, "scaffold", "train", 0)

    def test_preprocessor_requests_only_configured_splits(self):
        with patch("src.data.good_hiv.read_good_hiv_archive", return_value=({}, {})) as reader:
            load_raw_dataset(
                dict(
                    raw_path="official.zip",
                    raw_format="good_hiv_size_covariate_zip",
                    split_names=["train", "ood_val"],
                )
            )
            reader.assert_called_once_with("official.zip", "size", ["train", "ood_val"])


if __name__ == "__main__":
    unittest.main()
