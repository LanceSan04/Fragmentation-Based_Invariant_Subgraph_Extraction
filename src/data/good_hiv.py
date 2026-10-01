"""Read published GOOD-HIV covariate splits without regenerating the benchmark.

Data source: https://github.com/divelab/GOOD (MIT-licensed datasets).
No GOOD package/code is imported. Molecular descriptors are recomputed by our
existing featurizer from the original SMILES; these are not GOOD's 9/3 features.
"""

from collections import Counter
import hashlib
import io
from pathlib import Path
import zipfile

from rdkit import Chem
import torch
from torch_geometric.data.separate import separate

GOOD_HIV_URL = "https://drive.google.com/file/d/1CoOqYCuLObnG5M0D8a2P2NyL61WjbCzo/view"
GOOD_HIV_SHA256 = "767eebf62bda13a96da9507ec80391adfd2bf3f1b33aaff1f8ff3064aaaed4ee"
SPLITS = {
    "train": "train",
    "ood_val": "val",
    "ood_test": "test",
    "iid_val": "id_val",
    "iid_test": "id_test",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def molecule_record(data, domain, official_split, position):
    """Validate SMILES/graph correspondence before using SMILES for fragments."""
    label = torch.as_tensor(data.y).reshape(-1)
    if label.numel() != 1 or not torch.isfinite(label).all() or label.item() not in (0, 1):
        raise ValueError("GOOD-HIV requires one finite binary label per molecule")
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid GOOD-HIV molecule at {official_split}:{position}")
    if mol.GetNumAtoms() != data.x.size(0):
        raise ValueError("SMILES atom count differs from the official GOOD graph")
    if [atom.GetAtomicNum() for atom in mol.GetAtoms()] != data.x[:, 0].tolist():
        raise ValueError("SMILES atom ordering/elements differ from the official GOOD graph")
    expected_edges = sorted(
        (a, b)
        for bond in mol.GetBonds()
        for a, b in (
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()),
        )
    )
    if expected_edges != sorted(map(tuple, data.edge_index.t().tolist())):
        raise ValueError("SMILES bonds differ from the official GOOD graph")
    source_id = int(torch.as_tensor(data.idx).item())
    domain_id = int(torch.as_tensor(data.domain_id).item())
    env = getattr(data, "env_id", None)
    return dict(
        smiles=data.smiles,
        cls_label=int(label.item()),
        molecule_id=f"GOODHIV:{source_id}",
        domain_id=domain_id,
        good_source_id=source_id,
        good_domain=domain,
        good_official_split=official_split,
        good_split_position=position,
        good_env_id=None if env is None else int(torch.as_tensor(env).item()),
    )


def read_good_hiv_archive(path, domain, split_names, expected_sha256=GOOD_HIV_SHA256):
    """Only deserialize the pinned official archive, reading requested members.

    ``expected_sha256`` exists for generated test fixtures; production callers
    use the fixed, locally verified publisher download digest above.
    """
    if domain not in ("scaffold", "size"):
        raise ValueError("GOOD-HIV domain must be scaffold or size")
    if (
        not split_names
        or len(set(split_names)) != len(split_names)
        or set(split_names) - SPLITS.keys()
    ):
        raise ValueError("Unique supported GOOD-HIV split names required")
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ValueError(
            "GOOD-HIV archive checksum mismatch; do not regenerate or silently substitute splits"
        )
    records, source_ids, canonical, counts, member_hashes = {}, {}, {}, {}, {}
    with zipfile.ZipFile(path) as archive:
        for name in split_names:
            official = SPLITS[name]
            member = f"GOODHIV/{domain}/processed/covariate_{official}.pt"
            content = archive.read(member)
            member_hashes[member] = hashlib.sha256(content).hexdigest()
            data, slices = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
            rows = []
            for i in range(len(slices["y"]) - 1):
                graph = separate(data.__class__, data, i, slices, decrement=False)
                rows.append(molecule_record(graph, domain, official, i))
            ids = [row["good_source_id"] for row in rows]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate source indices inside GOOD-HIV {name}")
            labels = Counter(row["cls_label"] for row in rows)
            if set(labels) != {0, 1}:
                raise ValueError(f"Both binary classes required in {name}: {labels}")
            records[name], source_ids[name] = rows, set(ids)
            canonical[name] = {Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles"])) for row in rows}
            counts[name] = dict(molecules=len(rows), negative=labels[0], positive=labels[1])
    overlaps = {}
    for i, first in enumerate(split_names):
        for second in split_names[i + 1 :]:
            if source_ids[first] & source_ids[second]:
                raise ValueError(f"GOOD source index appears in both {first} and {second}")
            overlaps[f"{first}:{second}"] = len(canonical[first] & canonical[second])
    return records, dict(
        raw_format=f"good_hiv_{domain}_covariate_zip",
        source_url=GOOD_HIV_URL,
        raw_sha256=digest,
        member_sha256=member_hashes,
        counts=counts,
        source_split_names=list(SPLITS.values()),
        split_mapping=SPLITS,
        canonical_smiles_overlap=overlaps,
        feature_note="Official molecular topology checked; project 46/14 descriptors replace GOOD 9/3 descriptors. No split regeneration or row dropping.",
    )
