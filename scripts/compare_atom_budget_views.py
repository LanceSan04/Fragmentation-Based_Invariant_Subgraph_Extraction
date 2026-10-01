"""Render matched validation molecules from saved atom-budget selections."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from src.data.dataloader import load_processed_split


ARMS = ("reference", "atom_learned", "atom_random_rank")


def svg(molecule, atom_ids, bond_ids):
    drawer = rdMolDraw2D.MolDraw2DSVG(650, 440)
    drawer.DrawMolecule(molecule, highlightAtoms=atom_ids, highlightBonds=bond_ids)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8401)
    parser.add_argument("--indices", type=int, nargs="+", default=[2438, 732, 705, 1288, 1337])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    plan = json.loads((args.results / "plan.json").read_text())
    digest = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if digest != plan["data_sha256"]["ood_val"]:
        raise ValueError("Validation cache hash differs from frozen training plan")
    molecules = load_processed_split(args.cache)["molecule_data"]
    selected = {}
    for arm in ARMS:
        path = args.results / arm / f"seed{args.seed}" / "retention.json"
        report = json.loads(path.read_text())
        selected[arm] = {(r["index"], r["view"]): r for r in report["rows"]}
    for index in args.indices:
        for arm in ARMS:
            for view in range(5):
                if (index, view) not in selected[arm]:
                    raise ValueError(f"Index {index} view {view} was not in the saved 100-molecule sample")

    args.output_dir.mkdir(parents=True)
    pages = []
    for index in args.indices:
        source = molecules[index]
        molecule = Chem.MolFromSmiles(source.smiles)
        if molecule is None or molecule.GetNumAtoms() != source.x.size(0):
            raise ValueError(f"Molecule {index} has incompatible atom order")
        AllChem.Compute2DCoords(molecule)
        folder = args.output_dir / f"molecule_{index}"
        folder.mkdir()
        (folder / "input.svg").write_text(svg(molecule, [], []), encoding="utf-8")
        rows = []
        bonds = source.bond_index.T.tolist()
        for view in range(5):
            cells = [f"<td>View {view + 1}</td>"]
            for arm in ARMS:
                record = selected[arm][(index, view)]
                atom_ids = record["atom_ids"]
                bond_ids = []
                for bond_id in record["bond_ids"]:
                    u, v = bonds[bond_id]
                    bond = molecule.GetBondBetweenAtoms(u, v)
                    if bond is None:
                        raise ValueError("Stored bond is absent in RDKit molecule")
                    bond_ids.append(bond.GetIdx())
                name = f"view{view + 1}_{arm}.svg"
                (folder / name).write_text(svg(molecule, atom_ids, bond_ids), encoding="utf-8")
                cells.append(
                    f'<td><img src="{name}" width="470"><br>'
                    f'{len(atom_ids)}/{molecule.GetNumAtoms()} atoms; '
                    f'{len(bond_ids)}/{molecule.GetNumBonds()} bonds; '
                    f'{record["components"]} connected pieces; '
                    f'{record["selected_connections"]} fragment connections</td>'
                )
            rows.append("<tr>" + "".join(cells) + "</tr>")
        page = (
            "<html><head><meta charset='utf-8'><title>Atom budget comparison</title>"
            "<style>body{font-family:Arial;max-width:1700px;margin:auto}table{border-collapse:collapse}"
            "td,th{border:1px solid #bbb;padding:6px;vertical-align:top}</style></head><body>"
            f"<h1>Validation molecule {index}: {molecule.GetNumAtoms()} atoms</h1>"
            f"<p>{html.escape(source.smiles)}</p><img src='input.svg'>"
            "<p>Colored atoms and bonds are the saved model inputs. Uncolored atoms show the "
            "original molecule for context. The columns align the same seed, molecule and "
            "view across methods. Examples are illustrative, not a performance sample.</p>"
            "<table><tr><th>View</th><th>50% edge reference</th>"
            "<th>65% atom learned</th><th>65% atom random rank</th></tr>"
            + "".join(rows) + "</table></body></html>"
        )
        (folder / "index.html").write_text(page, encoding="utf-8")
        pages.append(f'<li><a href="molecule_{index}/index.html">Molecule {index}: '
                     f'{molecule.GetNumAtoms()} atoms</a></li>')
    (args.output_dir / "index.html").write_text(
        "<html><body><h1>Paired atom-budget views</h1><p>Saved validation selections "
        f"from seed {args.seed}; no model rerun.</p><ul>" + "".join(pages) + "</ul></body></html>",
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed, "indices": args.indices, "arms": ARMS,
        "validation_cache_sha256": digest,
        "source": str(args.results.resolve()),
        "selection": "Saved geometry only, independent of validation labels and scores",
    }, indent=2), encoding="utf-8")
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
