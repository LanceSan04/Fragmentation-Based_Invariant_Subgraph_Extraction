"""Inspect pinned upstream without importing or executing its entry points."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def inspect():
    provenance = json.loads((ROOT / "provenance.json").read_text())
    changed = [name for name, expected in provenance["files"].items()
               if not (ROOT / "upstream" / name).is_file()
               or hashlib.sha256((ROOT / "upstream" / name).read_bytes()).hexdigest() != expected]
    dependencies = {name: importlib.util.find_spec(name) is not None
                    for name in ("torch", "torch_geometric", "ogb", "mmcv", "dgl", "torchcontrib")}
    return dict(commit=provenance["commit"], upstream_integrity_passed=not changed,
                changed_files=changed, dependency_presence=dependencies, ready_for_training=False,
                blockers=[
                    "Pinned checkout provides no DrugOOD config .py files; external DrugOOD source/config is needed.",
                    "main.py passes seed= to DrugOOD.__init__, whose pinned signature has no seed argument.",
                    "ens.py DrugOOD path references Config/build_dataset while their imports are commented out.",
                    "Upstream masks use legacy __explain__/__edge_mask__; verify gating behavior under the chosen PyG version.",
                    "DrugOOD dimensions are hardcoded to atom 39/bond 10; our cached atom 46/bond 14 are not interchangeable.",
                    "No top-level license found; confirm permission before publishing vendored upstream source.",
                ], note="Dependency presence is not version compatibility. No training or dataset loading performed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = inspect()
    print(json.dumps(report, indent=2))
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["upstream_integrity_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
