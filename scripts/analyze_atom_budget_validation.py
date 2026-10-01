"""Replay frozen Assay validation checkpoints and break AUC down by group."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.data.dataloader import load_processed_split
from src.experiments.assay_followup import evaluate


ARMS = ("reference", "atom_learned", "atom_random_rank")
SEEDS = tuple(range(8401, 8406))


def size_bin(atoms):
    if atoms <= 25:
        return "1-25"
    if atoms <= 50:
        return "26-50"
    if atoms <= 100:
        return "51-100"
    return "101+"


def auc_report(rows):
    labels = [r["label"] for r in rows]
    pos = sum(labels)
    neg = len(labels) - pos
    return dict(n=len(rows), positive=pos, negative=neg,
                auc=float(roc_auc_score(labels, [r["probability"] for r in rows]))
                if pos and neg else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    plan = json.loads((args.results / "plan.json").read_text())
    digest = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if digest != plan["data_sha256"]["ood_val"]:
        raise ValueError("Frozen validation cache hash mismatch")
    molecules = load_processed_split(args.cache)["molecule_data"]
    sizes = [m.x.size(0) for m in molecules]
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True)
    data = {}
    for seed in SEEDS:
        for arm in ARMS:
            folder = args.results / arm / f"seed{seed}"
            checkpoint = folder / "best_independent.pt"
            saved = json.loads((folder / "results.json").read_text())
            if set(saved["results"]["independent"]["independent"]) != {"ood_val"}:
                raise ValueError("Unexpected test results in study")
            report = evaluate(checkpoint, args.cache.parent, "cpu", "ood_val", seed)
            rows = report["molecules"]
            if len(rows) != len(sizes) or [r["index"] for r in rows] != list(range(len(sizes))):
                raise ValueError("Validation replay rows misaligned with original molecules")
            reported_auc = saved["results"]["independent"]["independent"]["ood_val"]["auc"]
            if abs(report["pooled_auc"] - reported_auc) > .0001:
                raise ValueError(f"CPU replay differs materially for {arm}, seed {seed}")
            for row in rows:
                row["atom_count"] = sizes[row["index"]]
                row["size_bin"] = size_bin(row["atom_count"])
            size_reports = {}
            for name in ("1-25", "26-50", "51-100", "101+"):
                size_reports[name] = auc_report([r for r in rows if r["size_bin"] == name])
            reduced = dict(arm=arm, seed=seed, checkpoint_sha256=report["checkpoint_sha256"],
                           original_validation_auc=reported_auc,
                           replay_validation_auc=report["pooled_auc"],
                           macro_assay_auc=report["macro_assay_auc"],
                           within_pair_weighted_auc=report["within_pair_weighted_auc"],
                           within_pair_fraction=report["within_pair_fraction"],
                           size=size_reports, assays=report["assays"], molecules=rows)
            (args.output_dir / f"{arm}_seed{seed}.json").write_text(json.dumps(reduced, indent=2))
            data[(arm, seed)] = reduced
            print(f"{arm} seed{seed}: replay {report['pooled_auc']:.6f}; "
                  f"macro assay {report['macro_assay_auc']:.6f}", flush=True)
    summary = {}
    for arm in ARMS:
        entries = [data[(arm, seed)] for seed in SEEDS]
        summary[arm] = dict(
            pooled_auc_mean=float(np.mean([r["replay_validation_auc"] for r in entries])),
            macro_assay_auc_mean=float(np.mean([r["macro_assay_auc"] for r in entries])),
            within_pair_weighted_auc_mean=float(np.mean([r["within_pair_weighted_auc"] for r in entries])),
            sizes={name:dict(n=entries[0]["size"][name]["n"],
                             positive=entries[0]["size"][name]["positive"],
                             negative=entries[0]["size"][name]["negative"],
                             auc_mean=float(np.mean([r["size"][name]["auc"] for r in entries]))
                             if all(r["size"][name]["auc"] is not None for r in entries) else None)
                   for name in ("1-25", "26-50", "51-100", "101+")},
        )
    for arm in ARMS[1:]:
        summary[arm]["paired_macro_assay_delta_vs_reference"] = [
            data[(arm, seed)]["macro_assay_auc"] - data[("reference", seed)]["macro_assay_auc"]
            for seed in SEEDS]
        summary[arm]["paired_within_assay_delta_vs_reference"] = [
            data[(arm, seed)]["within_pair_weighted_auc"] - data[("reference", seed)]["within_pair_weighted_auc"]
            for seed in SEEDS]
    (args.output_dir / "summary.json").write_text(json.dumps(dict(
        validation_cache_sha256=digest, seeds=SEEDS, arms=summary,
        note="Frozen independent checkpoints, CPU validation replay only. No test labels used. "
             "AUC within each size bin compares labels only inside that bin; class imbalance matters."
    ), indent=2))
    print(args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
