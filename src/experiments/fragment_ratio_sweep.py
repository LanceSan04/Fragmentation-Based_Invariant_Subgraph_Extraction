"""Bounded validation-only edge/sample-ratio sweep under independent voting."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from src.experiments import selection
from src.experiments.plans import base_config, make_plan
from train_sugar import Config


EDGE_RATIOS = (0.4, 0.5, 0.6)
SAMPLE_RATIOS = (0.7, 0.9)
BASELINE = (0.5, 0.7)
DISCOVERY_SEEDS = (8501, 8502)
CONFIRMATION_SEEDS = (8601, 8602, 8603, 8604, 8605)


def label(edge, sample):
    return f"edge{edge:.1f}_sample{sample:.1f}"


def auc(result):
    return result["results"]["independent"]["independent"]["ood_val"]["auc"]


def grid():
    return [(edge, sample) for edge in EDGE_RATIOS for sample in SAMPLE_RATIOS]


def choose_winner(scores, candidates):
    """Maximum mean discovery AUC; fixed grid order resolves exact ties."""
    means = [float(np.mean(scores[label(*candidate)])) for candidate in candidates]
    if not all(math.isfinite(value) for value in means):
        raise ValueError("Discovery must contain finite validation AUCs")
    return candidates[int(np.argmax(means))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir")
    parser.add_argument("--raw-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if not args.smoke and not (args.processed_dir and args.raw_path):
        parser.error("Real sweep requires --processed-dir and --raw-path")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    out = Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    cache = Path(args.processed_dir).resolve() if args.processed_dir else out / "processed"
    base = base_config("assay", cache, args.device)
    base.update(processed_dir=str(cache), epochs=2 if args.smoke else 100,
                learning_rate=.0003, dropout=.5, weight_decay=0., num_views=5,
                lambda_contrast=.1, lambda_diversity=0., lambda_irm=0.,
                contrast_scope="pooled", selection_mode="learned", scoring_mode="ordered",
                selection_budget="edges", ranking_mode="learned",
                inference_policy="independent", early_stopping=False,
                loader_split_names=["train", "ood_val"], test_split=None, report_splits=[],
                log_gradient_influence=True)
    if args.raw_path:
        base["raw_path"] = str(Path(args.raw_path).resolve())
    candidates = grid()
    discovery = DISCOVERY_SEEDS
    confirmation = CONFIRMATION_SEEDS
    if args.smoke:
        _, fixture = make_plan(out, args.device, True)
        for key in ("dataset", "raw_path", "raw_format", "splits", "label_key",
                    "processed_dir", "strict", "batch_size", "num_workers",
                    "pin_memory", "hidden_dim", "gnn_layers"):
            base[key] = fixture[key]
        base["split_names"] = ["train", "ood_val"]
        base["splits"] = {key: fixture["splits"][key] for key in base["split_names"]}
        candidates = [BASELINE, (0.4, 0.7)]
        discovery = (8501,)
        confirmation = (8601,)
        torch.set_num_threads(1)
    else:
        torch.set_num_threads(4)
        selection.verify_study_cache(cache, args.raw_path)
    plan = dict(dataset="EC50 Assay", seeds_discovery=discovery,
                seeds_confirmation=confirmation,
                candidates=[dict(edge_ratio=e, fragment_sample_ratio=s, label=label(e,s))
                            for e,s in candidates], baseline=label(*BASELINE),
                epochs=base["epochs"], maximum_training_runs=len(candidates)*len(discovery)+2*len(confirmation),
                inference_policy="independent", checkpoint_policy="independent",
                evaluate_test=False, synthetic=args.smoke,
                discovery_rule="Highest mean independent validation AUC; exact ties use fixed grid order",
                confirmation_screen="Mean paired validation AUC +0.005 and positive in 4/5 fresh seeds",
                note="The winner is selected from discovery validation only; confirmation uses fresh paired seeds.",
                config_base=base)
    if not args.smoke:
        plan["data_sha256"] = {name: hashlib.sha256((cache/f"{name}.pt").read_bytes()).hexdigest()
                               for name in ("train", "ood_val")}
        plan["raw_sha256"] = hashlib.sha256(Path(args.raw_path).read_bytes()).hexdigest()
    out.mkdir(parents=True)
    selection.write(out/"plan.json", plan)
    if args.smoke:
        selection.run_preprocessing(Config(base))
    scores = {label(*candidate):[] for candidate in candidates}
    for seed in discovery:
        for edge, sample in candidates:
            name = label(edge,sample)
            config = dict(base, edge_ratio=edge, fragment_sample_ratio=sample)
            result = selection.run_seed(config, out/"discovery"/name/f"seed{seed}", seed,
                                        policies=("independent",))
            scores[name].append(auc(result))
            selection.write(out/"discovery_progress.json", scores)
    winner = choose_winner(scores,candidates)
    selection.write(out/"discovery_summary.json", dict(
        scores=scores, means={k:float(np.mean(v)) for k,v in scores.items()},
        winner=label(*winner), winner_parameters=dict(edge_ratio=winner[0],fragment_sample_ratio=winner[1])))
    confirmation_results = {}
    for seed in confirmation:
        for candidate in dict.fromkeys((BASELINE,winner)):
            edge,sample = candidate
            name = label(edge,sample)
            result = selection.run_seed(dict(base,edge_ratio=edge,fragment_sample_ratio=sample),
                                        out/"confirmation"/name/f"seed{seed}", seed,
                                        policies=("independent",))
            confirmation_results.setdefault(name,[]).append(auc(result))
            selection.write(out/"confirmation_progress.json", confirmation_results)
    reference = np.asarray(confirmation_results[label(*BASELINE)])
    selected = np.asarray(confirmation_results[label(*winner)])
    paired = selected-reference
    summary = dict(baseline=label(*BASELINE), winner=label(*winner),
                   aucs=confirmation_results, paired_deltas=paired.tolist(),
                   mean_delta=float(paired.mean()), positive_seeds=int((paired>0).sum()),
                   passes_screen=bool(len(confirmation)==5 and paired.mean()>=.005 and (paired>0).sum()>=4),
                   test_evaluated=False,
                   note="If the discovery winner is the reference, confirmation trains it once per seed.")
    selection.write(out/"confirmation_summary.json",summary)
    selection.write(out/"complete.json",dict(complete=True,plan=plan))
    print("COMPLETE:",out/"confirmation_summary.json",flush=True)


if __name__ == "__main__":
    main()
