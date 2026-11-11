"""Fixed-trajectory dual-policy checkpoint selection; Assay first, no early stopping."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.experiments.plans import base_config
from src.experiments.plans import make_plan
from src.experiments.common import set_all_seeds
from src.training.pipeline import (
    Config,
    build_views_from_feature_dims,
    build_trainer,
    infer_feature_dimensions,
    strip_non_tensor_batch_fields,
    _evaluate_ensemble,
)
from src.data.dataloader import build_split_dataloaders
from src.data.preprocess import run_preprocessing
from src.models.sugar_ensemble import SuGAREnsemble
from src.training.checkpoint import save_training_checkpoint, load_checkpoint, load_view_state_dicts
from src.utils.reproducibility import deterministic_training
from src.utils.runtime_spec import runtime_spec

POLICIES = ("shared", "independent")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def verify_study_cache(cache, raw_path):
    """Check the source digest when using a cache prepared for this study."""
    marker = Path(cache) / "study_cache.json"
    if not marker.is_file():
        return
    if not raw_path:
        raise ValueError("Prepared study cache requires --raw-path")
    ready = json.loads(marker.read_text())
    actual = hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()
    if ready["raw_sha256"] != actual or ready["raw_path"] != str(Path(raw_path).resolve()):
        raise ValueError("Prepared study cache source does not match --raw-path")


@contextmanager
def preserve_rng():
    py = random.getstate()
    np_state = np.random.get_state()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


@contextmanager
def validation_execution(device):
    """Enforce deterministic CPU evaluation without changing GPU training flags."""
    enabled=torch.are_deterministic_algorithms_enabled()
    warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    if str(device)=='cpu':torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled,warn_only=warn_only)


def run_seed(base, out, seed, evaluate_test=False, policies=None, diagnostic=None):
    with deterministic_training(base.get("deterministic_training", False)):
        return _run_seed(base, out, seed, evaluate_test, policies, diagnostic)


def _run_seed(base, out, seed, evaluate_test, policies, diagnostic):
    started = time.perf_counter()
    if base.get("compare_checkpoint_rules", False) and base["epochs"] < 5:
        raise ValueError("Three-rule comparison requires at least five epochs")
    out.mkdir(parents=True, exist_ok=False)
    values = dict(
        base,
        seed=seed,
        sampler_seed=seed,
        early_stopping=False,
        selection_ma_window=1,
        selection_weight_avg=False,
        greedy_model_selection=False,
        loss_log_path=None,
        test_split=None,
        report_splits=[],
        loader_split_names=["train", "ood_val"],
    )
    policies = (
        tuple(policies)
        if policies is not None
        else (("shared",) if values.get("erm_baseline", False) else POLICIES)
    )
    if not policies or any(p not in POLICIES for p in policies):
        raise ValueError("Invalid selection policies")
    if values.get("erm_baseline", False):
        # Prediction-only remains the default. This explicit experimental
        # control preserves contrastive learning on the full atom graph.
        contrast = values.get("lambda_contrast", 0.0) if values.get("erm_contrast_control", False) else 0.0
        values.update(lambda_contrast=contrast, lambda_diversity=0.0, lambda_irm=0.0)
    cfg = Config(values)
    if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(cfg.device)
    write(out / "config.json", values)
    set_all_seeds(seed)
    loaders = build_split_dataloaders(cfg.processed_dir, cfg)
    assays = [(getattr(d, "metadata", None) or {}).get("assay_id")
              for d in loaders["ood_val"].dataset]
    strip_non_tensor_batch_fields(loaders)
    dims = infer_feature_dimensions(loaders["train"])
    views = build_views_from_feature_dims(cfg, dims)
    trainer = build_trainer(cfg, views, cfg.device)
    trainer.log_gradient_influence = bool(cfg.get("log_gradient_influence", False))
    write(out / "runtime_spec.json", runtime_spec(cfg, views, trainer, policies))
    validation_device=values.get('validation_device',cfg.device)
    if validation_device not in ('cpu',cfg.device):
        raise ValueError('validation_device must be cpu or the training device')
    evaluation_views=views
    if values.get('validation_device')=='cpu':
        if torch.get_num_threads()!=1:
            raise ValueError('Explicit CPU checkpoint selection requires one PyTorch thread')
        with preserve_rng():
            evaluation_views=build_views_from_feature_dims(Config(dict(values,device='cpu')),dims)
            cpu_loaders=build_split_dataloaders(cfg.processed_dir,Config(dict(values,
                device='cpu',loader_split_names=['ood_val'],num_workers=0,pin_memory=False)))
            strip_non_tensor_batch_fields(cpu_loaders)
            loaders['ood_val']=cpu_loaders['ood_val']
    def sync_evaluation_views():
        if evaluation_views is not views:
            for target,source in zip(evaluation_views,views):
                target.load_state_dict(source.state_dict(),strict=True)
                target.cpu().eval()
    ensembles = {
        p: SuGAREnsemble(evaluation_views, aggregation_mode=cfg.aggregation_mode, inference_policy=p)
        for p in policies
    }
    if diagnostic is not None:
        diagnostic.initialize(views, trainer, loaders["train"])
    history = []
    best = {p: dict(auc=-1.0, epoch=None) for p in policies}
    compare_rules = bool(values.get("compare_checkpoint_rules", False))
    rule_records = {p: {} for p in policies}
    for epoch in range(1, cfg.epochs + 1):
        from src.training.diversity_schedule import diversity_weight
        trainer.effective_diversity_weight = diversity_weight(
            trainer.lambda_diversity, epoch,
            trainer.diversity_delay_epochs, trainer.diversity_ramp_epochs)
        if diagnostic is None:
            loss, acc, components = trainer.train_epoch(loaders["train"], return_loss_components=True)
        else:
            loss, acc, components = trainer.train_epoch(
                loaders["train"], return_loss_components=True, trace=diagnostic.capture)
        row = dict(
            epoch=epoch,
            effective_diversity_weight=trainer.effective_diversity_weight,
            train_loss=loss,
            train_accuracy=acc,
            train_components=components,
            validation={},
        )
        if trainer.log_gradient_influence:
            row["first_batch_gradient_norms"] = trainer.last_gradient_report
        sync_evaluation_views()
        with preserve_rng(),validation_execution(validation_device):
            for policy in policies:
                saved_labels=[];saved_probabilities=[]
                evaluator=ensembles[policy]
                capture=bool(values.get('save_selected_validation_predictions',False))
                if capture:
                    class Collector:
                        def eval(self):ensembles[policy].eval()
                        def __call__(self,batch,**kwargs):
                            output=ensembles[policy](batch,**kwargs)
                            saved_labels.append(batch.y.detach().cpu().reshape(-1).numpy())
                            saved_probabilities.append(output.detach().exp().cpu().numpy()[:,1])
                            return output
                    evaluator=Collector()
                vl, va, auc = _evaluate_ensemble(
                    evaluator, loaders["ood_val"], validation_device, True
                )
                if auc is None or not math.isfinite(auc):
                    raise ValueError("Finite validation AUC required")
                row["validation"][policy] = dict(loss=vl, acc=va, auc=auc)
                if values.get("report_within_assay", False):
                    from src.evaluation.within_assay import within_assay_report
                    labels, probabilities = [], []
                    if capture:
                        labels=np.concatenate(saved_labels)
                        probabilities=np.concatenate(saved_probabilities)
                    else:
                        with torch.no_grad():
                            for batch in loaders["ood_val"]:
                                batch = batch.to(validation_device)
                                labels.extend(batch.y.reshape(-1).cpu().tolist())
                                probabilities.extend(ensembles[policy](batch).exp()[:, 1].cpu().tolist())
                    row["validation"][policy]["within_assay"] = within_assay_report(
                        labels, probabilities, assays, values.get("within_assay_min_per_class", 5))
                if compare_rules:
                    from src.training.selection_metrics import trailing_mean
                    series = [r["validation"][policy]["auc"] for r in history] + [auc]
                    choices = {"best_raw": auc, "final": auc,
                               "best_ma5": trailing_mean(series, 5)}
                    for rule, score in choices.items():
                        if score is None:
                            continue
                        prior = rule_records[policy].get(rule)
                        if rule != "final" and prior and score <= prior["selection_score"]:
                            continue
                        path = out / f"{rule}_{policy}.pt"
                        saved = Config(dict(values, inference_policy=policy,
                            checkpoint_selection_rule=rule, selection_ma_window=5 if rule == "best_ma5" else 1))
                        save_training_checkpoint(path, trainer, saved, dims, epoch=epoch,
                            history={"selected_checkpoint_epoch": epoch}, include_optimizers=False)
                        rule_records[policy][rule] = dict(epoch=epoch, selection_score=score,
                            validation_auc_at_selection=auc, checkpoint=str(path))
                if auc > best[policy]["auc"]:
                    best[policy] = dict(auc=auc, epoch=epoch)
                    saved = Config(dict(values, inference_policy=policy))
                    save_training_checkpoint(
                        out / f"best_{policy}.pt",
                        trainer,
                        saved,
                        dims,
                        epoch=epoch,
                        history={"selected_checkpoint_epoch": epoch},
                        include_optimizers=False,
                    )
                    if capture:
                        prediction_path=out/f'best_{policy}_validation_predictions.npz'
                        np.savez_compressed(prediction_path,labels=np.concatenate(saved_labels),
                            probabilities=np.concatenate(saved_probabilities),
                            assays=np.asarray([str(a) if a is not None else '' for a in assays]),
                            row_index=np.arange(sum(len(x) for x in saved_labels)))
                        write(out/f'best_{policy}_validation_reference.json',dict(epoch=epoch,
                            auc=auc,device=str(validation_device),
                            checkpoint_sha256=hashlib.sha256((out/f'best_{policy}.pt').read_bytes()).hexdigest(),
                            prediction_sha256=hashlib.sha256(prediction_path.read_bytes()).hexdigest()))
        if diagnostic is not None:
            row["reproducibility"] = diagnostic.finish_epoch(epoch)
        history.append(row)
        write(out / "history.json", history)
        write(out / "best.json", best)
        print(f"Seed {seed} epoch {epoch}/{cfg.epochs}: " + str(row["validation"]), flush=True)
    # Test data is first loaded only after all checkpoint-selection decisions for this seed.
    if evaluate_test:
        test_cfg = Config(dict(values, loader_split_names=["ood_test"]))
        test_loaders = build_split_dataloaders(cfg.processed_dir, test_cfg)
        strip_non_tensor_batch_fields(test_loaders)
        loaders.update(test_loaders)
    results = {}
    for selected_by in policies:
        load_view_state_dicts(views, load_checkpoint(out / f"best_{selected_by}.pt"))
        sync_evaluation_views()
        results[selected_by] = {}
        for policy in policies:
            results[selected_by][policy] = {}
            for split in ["ood_val"] + (["ood_test"] if evaluate_test else []):
                with validation_execution(validation_device):
                    loss, acc, auc = _evaluate_ensemble(
                        ensembles[policy], loaders[split], validation_device, True
                    )
                if auc is None or not math.isfinite(auc):
                    raise ValueError("Finite final AUC required")
                results[selected_by][policy][split] = dict(loss=loss, acc=acc, auc=auc)
    result = dict(
        seed=seed,
        best=best,
        results=results,
        layout="results[checkpoint selected by policy][evaluation policy][split]",
        note="Fixed training budget; no early stopping. Test never selects checkpoints. Ties keep earliest epoch.",
    )
    if compare_rules:
        rule_results = {}
        for policy, records in rule_records.items():
            rule_results[policy] = {}
            for rule, record in records.items():
                load_view_state_dicts(views, load_checkpoint(record["checkpoint"]))
                sync_evaluation_views()
                rule_results[policy][rule] = {}
                for split in ["ood_val"] + (["ood_test"] if evaluate_test else []):
                    with validation_execution(validation_device):
                        loss, acc, auc = _evaluate_ensemble(ensembles[policy], loaders[split], validation_device, True)
                    rule_results[policy][rule][split] = dict(loss=loss, acc=acc, auc=auc)
        result["checkpoint_rules"] = dict(records=rule_records, results=rule_results,
            moving_average_window=5, weights="endpoint, not averaged", test_selects=False)
    elapsed = time.perf_counter() - started
    memory = {}
    if str(cfg.device).startswith("cuda") and torch.cuda.is_available():
        memory.update(cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(cfg.device)),
                      cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(cfg.device)))
    if sys.platform == "linux":
        import resource
        memory["process_lifetime_peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    result["resources"] = dict(wall_seconds=elapsed, completed_epochs=cfg.epochs,
        wall_seconds_per_epoch_including_validation_and_checkpoint_io=elapsed / cfg.epochs,
        memory=memory, note="RSS is process-lifetime high water, not incremental per-run memory or worker total")
    write(out / "resources.json", result["resources"])
    write(out / "results.json", result)
    return result


def main(argv=None):
    from src.experiments.plans import workflow_defaults

    defaults = workflow_defaults()["selection"]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["fragment", "random", "erm"], default="fragment")
    p.add_argument("--scoring-mode", choices=["ordered", "symmetric"], default="ordered")
    p.add_argument("--contrast-scope", choices=["pooled", "per_view"], default="pooled")
    p.add_argument("--log-gradient-influence", action="store_true")
    p.add_argument("--compare-checkpoint-rules", action="store_true")
    p.add_argument("--report-within-assay", action="store_true")
    p.add_argument("--fragmentation", choices=["group_graph", "atom"], default="group_graph")
    p.add_argument("--selection-estimator", choices=["hard", "straight_through"], default="hard")
    p.add_argument("--selection-temperature", type=float, default=0.1)
    p.add_argument("--dataset", choices=["assay", "scaffold", "size"], default="assay")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--seeds", type=int, nargs="+", default=defaults["seeds"])
    p.add_argument("--epochs", type=int, default=defaults["epochs"])
    p.add_argument(
        "--lambda-contrast",
        type=float,
        default=0.1,
        help="Fragment contrastive coefficient; ERM always uses zero",
    )
    p.add_argument("--processed-dir")
    p.add_argument("--raw-path", help="Raw source path used in the processed-cache hash")
    p.add_argument("--output-dir")
    p.add_argument(
        "--prepare-cache", action="store_true", help="Build a fresh matching cache; do not train"
    )
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--dropout", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Final evaluation only, after selection; both checkpoints x both policies",
    )
    a = p.parse_args(argv)
    os.chdir(ROOT)
    if not math.isfinite(a.lambda_contrast) or a.lambda_contrast < 0:
        p.error("lambda-contrast must be finite and nonnegative")
    if a.epochs < 1 or not a.seeds or len(set(a.seeds)) != len(a.seeds):
        p.error("Positive epochs and unique seeds required")
    out = Path(
        a.output_dir or f"outputs/{a.dataset}/select/{datetime.now():%Y%m%d_%H%M%S_%f}"
    ).resolve()
    cache = Path(a.processed_dir or ROOT / "data/processed/ec50_followup_v1" / a.dataset).resolve()
    if a.fragmentation == "atom" and not a.processed_dir:
        cache = (ROOT / "data/processed/atom_control_v1" / a.dataset).resolve()
    base = base_config(a.dataset, cache, a.device)
    base["processed_dir"] = str(cache)
    if a.processed_dir:
        base["processed_dir"] = str(cache)
    if a.raw_path:
        base["raw_path"] = str(Path(a.raw_path).resolve())
    if a.dataset == "size":
        base.update(defaults["size_overrides"])
    for key in ("learning_rate", "dropout", "weight_decay"):
        value = getattr(a, key)
        if value is not None:
            base[key] = value
    base.update(
        epochs=a.epochs,
        inference_policy="shared",
        lambda_diversity=0.0,
        lambda_irm=0.0,
        lambda_contrast=a.lambda_contrast,
        diversity_delay_epochs=0,
        diversity_ramp_epochs=0,
    )
    base["erm_baseline"] = a.model == "erm"
    base["selection_mode"] = "random" if a.model == "random" else "learned"
    base["scoring_mode"] = a.scoring_mode
    base["contrast_scope"] = a.contrast_scope
    base["log_gradient_influence"] = a.log_gradient_influence
    base.update(compare_checkpoint_rules=a.compare_checkpoint_rules,
                report_within_assay=a.report_within_assay,
                selection_estimator=a.selection_estimator,
                selection_temperature=a.selection_temperature,
                view_fragmentations={a.fragmentation: base["num_views"]})
    base["topk_tie_policy"] = "stable_original_edge_order"
    if a.model == "erm":
        base.update(lambda_contrast=0.0, lambda_diversity=0.0, lambda_irm=0.0)
    policies = ("shared",) if a.model == "erm" else POLICIES
    seeds = a.seeds
    if a.smoke:
        if a.prepare_cache:
            p.error("--prepare-cache and --smoke cannot be combined")
        _, fixture = make_plan(out, a.device, True)
        for k in [
            "dataset",
            "raw_path",
            "raw_format",
            "splits",
            "label_key",
            "processed_dir",
            "strict",
            "batch_size",
            "num_workers",
            "pin_memory",
            "hidden_dim",
            "gnn_layers",
        ]:
            base[k] = fixture[k]
        base["epochs"] = 5 if a.compare_checkpoint_rules else 3
        seeds = [5001]
        torch.set_num_threads(1)
    if a.prepare_cache:
        if not a.processed_dir or not a.raw_path:
            p.error("--prepare-cache requires --processed-dir and --raw-path")
        if cache.exists() or cache.is_symlink():
            raise FileExistsError(f"Fresh cache required: {cache}")
        if not Path(base["raw_path"]).is_file():
            raise FileNotFoundError(base["raw_path"])
        print(
            f'Preparing {a.dataset} cache from {base["raw_path"]} at {cache}; no training',
            flush=True,
        )
        if a.dry_run:
            return
        out.mkdir(parents=True, exist_ok=False)
        run_preprocessing(Config(base))
        from src.data.dataloader import load_processed_split, _expected_metadata

        counts = {
            split: len(
                load_processed_split(cache / f"{split}.pt", _expected_metadata(Config(base)))[
                    "molecule_data"
                ]
            )
            for split in base["split_names"]
        }
        ready = dict(
            cache=str(cache),
            raw_path=base["raw_path"],
            raw_sha256=hashlib.sha256(Path(base["raw_path"]).read_bytes()).hexdigest(),
            counts=counts,
        )
        write(out / "prepared.json", ready)
        write(cache / "study_cache.json", ready)
        print(f"Cache prepared: {counts}", flush=True)
        return
    verify_study_cache(cache, base["raw_path"] if a.raw_path else None)
    print(
        f'{a.dataset}: {len(seeds)} trajectories x {base["epochs"]} epochs; model={a.model}, {len(policies)} validation policy/policies; final test={a.evaluate_test}; output={out}',
        flush=True,
    )
    if a.dry_run:
        return
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "provenance.json",
        dict(
            synthetic=a.smoke,
            python=sys.version,
            torch=torch.__version__,
            source_sha256={
                str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [
                    *ROOT.glob("*.py"),
                    *(ROOT / "src").rglob("*.py"),
                    *(ROOT / "scripts").glob("*.py"),
                ]
            },
        ),
    )
    if a.smoke:
        run_preprocessing(Config(base))
    records = [run_seed(base, out / f"seed{seed}", seed, a.evaluate_test) for seed in seeds]
    from src.experiments.common import _summarize

    summary = {
        s: {
            p: {
                split: _summarize([r["results"][s][p][split]["auc"] for r in records])
                for split in records[0]["results"][s][p]
            }
            for p in policies
        }
        for s in policies
    }
    write(
        out / "summary.json",
        dict(
            runs=records,
            summary=summary,
            layout="summary[selection policy][inference policy][split]",
        ),
    )
    print(f'COMPLETE: {out/"summary.json"}', flush=True)


if __name__ == "__main__":
    main()
