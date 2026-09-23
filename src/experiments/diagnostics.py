"""Bounded train/validation diagnostics. Never loads the test split or saves models."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations, islice
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from train_sugar import (Config, build_split_dataloaders, build_trainer,
    build_views_from_feature_dims, infer_feature_dimensions, seed_everything,
    strip_non_tensor_batch_fields)
from src.data.preprocess import run_preprocessing
from src.training.sugar_trainer import SuGARTrainer
from src.experiments.plans import make_plan


def gradient_vector(loss, parameters):
    grads = (torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
             if loss.requires_grad else [None] * len(parameters))
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                      for p, g in zip(parameters, grads)])


def summarize_gradients(vectors):
    norms = {k: float(v.norm()) for k, v in vectors.items()}
    cosines = {}
    for a, b in combinations(vectors, 2):
        denominator = norms[a] * norms[b]
        cosines[a + "__" + b] = (float(torch.dot(vectors[a], vectors[b])) / denominator
                                    if denominator > 1e-20 else None)
    return dict(norms=norms, cosines=cosines)


def domain_report(loader, cfg):
    groups = defaultdict(list)
    for data in loader.dataset:
        value = SuGARTrainer._extract_env_value(data, cfg.env_split_key)
        if value is None:
            raise ValueError("Missing domain metadata: refusing random environment fallback")
        groups[value].append((int(data.y.reshape(-1)[0]), int(data.x.shape[0])))
    values = list(groups)
    mapping = (SuGARTrainer._assign_env_buckets_by_size(loader.dataset,
               [SuGARTrainer._extract_env_value(d, cfg.env_split_key) for d in loader.dataset], cfg.num_envs)
               if cfg.env_bucket_strategy == "size" else SuGARTrainer._assign_env_buckets(values, cfg.num_envs))
    def describe(items):
        sizes = [x[1] for x in items]
        return dict(n=len(items), labels=dict(Counter(x[0] for x in items)),
                    atoms_min=min(sizes), atoms_max=max(sizes), atoms_mean=sum(sizes)/len(sizes))
    buckets = defaultdict(list)
    for key, items in groups.items():
        buckets[mapping[key]].extend(items)
    return dict(domains=[dict(domain=str(k), bucket=mapping[k], **describe(v)) for k,v in groups.items()],
                buckets={str(k): describe(v) for k,v in buckets.items()})


def probe(trainer, batch):
    captured = []
    handles = [v.atom_classifier.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
               for v in trainer.views]
    try:
        outputs = trainer._forward_views(batch, True)
    finally:
        for handle in handles:
            handle.remove()
    total, components, _ = trainer._compute_components(outputs, trainer._labels_from_batch(batch), batch.env_id)
    terms = {k: components[k] for k in ("prediction_loss", "contrastive_loss_weighted",
                                      "diversity_loss_weighted", "irm_loss_weighted")}
    report = dict(losses={k: float(v.detach()) for k,v in components.items()}, gradients={}, graphs=[],
                  environments=dict(Counter(batch.env_id.detach().cpu().tolist())))
    for name, attr in [("scorer", "fragment_masker"), ("predictor", "atom_classifier")]:
        params = [p for v in trainer.views for p in getattr(v, attr).parameters() if p.requires_grad]
        vectors = {k: gradient_vector(loss, params) for k,loss in terms.items()}
        report["gradients"][name] = summarize_gradients(vectors)
    for view, data, pv in zip(trainer.views, captured, outputs):
        edges = data.edge_index
        assert data.edge_attr.shape[0] == edges.shape[1]
        assert data.edge_weight.numel() == edges.shape[1]
        assert torch.isfinite(data.x).all() and torch.isfinite(data.edge_attr).all()
        if edges.numel():
            assert int(edges.min()) >= 0 and int(edges.max()) < data.x.shape[0]
            assert torch.equal(data.batch[edges[0]], data.batch[edges[1]])
        assert pv["predictions"].shape[0] == batch.num_graphs
        score_gradient = torch.autograd.grad(components["prediction_loss"], pv["fragment_scores"],
                                             retain_graph=True, allow_unused=True)[0]
        graph = dict(atoms=int(data.x.shape[0]), directed_edges=int(edges.shape[1]),
                     input_atoms=int(batch.x.shape[0]), atom_features=int(data.x.shape[1]),
                     bond_features=int(data.edge_attr.shape[1]),
                     eligible_fragment_edges=int(pv["fragment_edge_eligible"].sum()),
                     fragment_edges=int(pv["fragment_scores"].numel()),
                     score_prediction_gradient_norm=None if score_gradient is None else float(score_gradient.norm()))
        # Controlled predictor interventions: identical reconstructed graph; eval mode avoids dropout/BN changes.
        was_training = view.atom_classifier.training
        view.atom_classifier.eval()
        with torch.no_grad():
            original = view.atom_classifier(data)
            changed = data.clone()
            changed.edge_weight = torch.zeros_like(data.edge_weight)
            graph["zero_gate_max_logit_change"] = float((original-view.atom_classifier(changed)).abs().max())
            changed = data.clone()
            changed.edge_attr = torch.zeros_like(data.edge_attr)
            graph["zero_bond_features_max_logit_change"] = float((original-view.atom_classifier(changed)).abs().max())
        view.atom_classifier.train(was_training)
        report["graphs"].append(graph)
    assert all(torch.isfinite(v) for v in components.values())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT/"configs/preliminary_contrastive_test.yaml"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    if not 1 <= args.batches <= 10 or not 0 <= args.warmup_steps <= 50:
        parser.error("Use 1..10 probe batches and 0..50 warmup steps")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    output = Path(args.output_dir or ROOT/f"outputs/analysis/diagnostics/{datetime.now():%Y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="reu_diagnostic_") as temp:
        values = make_plan(Path(temp), args.device, smoke=True)[1] if args.smoke else yaml.safe_load(Path(args.config).read_text())
        # Explicit diagnostic probes, not a recommended training recipe.
        values.update(device=args.device, lambda_contrast=.1, lambda_diversity=.5, lambda_irm=1.,
                      num_envs=2, num_workers=0, seed=301, sampler_seed=301, loss_log_path=None,
                      test_split=None, report_splits=[], log_test_curve=False, output_root="")
        values["loader_split_names"] = [values["train_split"], values["validation_split"]]
        if args.smoke:
            values["splits"].pop("ood_test", None)
            values["split_names"] = values["loader_split_names"]
            torch.set_num_threads(1)
            run_preprocessing(Config(values))
        cfg = Config(values)
        seed_everything(301)
        loaders = build_split_dataloaders(cfg.processed_dir, cfg)
        domains = {name: domain_report(loader, cfg) for name,loader in loaders.items()}
        train = loaders[cfg.train_split]
        dims = infer_feature_dimensions(train)
        trainer = build_trainer(cfg, build_views_from_feature_dims(cfg,dims), args.device)
        trainer.prepare_irm_environments(list(loaders.values()))
        strip_non_tensor_batch_fields(loaders)
        coverage = {}
        for name, loader in loaders.items():
            counts = Counter()
            for batch in loader:
                counts[int(torch.unique(batch.env_id).numel())] += 1
            coverage[name] = dict(batches_by_populated_environments=dict(counts),
                                 batches_eligible_for_irm=sum(v for k,v in counts.items() if k >= 2),
                                 note="One loader pass; shuffle composition varies on later epochs")
        for view in trainer.views:
            view.train()
        # Fixed bounded batches reused before and after warmup; no test batches are loaded.
        batches = list(islice(train, args.batches))
        records = []
        for stage in ("initialization", "after_warmup"):
            if stage == "after_warmup":
                for step in range(args.warmup_steps):
                    batch = batches[step % len(batches)].clone().to(args.device)
                    for opt in trainer.optimizers: opt.zero_grad()
                    pv = trainer._forward_views(batch, True)
                    loss,_,_ = trainer._compute_components(pv, trainer._labels_from_batch(batch), batch.env_id)
                    loss.backward()
                    for view in trainer.views: torch.nn.utils.clip_grad_norm_(view.parameters(),1.)
                    for opt in trainer.optimizers: opt.step()
            for index,batch in enumerate(batches):
                print(f"Measuring {stage} batch {index+1}/{len(batches)}", flush=True)
                records.append(dict(stage=stage, batch=index, **probe(trainer,batch.clone().to(args.device))))
        report = dict(config=values, synthetic=args.smoke, domains=domains, batch_coverage=coverage, probes=records,
                      warmup_steps=args.warmup_steps, feature_dimensions=dims,
                      caveat="Diagnostic joint-loss trajectory; pre-clipping gradients. Probe forwards update BN statistics and consume RNG. Not a benchmark or isolated causal experiment.")
        (output/"diagnostics.json").write_text(json.dumps(report,indent=2,default=str))
    print(f"PASS: {output/'diagnostics.json'}")


if __name__ == "__main__":
    main()
