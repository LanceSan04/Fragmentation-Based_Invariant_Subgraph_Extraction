"""Fixed Scaffold study of trainable ranking and message-weight pathways."""
import argparse
from copy import deepcopy
from pathlib import Path
from src.experiments.cross_domain import dataset_config, verify_cache, prepare_cache
from src.experiments.selection import run_seed, write
from src.experiments.plans import make_plan
from src.experiments.checkpoints import digest

ARMS = {
    "hard_learned_gates": dict(selection_mode="learned", ranking_mode="learned",
        selection_estimator="hard", message_gate_mode="learned"),
    "st_learned_gates": dict(selection_mode="learned", ranking_mode="learned",
        selection_estimator="straight_through", message_gate_mode="learned"),
    "st_unit_gates": dict(selection_mode="learned", ranking_mode="learned",
        selection_estimator="straight_through", message_gate_mode="unit"),
    "random_learned_gates": dict(selection_mode="learned", ranking_mode="random",
        selection_estimator="hard", message_gate_mode="learned"),
    "random_unit_gates": dict(selection_mode="learned", ranking_mode="random",
        selection_estimator="hard", message_gate_mode="unit"),
}
PAIRS = {
    "st_vs_hard": ("st_learned_gates", "hard_learned_gates"),
    "ranking_with_unit_gates": ("st_unit_gates", "random_unit_gates"),
    "gating_with_random_ranking": ("random_learned_gates", "random_unit_gates"),
    "gating_with_st_ranking": ("st_learned_gates", "st_unit_gates"),
}
TRANSFER_ARMS = {name: dict(ARMS[name], erm_baseline=False) for name in
    ("hard_learned_gates", "random_learned_gates", "random_unit_gates")}
TRANSFER_ARMS["erm"] = dict(erm_baseline=True, selection_mode="learned", ranking_mode="learned",
    selection_estimator="hard", message_gate_mode="learned", lambda_contrast=0.0,
    lambda_diversity=0.0, lambda_irm=0.0)
TRANSFER_PAIRS = {
    "gating_with_random_ranking": ("random_learned_gates", "random_unit_gates"),
    "weighted_random_vs_erm": ("random_learned_gates", "erm"),
    "weighted_random_vs_hard": ("random_learned_gates", "hard_learned_gates"),
}
ATOM_RANKING_ARMS = {name: dict(TRANSFER_ARMS[name], selection_budget="atoms",
    atom_retention_ratio=0.65) for name in ("hard_learned_gates", "random_learned_gates")}
ATOM_RANKING_PAIRS = {"learned_vs_random_ranking": ("hard_learned_gates", "random_learned_gates")}
GOOD_WEIGHTING_ARMS = {name: dict(TRANSFER_ARMS[name]) for name in
                      ("random_learned_gates", "random_unit_gates", "erm")}
GATING_MECHANISM_ARMS = {
    "random_sampled_context": dict(TRANSFER_ARMS["random_learned_gates"], scorer_context="sampled"),
    "random_full_context": dict(TRANSFER_ARMS["random_learned_gates"], scorer_context="full"),
    "full_graph_gated": dict(TRANSFER_ARMS["hard_learned_gates"], scorer_context="full",
                             edge_ratio=1.0, fragment_sample_ratio=1.0, full_graph_gating=True),
    "erm": dict(TRANSFER_ARMS["erm"], scorer_context="sampled"),
    "erm_contrast": dict(TRANSFER_ARMS["erm"], scorer_context="sampled", lambda_contrast=.1,
                         erm_contrast_control=True),
}
GATING_MECHANISM_PAIRS = {
    "full_gated_vs_cropped_full_context": ("full_graph_gated", "random_full_context"),
    "scorer_context_change": ("random_full_context", "random_sampled_context"),
    "gating_vs_contrast_erm": ("full_graph_gated", "erm_contrast"),
    "contrast_in_erm": ("erm_contrast", "erm"),
}
PROFILES = {"scorer_paths": (ARMS, PAIRS), "weighting_transfer": (TRANSFER_ARMS, TRANSFER_PAIRS),
            "atom_ranking": (ATOM_RANKING_ARMS, ATOM_RANKING_PAIRS),
            "good_weighting": (GOOD_WEIGHTING_ARMS, {k: TRANSFER_PAIRS[k] for k in
                                ("gating_with_random_ranking", "weighted_random_vs_erm")}),
            "gating_mechanism": (GATING_MECHANISM_ARMS, GATING_MECHANISM_PAIRS)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--processed-dir")
    p.add_argument("--cache-root")
    p.add_argument("--profile", choices=list(PROFILES), default="scorer_paths")
    p.add_argument("--datasets", nargs="+", choices=["ec50_scaffold", "ec50_size", "good_hiv_scaffold"])
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--retention-samples", type=int, default=100)
    p.add_argument("--stage", choices=["train", "evaluate"], default="train")
    p.add_argument("--source-results")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.seeds is None:
        start = {"scorer_paths": 9001, "weighting_transfer": 9101, "atom_ranking": 9201, "good_weighting": 9301, "gating_mechanism": 9401}[a.profile]
        a.seeds = list(range(start, start + 5))
    if a.stage == "evaluate":
        from src.experiments.granularity import evaluate_frozen
        if not a.source_results or a.smoke or a.dry_run:
            p.error("Evaluation requires source results; no smoke/dry-run")
        return evaluate_frozen(a.source_results, a.output_dir, a.device)
    if a.epochs < 5 or len(set(a.seeds)) != len(a.seeds) or not a.seeds:
        p.error("At least five epochs and unique seeds required")
    datasets = a.datasets or (["ec50_scaffold", "ec50_size"] if a.profile != "scorer_paths" else ["ec50_scaffold"])
    if a.profile == "good_weighting":
        datasets = a.datasets or ["good_hiv_scaffold"]
        if datasets != ["good_hiv_scaffold"]:
            p.error("GOOD weighting profile requires only GOOD-HIV Scaffold")
    elif "good_hiv_scaffold" in datasets:
        p.error("GOOD-HIV requires the good_weighting profile")
    if (not a.processed_dir and not a.cache_root) or a.retention_samples < 1 or len(set(datasets)) != len(datasets):
        p.error("Training requires cache root or processed-dir, positive retention count, unique datasets")
    if len(datasets) > 1 and a.processed_dir:
        p.error("Multiple datasets require --cache-root")
    arms, pairs = PROFILES[a.profile]
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise FileExistsError(out)
    configs = {}
    seeds = a.seeds
    for dataset in datasets:
        cache = Path(a.processed_dir).resolve() if a.processed_dir else Path(a.cache_root).resolve() / dataset / "group_graph"
        base = dataset_config(dataset, a.raw_root, cache, a.device)
        base.update(processed_dir=str(cache), epochs=a.epochs, compare_checkpoint_rules=True,
            report_within_assay=dataset.startswith("ec50"), view_fragmentations={"group_graph": 5})
        if a.smoke:
            import torch
            torch.set_num_threads(1)
            _, fixture = make_plan(out / dataset, a.device, True)
            base.update(fixture)
            base.update(epochs=5, compare_checkpoint_rules=True, report_within_assay=False)
            seeds = [17]
        configs[dataset] = base
    plan = dict(study=a.profile, datasets=datasets, arms=list(arms),
        arm_settings=arms, seeds=seeds, epochs=base["epochs"], synthetic=a.smoke,
        primary_rule="best_raw", inference_policy="independent", test_selects=False,
        training_runs=len(datasets) * len(arms) * len(seeds), configs=configs,
        primary_comparison={"scorer_paths": "st_vs_hard", "weighting_transfer": "gating_with_random_ranking",
                            "atom_ranking": "learned_vs_random_ranking", "good_weighting": "gating_with_random_ranking",
                            "gating_mechanism": "full_gated_vs_cropped_full_context"}[a.profile],
        paired_comparisons=pairs,
        retention_samples=a.retention_samples,
        screening_rule="Primary validation mean ST-minus-hard >= 0.01 and positive in >=4/5 seeds. Report all arms and rules; not a significance test.",
        caveats="Unit weights preserve selection masks. Same connection ratio is not an equal atom budget. ST uses full masked tensors during training. This is a pathway intervention, not pure factorial isolation.")
    if a.profile in ("weighting_transfer", "good_weighting"):
        plan.update(screening_rule="Follow-up test contrast random learned minus random unit: mean >= .01 and positive in >=4/5 seeds on both splits; mixed if only one split passes; otherwise not established. No validation screen or automatic next sweep.",
            erm_matching="Five full-graph views, same architecture/initialization recipe/optimizer/budget/voting/checkpoint rules. Prediction-only ERM; fragment arms retain contrastive .1. Not an auxiliary-loss- or effective-capacity-matched causal control.",
            caveats="Adaptive follow-up after repeated benchmark test use. Equal connection ratios do not impose equal atom budgets. ERM comparison does not separate full-context information from capacity.")
    if a.profile == "good_weighting":
        plan.update(screening_rule="Random learned minus random unit test mean >= .01 and positive in >=4/5 seeds on GOOD-HIV Scaffold; otherwise not established. ERM secondary; no automatic next sweep.",
                    uncertainty_note="GOOD-HIV has no assay clusters. Report paired seed dispersion; assay bootstrap is unavailable, not zero uncertainty.")
    if a.profile == "atom_ranking":
        plan.update(screening_rule="Learned minus random ranking test mean >= .01 and positive in >=4/5 seeds on both splits; mixed if one split passes; otherwise not established. No automatic follow-up sweep.",
            caveats="Common 65% atom target, not exact size matching. Whole-fragment ranked prefixes can miss target; ranking also changes connectivity. Adaptive benchmark reuse; no claim of pure ranking isolation.")
    if a.profile == "gating_mechanism":
        plan.update(screening_rule="Report the full-minus-cropped full-context contrast on both splits. Pilot makes no success decision. For a separately approved five-seed study, mean >=.01 and >=4/5 positive seeds per split supports full-graph advantage; failure does not establish equivalence or cropping benefit.",
            caveats="Full-versus-cropped also changes eligibility sampling (1 vs .7). Context change affects both encoder inputs and internal gates from ineligible edges. ERM+contrast aligns objective coefficients, not capacity. Shared initialization and 100-epoch recipe unchanged; no Ki training or automatic extension.")
    if a.dry_run:
        print(plan)
        return plan
    out.mkdir(parents=True, exist_ok=False)
    write(out / "plan.json", plan)
    for base in configs.values():
        if a.smoke:
            prepare_cache(base)
        else:
            verify_cache(base)
    frozen, resources = [], []
    for dataset in datasets:
      for seed in seeds:
        for arm, settings in arms.items():
            base = configs[dataset]
            config = deepcopy(base)
            config.update(settings)
            record = run_seed(config, out / dataset / arm / f"seed{seed}", seed,
                              policies=("independent",))
            resources.append(dict(dataset=dataset, seed=seed, arm=arm, **record["resources"]))
            write(out / "training_resources.json", resources)
            for rule, selected in record["checkpoint_rules"]["records"]["independent"].items():
                frozen.append(dict(dataset=dataset, seed=seed, arm=arm, rule=rule,
                    checkpoint=selected["checkpoint"], checkpoint_sha256=digest(selected["checkpoint"]),
                    epoch=selected["epoch"], cache=config["processed_dir"],
                    selected_auc=selected["validation_auc_at_selection"]))
            write(out / "training_progress.json", frozen)
    write(out / "frozen_checkpoints.json", frozen)
    from src.experiments.checkpoints import restore
    from src.data.dataloader import load_processed_split, _expected_metadata
    from src.analysis.retention import measure
    retention_records = []
    for entry in frozen:
        if entry["rule"] != "best_raw":
            continue
        config, views = restore(entry["checkpoint"], entry["cache"], "cpu")
        data = load_processed_split(Path(entry["cache"]) / "ood_val.pt",
                                    _expected_metadata(config))["molecule_data"]
        if arms[entry["arm"]].get("erm_baseline", False):
            from src.analysis.retention import stratified_indices
            from src.analysis.fragment_structure import topology, describe
            indices = stratified_indices(data, a.retention_samples)
            components = [topology(data[i].x.size(0), data[i].bond_index.t().tolist())["components"] for i in indices]
            retained = dict(indices=indices, full_graph=True,
                summary=dict(atom_fraction=describe([1.0] * len(indices)),
                             bond_fraction=describe([1.0] * len(indices)), components=describe(components)),
                note="Full atom graph; no fragment rationale extraction. Edgeless bond fraction is recorded as full retention.")
        else:
            retained = measure(views, config, data, edge_ratio=config.edge_ratio,
                               count=a.retention_samples)
        path = Path(entry["checkpoint"]).with_suffix(".retention.json")
        write(path, retained)
        retention_records.append(dict(dataset=entry["dataset"], arm=entry["arm"], seed=entry["seed"],
            rule=entry["rule"], path=str(path), summary=retained["summary"]))
    write(out / "validation_retention.json", retention_records)
    write(out / "completed.json", dict(training_runs=plan["training_runs"],
        checkpoints=len(frozen), test_evaluated=False))
    return plan
