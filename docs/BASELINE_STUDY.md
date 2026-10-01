# Assay baseline and random-selection study

Status: frozen validation study and approved final test evaluation completed on 2026-09-24. All three five-seed arms and the four-condition ERM tuning finished. All 15 checkpoint validation replays passed before test evaluation. Test ROC-AUC mean ± sample SD: learned 0.7608 ± 0.0148; random 0.7527 ± 0.0119; ERM 0.7680 ± 0.0201. The final record is `outputs/assay/baseline_study_v1/final_comparison/results/comparison.json` on the server; a local copy is under `outputs/assay/audit_20260924/source_records/final_comparison/`. See `docs/AUDIT_2026_09_24.md` for the subsequent diagnostic audit.

## Directory convention

New Assay records belong under `REU_Summer_2026/outputs/assay/baseline_study_v1/` with one named directory per pipeline (`cache_preparation`, `resource_pilot`, `erm_tuning`, and later `final_erm`, `final_learned`, `final_random`). Each launcher record contains `launch.json`, `console.log`, `source-manifest.json`, and `results/`. The fresh reusable cache belongs at `REU_Summer_2026/data/processed/baseline_study_v1/assay/`. This matches the local cleanup rule of one `outputs/` root and separate `data/processed/` caches. Verified source snapshots remain at `~/reu_releases/`.

The two former `~/reu_runs/` records were moved with file-hash verification into this project's `outputs/` tree; the empty `~/reu_runs/` directory was removed. Older root-level `outputs_*` entries were also migrated. See `docs/PATH_MIGRATION.md` for recovery maps and destinations.

## Questions and controls

- **Stronger ERM:** Can a full-molecule, five-classifier ensemble improve when its learning rate, dropout, and weight decay are selected for ERM rather than inherited from the fragment model?
- **Random fragments:** Does the learned fragment mechanism improve OOD validation and test AUC over a control that selects the same number of fragment connections at random?

The random control uses the same five-view classifier architecture, fragment candidate sampler, edge ratio, optimizer, contrastive coefficient, seeds, data, and epoch budget as the learned arm. It replaces learned top-k ranking with fresh random ranks during training and stable molecule-specific random ranks during validation/test. Selected bonds have unit message weights; the learned fragment scorer is not used. Thus this control tests the **combined contribution of learned selection and learned soft gating**, not the ranking step in isolation. A second control retaining learned gates while randomizing only top-k would be needed to separate those effects.

## Frozen decision rule

1. Prepare one fresh EC50-Assay cache with `train`, `ood_val`, and `ood_test` from the existing DrugOOD JSON. Record the raw SHA256 and reject caches with mismatched preprocessing metadata. Test is stored but not loaded during validation tuning.
2. Tune ERM on four predeclared configurations in `src/experiments/erm_tune.py`: two seeds (8201–8202), up to 60 epochs each, no early stopping. Select the highest mean per-seed best `ood_val` ROC-AUC. Ties keep the first grid entry. All five ERM classifiers train on the full molecule with prediction loss only.
3. Run five paired seeds (8101–8105) for each final arm, 100 epochs per seed, without early stopping. The arms are tuned ERM, learned fragments, and random fragments. Fragment arms use contrastive coefficient 0.1, diversity 0, IRM 0. ERM uses all three auxiliary coefficients 0. Both fragment arms keep the same edge ratio (0.5), fragment sample ratio (0.7), architecture, and other settings.
4. Select checkpoints using `ood_val` only. Primary fragment result uses independently selected checkpoint and independent inference; ERM uses shared/full-graph selection and inference. Other fragment policy combinations are secondary. Load `ood_test` only after all checkpoint choices and ERM settings are fixed. Summarize per-seed ROC-AUC, sample standard deviation, and paired differences with `scripts/summarize_baseline_study.py`.

The Assay test split has already been inspected during earlier research, so this is an exploratory follow-up rather than a pristine confirmation. Do not choose a winner by looking for favorable test seeds. Stop conditions are a failed data-provenance check, missing/nonfinite validation AUC, a training error, or a user instruction to stop; the workflow does not automatically retry or expand its budget.

The saved selection checkpoints retain a train/validation-only loader list. The generic `main.py eval` path does not override that list to load `ood_test`, so use the dedicated `main.py experiment baseline-eval` workflow for the final comparison. It checks the three arms' matching seeds, frozen policies, raw-data hash, selected ERM settings, and scientific source hashes. It reloads every checkpoint and checks validation AUC within 1e-4 **before loading any test split**. If a parity check fails, the test phase does not start. After all 15 checks pass, it evaluates the frozen checkpoints once on `ood_test` and writes per-seed results, sample standard deviations, and paired differences under one fresh output directory. No training or checkpoint reselection occurs in this workflow.

## Deployment boundary

All remote pipelines require an exact-command approval under `docs/REMOTE_WORKING_AGREEMENT.md`. They run one at a time in the existing `sugar_run` tmux session. A verified source release is uploaded to a new directory; data remains in the existing project tree. The first pilot on 2026-09-23 exited before training because the older `controls_v1` cache had a different preprocessing config hash. The fresh Assay cache contains 4,540 train, 2,572 OOD validation, and 2,490 OOD test molecules. The approved seed-8101 two-epoch replacement pilot completed in about 38 seconds on GPU 1; epoch-2 OOD-validation AUC was 0.7513 shared and 0.7633 independent. No test evaluation ran. The longer ERM tuning and five-seed arms still need separate approval and resource estimates.
