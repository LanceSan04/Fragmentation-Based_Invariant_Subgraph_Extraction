# Assay follow-up results — 2026-09-26

All four recommendations were executed as a frozen-checkpoint study. No new training or hyperparameter search was performed. The ordered pooled-contrastive model remains the declared reference. Agent/skill files were not changed.

## Main findings in plain language

- Keep pooled contrastive as our reference. It had the highest primary mean validation AUC and now the highest primary mean test AUC, but five seeds do not establish a decisive advantage.
- Turning contrastive off improved within-assay validation ranking slightly, while lowering primary overall test AUC. We should track both metrics rather than replace benchmark AUC with a newly preferred diagnostic.
- Checkpoint policy still matters: the secondary shared-selected results put all three arms near 0.77. This does not justify choosing the best test-performing policy after seeing these results.
- At edge ratio 0.5, atom retention is similar across the three size bins, around 62–65%. Connectivity differs much more: the 14 sampled molecules above 50 atoms averaged 9.75 selected components, although every original sampled molecule was connected. This is descriptive evidence of reconstruction disconnecting molecules, not proof that disconnection causes poor AUC. The large bin spans 51–335 atoms and is not representative enough to claim all large molecules behave this way.
- Views differ without diversity loss. Removing contrastive produced slightly less atom overlap (0.701) than pooled (0.710); per-view contrastive produced more (0.735). More difference is not automatically more useful.
- Increasing edge ratio from 0.3 to 0.7 increases retained atoms from 41.6% to 80.1% and overlap from 0.584 to 0.817 on the fixed evaluation sample. These ratio probes measured geometry, not AUC improvement.

## What was completed and changed

The reference configurations and all 30 checkpoint hashes were frozen before test predictions. All validation replays passed within 1e-4; the largest discrepancy was below 0.000005. The server job succeeded in 4 minutes 4 seconds, much faster than the conservative estimate. It performed zero optimizer steps and no hyperparameter search.

New reusable analysis modules: `src/experiments/assay_followup.py` and `src/experiments/retention_audit.py`, routed through `main.py experiment assay-followup`. The guarded server launcher accepts that workflow. `scripts/summarize_assay_followup.py` builds this report, and `tests/test_assay_followup.py` checks parity failures, size bins, sampling-only mode restoration and eligibility caps. No predictive architecture, loss, training config or saved checkpoint was modified in this task. All 171 unit tests passed; local and server synthetic workflow smoke checks passed. AGENTS.md and the SSH skill were verified unchanged by SHA256. Nothing was committed or pushed, and no recurring task was created.

## Recommended next study — not launched

First visually inspect the most disconnected large-molecule cases against the original molecule and confirm whether selected pieces contain meaningful chemistry. Then define a small validation-only comparison that controls actual retained atom/bond budget and measures connectivity, including a matched random-selection control. This tests whether the current connection-count budget causes a disadvantage without confounding it with simply keeping more atoms. Use a fixed seed list and a limited training budget; do not select its design by this batch's test outcomes. This batch completed the initial frozen-weight retention study, not a newly trained atom-budget architecture.

## Test ROC-AUC

Primary: independent checkpoint selection and independent inference. Secondary: shared checkpoint selection and independent inference. These are different saved epochs from the same training trajectories. All predeclared seeds were included.

| Seed | Pooled 0.1 | Off | Per-view 0.1 |
|---|---:|---:|---:|
| 8301 | 0.7705 | 0.7722 | 0.7586 |
| 8302 | 0.7700 | 0.7676 | 0.7840 |
| 8303 | 0.7758 | 0.7837 | 0.7753 |
| 8304 | 0.7567 | 0.7424 | 0.7488 |
| 8305 | 0.7902 | 0.7621 | 0.7575 |

| Arm | Primary test mean ± sample SD | Secondary test mean ± sample SD |
|---|---:|---:|
| pooled | 0.7726 ± 0.0121 | 0.7699 ± 0.0038 |
| off | 0.7656 ± 0.0152 | 0.7731 ± 0.0155 |
| per_view | 0.7648 ± 0.0144 | 0.7729 ± 0.0111 |

Paired differences compare the same seed. Positive means the alternative is higher than pooled:

- off: mean difference -0.0070; higher on 2/5 seeds. Individual differences: +0.0017, -0.0024, +0.0079, -0.0143, -0.0281.
- per_view: mean difference -0.0078; higher on 1/5 seeds. Individual differences: -0.0119, +0.0140, -0.0004, -0.0079, -0.0328.

Secondary per-seed test AUC (shared selection, independent inference):

| Seed | Pooled 0.1 | Off | Per-view 0.1 |
|---|---:|---:|---:|
| 8301 | 0.7705 | 0.7730 | 0.7833 |
| 8302 | 0.7700 | 0.7980 | 0.7840 |
| 8303 | 0.7758 | 0.7742 | 0.7686 |
| 8304 | 0.7665 | 0.7580 | 0.7713 |
| 8305 | 0.7665 | 0.7622 | 0.7575 |

## Validation and within-assay diagnostics

An assay is one experimental measurement context. Within-assay AUC compares positive and negative molecules with the same assay ID. Overall AUC also includes comparisons across assays. Macro gives each usable assay equal weight; pair-weighted gives more weight to assays with more positive-negative pairs. Single-class assays cannot supply within-assay AUC.

| Arm | Overall validation | Macro within-assay | Pair-weighted within-assay |
|---|---:|---:|---:|
| pooled | 0.8110 | 0.5362 | 0.5175 |
| off | 0.8075 | 0.5633 | 0.5306 |
| per_view | 0.8101 | 0.5539 | 0.5244 |

## What each view retains

Descriptive sample: 100 validation molecules per checkpoint, evenly spaced by original atom-count rank, using the same molecules across arms and seeds. This is not a random dataset sample. Jaccard is intersection divided by union; 1 means identical selections.

| Arm | Atoms retained | Bonds retained | Components | Atom Jaccard | Bond Jaccard |
|---|---:|---:|---:|---:|---:|
| pooled | 63.9% | 59.4% | 2.87 | 0.710 | 0.686 |
| off | 63.6% | 59.0% | 2.93 | 0.701 | 0.675 |
| per_view | 61.0% | 55.7% | 2.78 | 0.735 | 0.708 |

## Pooled-reference ratio sensitivity

Frozen weights only. Normal evaluation uses all fragment edges as candidates; training sample ratio does not filter them. Changing an inference ratio does not establish how a model trained with that ratio would perform.

| Edge ratio | Atoms retained | Bonds retained | Components | Atom overlap |
|---|---:|---:|---:|---:|
| 0.3 | 41.6% | 37.5% | 2.38 | 0.584 |
| 0.5 | 63.9% | 59.4% | 2.87 | 0.710 |
| 0.7 | 80.1% | 76.1% | 2.68 | 0.817 |

At the original edge ratio 0.5:

| Original molecule size | Atoms retained | Bonds retained | Components |
|---|---:|---:|---:|
| small_1_25 | 63.4% | 58.4% | 1.54 |
| medium_26_50 | 64.7% | 60.7% | 1.86 |
| large_51_plus | 61.7% | 56.5% | 9.75 |

Original connectivity was checked locally against the identical SHA256-verified validation cache:

- small_1_25: 31 distinct molecules, 6–25 atoms; original components mean 1.00, selected mean 1.54, selected median 1.0. Counts are repeated over five seeds and five views, not independent molecule samples.
- medium_26_50: 55 distinct molecules, 26–49 atoms; original components mean 1.00, selected mean 1.86, selected median 2.0. Counts are repeated over five seeds and five views, not independent molecule samples.
- large_51_plus: 14 distinct molecules, 51–335 atoms; original components mean 1.00, selected mean 9.75, selected median 9.0. Counts are repeated over five seeds and five views, not independent molecule samples.

Sampling-only probes use 40 size-stratified validation molecules and two fixed draws per checkpoint. Only the sampler is in training mode; dropout stays off. They are not training runs or standard evaluation predictions.

| Edge ratio | Sample ratio | Atoms retained | Bonds retained | Selected/full connections |
|---|---|---:|---:|---:|
| 0.3 | 0.5 | 46.0% | 40.5% | 27.1% |
| 0.3 | 0.7 | 44.6% | 39.9% | 27.1% |
| 0.3 | 0.9 | 42.2% | 38.3% | 27.1% |
| 0.5 | 0.5 | 71.8% | 64.4% | 47.4% |
| 0.5 | 0.7 | 70.5% | 64.2% | 47.4% |
| 0.5 | 0.9 | 66.6% | 61.9% | 47.4% |
| 0.7 | 0.5 | 71.8% | 64.4% | 47.4% |
| 0.7 | 0.7 | 84.9% | 78.6% | 64.0% |
| 0.7 | 0.9 | 82.5% | 77.6% | 64.0% |

## Interpretation limits and next decision

- No test results were used to pick checkpoints, seeds or ratios for this batch. All validation parity checks passed before test predictions. The pre-test decision is stored in decision_before_test.json.
- The assay test split was inspected during earlier project work. This is an exploratory comparison, not a new untouched holdout.
- Five seeds give a limited view of training variability. They do not quantify uncertainty across new assays/datasets.
- Per-view contrastive changes both which embeddings are compared and the scale of the loss. A performance difference cannot isolate those mechanisms.
- Retention and overlap describe geometry, not chemical usefulness or causal explanations. Sampler-only probes do not simulate retraining.
- The 40-molecule sampling probes and 100-molecule evaluation probes use different rank grids. Do not interpret their aggregate difference as a paired training/evaluation gap.
- Preserve these results. Any later retention-controlled training comparison needs a separately frozen validation protocol; do not chase the best observed test setting.

## Files

- Full local results: `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\assay\assay_followup_v1\results`
- `frozen_plan.json`: every checkpoint/config/hash and fixed comparison.
- `validation_summary.json` and `validation_*.json`: validation AUC and per-assay/prediction data.
- `retention/<arm>_<seed>/*.json`: original atom/bond IDs, counts, size bins, overlap and ratio probes.
- `decision_before_test.json`: timestamped validation-only decision.
- `summary.json` and `test_*.json`: final test AUCs and individual predictions.
- `../launch.json`, `../console.log`, `../source-manifest.json`: server execution and source provenance.
