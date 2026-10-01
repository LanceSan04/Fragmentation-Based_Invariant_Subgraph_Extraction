# GOOD-HIV Size and Scaffold gating results - October 1, 2026

Both approved jobs succeeded. Size completed September 30 after 19h32m44s; the queued diagnostic completed in about 51 seconds. No new job launched during this review.

## GOOD-HIV Size: final results

15 runs, five seeds, 100 epochs each. All validation replays passed (maximum AUC difference 0.00000658). Fixed Assay settings; no GOOD-specific tuning.

| Model | Validation AUC (%) | Test AUC (%) |
|---|---:|---:|
| learned | 74.36 +/- 0.98 | 59.06 +/- 2.25 |
| random | 73.84 +/- 1.49 | 60.31 +/- 1.88 |
| erm | 74.16 +/- 1.10 | 63.71 +/- 2.44 |

| Seed | Learned test | Random test | ERM test |
|---|---:|---:|---:|
| 8701 | 60.45% | 63.56% | 66.84% |
| 8702 | 56.71% | 59.27% | 65.11% |
| 8703 | 62.25% | 58.90% | 60.39% |
| 8704 | 57.69% | 60.17% | 63.40% |
| 8705 | 58.20% | 59.64% | 62.80% |

Learned has the highest validation mean, but the lowest test mean. ERM exceeds learned by 4.65 AUC percentage points on average; random exceeds learned by 1.25 points. Learned wins only one of five test seeds against either control. This does not isolate ranking, gates, contrastive loss or cropping as a cause. GOOD Size has only 63 positive test molecules, so high accuracy is not evidence of good ranking. Seed SD is not uncertainty over new test molecules.

## Frozen GOOD-HIV Scaffold diagnostic: validation only

Same five learned checkpoints; zero optimizer steps and no test loading. All replay checks passed. Random rows below average the three separate draw AUCs within each seed, then summarize across five seeds; they are not a new voting ensemble or 15 independent seeds.

| Ranking | Gates | Mean validation AUC (%) |
|---|---|---:|
| Learned | Learned | 75.73 |
| Learned | Unit | 75.34 |
| Random | Learned | 74.32 |
| Random | Unit | 73.99 |

- Removing gates on the identical selected graph lowers mean validation AUC by 0.39 points; it lowers it on 3/5 seeds and improves it on 2/5.
- Randomizing selection while retaining learned scores as gates lowers mean AUC by 1.41 points. The learned ranking exceeds the three-draw average in all five seeds.
- The saved classifiers are adapted to their learned selected graphs. This demonstrates inference sensitivity, not a causal benefit from training a scorer versus training a random control. Random selection changes atom retention/connectivity despite matched fragment-edge counts.
- The scorer is not inert under this intervention. Neither this result nor gate removal proves which learning mechanism caused the final test outcomes.

## Decision

Freeze these results. The current learned fragment configuration did not beat matched full-graph ERM on either GOOD dataset. Consider the proposed singleton/atom granularity control as a bounded next hypothesis, not an automatic launch. Its design must explicitly account for changed retention and internal-bond gating, and compare the learned-minus-random differences across representations. No further coefficient sweep is justified by these results alone.

## Evidence

Size: `outputs/cross_domain/frozen_transfer_v1/good_hiv_size_study/results/`. Diagnostic: `outputs/good_hiv/scaffold_selection_gating_v1/results/`. Summary/launch/provenance JSONs downloaded locally; full training checkpoints and histories remain on the server.
