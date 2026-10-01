# GOOD-HIV Scaffold results - September 29, 2026

Completed successfully: 15 runs, seeds 8701-8705, 100 epochs each. Wall time 9h04m57s. All 15 restored-checkpoint validation checks passed; largest absolute AUC difference 0.00001095. Test evaluation completed.

## Mean ROC-AUC (percent) +/- sample SD

| Model | OOD validation | OOD test |
|---|---:|---:|
| learned | 75.73 +/- 0.99 | 67.51 +/- 3.06 |
| random | 74.73 +/- 0.39 | 68.41 +/- 1.78 |
| erm | 77.66 +/- 0.57 | 70.60 +/- 1.35 |

## Each seed: test ROC-AUC (percent)

| Seed | Learned | Random | ERM |
|---|---:|---:|---:|
| 8701 | 69.06 | 67.78 | 69.86 |
| 8702 | 70.18 | 70.86 | 71.77 |
| 8703 | 67.96 | 69.31 | 72.08 |
| 8704 | 62.27 | 66.10 | 70.45 |
| 8705 | 68.08 | 67.97 | 68.83 |

## Interpretation

- ERM beats learned selection on all five test seeds, by 3.09 AUC percentage points on average. ERM also beats random on all five test seeds.
- Learned beats random on all five validation seeds, but only two of five test seeds. Its mean test score is 0.90 points lower. This is a validation-to-test ranking reversal, not evidence by itself of leakage or a checkpoint bug.
- Seed 8704 is particularly weak for learned selection (62.27%). Retain it; it is part of the predeclared seed set.
- These fixed Assay hyperparameters were transferred without GOOD-specific tuning. This is evidence against a consistent benefit of the current fragment method under this protocol, not proof that every fragment architecture will fail.
- Learned/random differ in both ranking and message gating; ERM additionally has no contrastive objective. The comparison cannot isolate a single cause.
- There are only 81 positive test molecules among 4108. About 98% accuracy is attainable by predicting all negatives, so ROC-AUC is the primary metric. Seed SD does not measure uncertainty from sampling test molecules.
- No published SuGAr-on-GOOD comparison was made. GOOD does not establish whether DrugOOD files exactly reproduce SuGAr.

## Next steps

Freeze this result. Retain the previously specified protocol for GOOD-HIV Size, with separate launch approval; do not modify it to chase Scaffold test outcomes. Before a larger search, prioritize the validation-only ranking-versus-gating diagnostic and controlled training design. No new training was launched during this review.

## Evidence

Local JSON records: `outputs/cross_domain/frozen_transfer_v1/good_hiv_scaffold_study/results/`. Full checkpoints and histories remain on the server in the same project-relative directory.
Source release: `REU_Cross_Domain_20260928_v2`; manifest `b90b37d5548f9499b2111dc0b8990d7659352f8dc5d5544f1984f38df527fac9`. GPU 1, existing sugar_run session. Finished 2026-09-29 11:50:34 UTC.
