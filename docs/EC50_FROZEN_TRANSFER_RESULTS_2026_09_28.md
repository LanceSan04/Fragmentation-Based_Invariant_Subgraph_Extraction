# EC50 frozen transfer study: completed results

All 30 trajectories completed: two datasets, three models, five seeds (8701-8705), 100 epochs each. Server runtime: 4 hours 25 minutes 13 seconds. Both validation and test results are available. All 30 checkpoint replays passed; maximum validation AUC difference: 1.1153867603752587e-05. Test evaluation followed completion of all training and validation checks.

## Mean ROC-AUC and sample standard deviation

| Dataset | Model | OOD validation | OOD test |
|---|---|---:|---:|
| ec50_scaffold | learned | 0.7415 +/- 0.0066 | 0.6873 +/- 0.0048 |
| ec50_scaffold | random | 0.7368 +/- 0.0065 | 0.6864 +/- 0.0035 |
| ec50_scaffold | erm | 0.7288 +/- 0.0007 | 0.6800 +/- 0.0015 |
| ec50_size | learned | 0.7464 +/- 0.0063 | 0.6202 +/- 0.0067 |
| ec50_size | random | 0.7574 +/- 0.0047 | 0.6316 +/- 0.0031 |
| ec50_size | erm | 0.7681 +/- 0.0025 | 0.6012 +/- 0.0091 |

## ec50_scaffold: each seed, test ROC-AUC

| Seed | Learned | Random | ERM |
|---|---:|---:|---:|
| 8701 | 0.6876 | 0.6866 | 0.6813 |
| 8702 | 0.6912 | 0.6804 | 0.6789 |
| 8703 | 0.6907 | 0.6889 | 0.6780 |
| 8704 | 0.6877 | 0.6882 | 0.6808 |
| 8705 | 0.6792 | 0.6877 | 0.6810 |

## ec50_size: each seed, test ROC-AUC

| Seed | Learned | Random | ERM |
|---|---:|---:|---:|
| 8701 | 0.6207 | 0.6288 | 0.5903 |
| 8702 | 0.6103 | 0.6369 | 0.6081 |
| 8703 | 0.6277 | 0.6305 | 0.5986 |
| 8704 | 0.6178 | 0.6317 | 0.6127 |
| 8705 | 0.6247 | 0.6302 | 0.5963 |

## Interpretation

- Scaffold: learned improves over ERM by 0.007263 AUC (4/5 seeds), but over random by only 0.000900 (3/5). There is no convincing learned-versus-random advantage in this sample.
- Size: learned improves over ERM by 0.019036 AUC (5/5), but loses to random by 0.011392 (5/5). The learned ranking plus message-gating mechanism is not providing a consistent benefit over this random control. This does not isolate whether ranking or gating causes the difference.
- Size validation ranks ERM > random > learned, while test ranks random > learned > ERM. High validation AUC did not identify the best test-performing model here. This observation alone does not establish leakage or a checkpoint bug.
- Random selection may provide useful regularization or avoid misleading learned selection; those are hypotheses, not conclusions established by these results. Learned/random also differ in the selected graph geometry.
- All settings were transferred from Assay without per-dataset tuning. ERM is a matched control, not a separately optimized baseline. Learned versus ERM also changes the auxiliary contrastive objective, so that comparison cannot isolate fragmentation.
- Fragment policies were independent validation selection and independent test inference. ERM used full-graph ensemble selection and evaluation. No test-driven policy or seed choice occurred in this batch. Earlier EC50 test results have been inspected, so these are exploratory results, not pristine holdouts.

## Next step

Freeze these results and retain all three models for the planned GOOD-HIV Scaffold comparison. Keep its predeclared configuration instead of redesigning it after looking at these EC50 test scores. After GOOD, a targeted ranking-versus-gating ablation is more justified than assuming learned edge selection helps. Do not extend the epoch budget just because several validation-best epochs are near 100; first inspect validation-only learning curves and predeclare any follow-up. No additional training was launched during this review.

## Evidence

Local: outputs/cross_domain/frozen_transfer_v1/ec50_study/results/comparison.json, validation_parity.json, timings.json, training_progress.json and run_audit.json. Server retains complete histories, model checkpoints and effective configurations. Source release: REU_Cross_Domain_20260928_v2, manifest b90b37d5548f9499b2111dc0b8990d7659352f8dc5d5544f1984f38df527fac9. GPU 0; sugar_run; completion 2026-09-28T12:56:37.772552+00:00.
