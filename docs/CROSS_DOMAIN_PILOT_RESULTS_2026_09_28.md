# Cross-domain resource pilot results — 2026-09-28

Completed successfully: 12 real-data trajectories, seed 8700, one epoch each. Total server wall time: 12 minutes 40 seconds, including fresh preprocessing. No test evaluation.

| Dataset | Learned validation AUC | Random validation AUC | ERM validation AUC | Projected 15-run, 100-epoch time |
|---|---:|---:|---:|---:|
| ec50_scaffold | 0.6441 | 0.6202 | 0.6150 | 3.3 hours |
| ec50_size | 0.6525 | 0.6338 | 0.6572 | 4.1 hours |
| good_hiv_scaffold | 0.6672 | 0.6729 | 0.7180 | 14.6 hours |
| good_hiv_size | 0.5854 | 0.6255 | 0.5893 | 14.8 hours |

These are pilot scores, not scientific comparisons of converged models. Each arm had only one epoch and one seed. Do not select configurations or change the protocol from these rankings. GOOD accuracy closely matches its negative-class prevalence and is not evidence of high ranking quality.

All four datasets passed strict preprocessing with full counts preserved. All 12 validation replays passed; maximum AUC difference was 0.00001194, below tolerance 0.0001. Peak per-run PyTorch allocated CUDA memory ranged from about 95 to 357 MiB; this is not total device memory or CPU RAM.

Time projections multiply each one-epoch run duration (which includes initialization and repeated validation) by 100 epochs and five seeds. They are conservative planning estimates under similar server conditions, not measured full-training runtimes. EC50 combined projects to 7.4 hours: allow about 7–9 hours. Each GOOD study projects to roughly 15 hours. Initial preprocessing is complete and will be reused.

Recommendation: run the frozen five-seed EC50 Scaffold/Size study first (30 trajectories x 100 epochs), then GOOD-HIV Scaffold and subsequently Size after reviewing runtime and correctness. Keep all three arms and fixed settings. Full studies have not launched. Use the verified v2 release; the old v1 launcher lacked cross-domain registration.

Evidence: outputs/cross_domain/frozen_transfer_v1/resource_pilot/results/{complete,comparison,timings,validation_parity,data_manifest}.json. Server launch completed at 2026-09-28T06:48:32.293880+00:00 with exit code 0.
