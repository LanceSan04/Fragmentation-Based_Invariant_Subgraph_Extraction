# Scorer study results

All ten runs completed on September 25, 2026, in 2h 33m. Five matched seeds, 100 epochs per arm, validation only. Config comparison confirmed scoring_mode was the only paired config difference. Both arms used stable top-k ties. Release manifest: e45843c27fbddff7e03ba488aa58381bc60b4285c8856ebd36908b56b78891fb.

Primary policy: independent selection / independent prediction.

| Seed | Ordered AUC | Symmetric AUC | Symmetric minus ordered |
|---|---:|---:|---:|
| 8301 | 0.8019 | 0.8126 | +0.0107 |
| 8302 | 0.8049 | 0.8064 | +0.0015 |
| 8303 | 0.8178 | 0.8097 | -0.0081 |
| 8304 | 0.8186 | 0.8035 | -0.0151 |
| 8305 | 0.8118 | 0.8075 | -0.0043 |

Ordered mean ± sample SD: 0.8110 ± 0.0075.
Symmetric mean ± sample SD: 0.8079 ± 0.0034.
Paired difference: -0.0031, sample SD 0.0098; symmetric won 2/5 seeds.

Secondary shared-selection / independent-prediction means: ordered 0.8080, symmetric 0.8067. Independent-selected validation AUC is expected to be at least as high under independent evaluation, since that same metric selects the checkpoint. This does not establish which selection policy generalizes better to test.

Conclusion: symmetry fixes numbering sensitivity but has not demonstrated better validation performance. Lower variability across these five seeds is descriptive, not proof of more reliable generalization. Keep stable tie handling; retain ordered as the current empirical reference and symmetric as an experimental option. This experiment does not isolate the effect of the tie fix because both arms include it, nor does it establish a test-AUC result.

Next recommended analysis: compare the new checkpoints' within-assay performance, actual atom/bond retention and prediction diversity. These can distinguish a change in between-assay ranking from improved within-assay discrimination. Avoid another broad search or changing the primary policy based on these data. An eventual one-time test evaluation should use a fixed comparison plan; no new server job was launched during this review.

Local results: C:/Users/Lance/Documents/GitHub/REU_Summer_2026/outputs/assay/scorer_study_v1/results/comparison.json
Server results: /home/lsantiago4/REU_Summer_2026/outputs/assay/scorer_study_v1/results/comparison.json
