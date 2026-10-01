# EC50 Assay fragment-ratio sweep results — 2026-09-27

The approved sweep completed successfully in 4 hours 38 minutes on GPU 1. All 22 independent-view runs trained for 100 epochs. Histories, configs and final validation records were checked for every trajectory. The maximum difference between checkpoint-selection AUC and the restored checkpoint's validation AUC was 0.00000312. No test split was evaluated.

## Discovery: two paired seeds 8501 and 8502

| Edge ratio | Fragment sample ratio | Mean validation ROC-AUC |
|---:|---:|---:|
| 0.4 | 0.7 | 0.811504 |
| 0.4 | 0.9 | 0.808061 |
| 0.5 | 0.7 (current) | 0.809506 |
| 0.5 | 0.9 | 0.798751 |
| 0.6 | 0.7 | **0.811908** |
| 0.6 | 0.9 | 0.801575 |

The prespecified discovery rule selected edge ratio 0.6, sample ratio 0.7. Its discovery advantage over 0.5/0.7 was 0.002402 AUC. The 0.9 sampling option scored lower than 0.7 at all three edge ratios on both discovery seeds. Two seeds make this a limited finding; it was not separately confirmed.

## Fresh-seed confirmation: seeds 8601–8605

| Seed | Current 0.5 / 0.7 | Discovery winner 0.6 / 0.7 | Winner minus current |
|---:|---:|---:|---:|
| 8601 | 0.789798 | 0.784748 | -0.005049 |
| 8602 | 0.815289 | 0.810181 | -0.005109 |
| 8603 | 0.805445 | 0.816534 | +0.011089 |
| 8604 | 0.799323 | 0.796957 | -0.002366 |
| 8605 | 0.800717 | 0.804826 | +0.004109 |
| Mean ± sample SD | 0.802114 ± 0.009302 | 0.802649 ± 0.012320 | +0.000535 mean |

The winner improved only **2/5** paired seeds. It failed the prespecified confirmation screen of a mean gain >=0.005 and >=4/5 positive paired differences. Keep edge ratio 0.5 and sample ratio 0.7 as the empirical reference; do not adopt 0.6 based on the two discovery seeds or the single favorable confirmation seed. Confirmation results are validation-only and do not establish a test ROC-AUC.

## Interpretation and next step

A small discovery advantage did not reproduce strongly on fresh seeds. The run illustrates why choosing settings from a few validation runs is unstable. It does not prove 0.6 is harmful or that every ratio is equally good. The current setting remains a practical reference pending broader dataset evidence. A further Assay ratio sweep has a weaker case than fixed-protocol evaluation across other DrugOOD environments or a new mechanism-driven change. Any future test evaluation should predeclare the compared checkpoints, all seeds, and inference policy before reading test outcomes.

Server: `/home/lsantiago4/REU_Summer_2026/outputs/assay/fragment_ratio_sweep_v1/launch/`.
Local downloaded JSON: `outputs/assay/fragment_ratio_sweep_v1/launch/results/`. Discovery: `discovery_summary.json`; confirmation: `confirmation_summary.json`; per-run `config.json`, `best.json`, `history.json`, `results.json`. The source-release manifest SHA256 was `12688a8b87f289650665996fc187e66bb6452d1c9bd5f685df345a48ef4bee54`.
