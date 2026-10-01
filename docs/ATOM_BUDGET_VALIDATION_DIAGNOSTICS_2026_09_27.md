# Atom-budget validation breakdown — 2026-09-27

This is a CPU replay of all 15 **frozen independent-view checkpoints** on the SHA256-matched OOD validation cache (2,572 molecules). Each replay agreed with its server validation AUC within 0.0001. No test split was read. The server results and prespecified primary screen remain the source of the model choice; this breakdown is secondary and exploratory.

| Five-seed mean ROC-AUC | Edge reference | Atom learned | Atom random rank |
|---|---:|---:|---:|
| Overall OOD validation | 0.80654 | 0.81111 | 0.80610 |
| Macro average over usable assays | 0.55042 | 0.56791 | 0.54255 |
| Pair-weighted within-assay | 0.51526 | 0.53754 | 0.54068 |

Learned atom budgeting improved assay macro AUC by 0.01749 and within-assay pair-weighted AUC by 0.02228 versus reference. Both paired differences were positive in all five seeds. However, the primary **overall** AUC screen failed (mean +0.00457, positive 3/5 seeds). Random ranking also improved within-assay AUC (mean +0.02542), so the within-assay benefit cannot be assigned to learned ranking alone. Only 24 of 46 assays contain both classes and contribute an individual assay AUC. Within-assay positive/negative pairs are 1.068% of all pairs, which helps explain why within-assay gains and pooled AUC can move differently. These comparisons do not establish leakage or test-set benefit.

| Original atom count | Validation n (positive/negative) | Reference mean AUC | Atom learned | Atom random |
|---|---:|---:|---:|---:|
| 1–25 | 796 (652/144) | 0.74079 | 0.73822 | 0.75291 |
| 26–50 | 1,425 (1,298/127) | 0.81928 | 0.82947 | 0.80702 |
| 51–100 | 111 (106/5) | 0.73472 | 0.74000 | 0.70566 |
| 101+ | 240 (236/4) | 0.46112 | 0.33199 | 0.48919 |

The 26–50 bin has enough negatives to show a descriptive improvement with learned atom budgeting. The two bins above 50 atoms contain only 5 and 4 negatives respectively; their AUC estimates are fragile and should not guide a model choice alone. These are within-bin AUCs, not contributions added together to obtain overall AUC.

Paired saved selections are viewable at [the local gallery](../outputs/assay/atom_budget_study_v1/paired_visualizations/index.html). It aligns the same five views across all three methods for molecules 2438, 732, 705, 1288 and 1337. These examples were chosen from the existing size-stratified sample for illustration, not to demonstrate predictive success. Across the full 100 sampled validation molecules × five views × five seeds, learned and random atom-budget arms retained 65.66% and 66.31% of atoms on average, but formed 2.84 and 5.04 connected pieces. In the 10 sampled molecules above 100 atoms, the corresponding means were 12.03 and 26.36 pieces. The gallery highlights selected atoms and bonds on the original molecule; uncolored atoms are context.

Evidence: `outputs/assay/atom_budget_study_v1/validation_breakdown_v1/summary.json` and 15 per-run JSON files; saved geometry at `outputs/assay/atom_budget_study_v1/resumed_training/results/*/seed*/retention.json`. The original five-seed primary metrics are in `resumed_training/results/summary.json`.
