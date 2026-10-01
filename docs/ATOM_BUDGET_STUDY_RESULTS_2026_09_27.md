# Retained-atom study: completed results

All 15 trajectories are accounted for: the original 100-epoch reference seed8401 was preserved, and the remaining 14 completed successfully. Recovery runtime: 3 hours 4 minutes. All results below are OOD validation ROC-AUC, with independent checkpoint selection and independent voting. No test evaluation.

| Seed | Reference: 50% edges | Learned: 65% atoms | Random rank: 65% atoms |
|---|---:|---:|---:|
| 8401 | 0.818249 | 0.815782 | 0.801862 |
| 8402 | 0.800612 | 0.807015 | 0.805130 |
| 8403 | 0.805938 | 0.809653 | 0.803882 |
| 8404 | 0.797736 | 0.816528 | 0.796444 |
| 8405 | 0.810188 | 0.806583 | 0.823192 |
| Mean ± sample SD | 0.806545 ± 0.008115 | 0.811112 ± 0.004758 | 0.806102 ± 0.010114 |

Learned atom budgeting gained 0.004568 AUC over the reference (0.457 percentage points), with improvements in 3/5 seeds. It fails the prespecified screen of >=0.005 mean gain AND >=4/5 positive paired differences. Do not promote it as a confirmed improvement. Its smaller sample SD is encouraging but only five seeds were tested.

Learned ranking beats random ranking at the same atom target in 4/5 seeds, by 0.005010 AUC on average. This supports further investigation of learned selection; it is not proof of superior OOD test performance or statistical significance. Random ranking retains learned message gates and contrastive training, so this is not the older ungated random baseline or ERM.

| Sample geometry | Reference | Atom learned | Atom random rank |
|---|---:|---:|---:|
| Mean atoms retained | 62.93% | 65.66% | 66.31% |
| Mean bonds retained | 58.40% | 61.25% | 58.70% |
| Mean connected components | 2.83 | 2.84 | 5.04 |
| Mean selected fragment connections | 10.67 | 11.27 | 8.77 |

Geometry uses the same 100 size-stratified validation molecules, five views and five seeds (2500 records per arm, not 2500 independent molecules). No edgeless fallback cases occurred in this sample. Whole fragments make the target approximate: mean absolute deviation from target was 1.36 atoms for learned and 1.43 for random.

Learned/random mean atom retention differs by only 0.65 percentage points, below the planned 2-point warning. However, atom matching does not match bonds or connectivity: random ranking produces more disconnected pieces. Thus the observed benefit may involve connectivity rather than only chemically meaningful ranking. The reference also retains fewer atoms on average, so the learned-vs-reference contrast combines target control and a modest increase in retained content.

Recommendation: retain the current reference as primary; preserve this result as suggestive. Before another training sweep, inspect paired learned/random visualizations and decompose validation performance by assay and molecule size using these saved checkpoints. Any subsequent control intended to isolate ranking should consider bond/connectivity matching. If evaluating test later, freeze the comparison across all five seeds in advance; do not select the best seed or policy from test outcomes. No new jobs launched during this review.

Local JSON evidence: outputs/assay/atom_budget_study_v1/resumed_training/results/. Summary: summary.json; geometry: retention_summary.json and each arm/seed/retention.json. Checkpoints remain on the server except the previously downloaded reference seed8401.
