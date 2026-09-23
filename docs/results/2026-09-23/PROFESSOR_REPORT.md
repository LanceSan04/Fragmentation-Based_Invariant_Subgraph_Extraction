# REU research update — September 23, 2026

We study molecular property prediction under distribution shift using five fragment-based graph neural network views. Each view scores fragment connections, selects a molecular subgraph and predicts a molecule-level label. ERM is a separately trained full-molecule control with prediction loss only.

## Main results

Test ROC-AUC ×100, mean ± sample SD over seeds 7001–7010, 100 epochs. Both policies select checkpoints using validation only. S/I means shared validation selection followed by independent inference.

| Dataset / model | Policy | Test AUC |
|---|---|---:|
| Assay, full-molecule ERM | Full graph | 75.61 ± 3.35 |
| Assay, fragments, contrastive λ=0.1 | S/S | 76.58 ± 1.50 |
| Assay, fragments, contrastive λ=0.1 | S/I | 76.83 ± 1.55 |
| Size, full-molecule ERM | Full graph | 59.82 ± 1.64 |
| Size, fragments, contrastive λ=0.1 | S/S | 62.65 ± 1.35 |
| Size, fragments, contrastive λ=0.1 | I/I | 64.11 ± 1.02 |

Size I/I beats matched ERM in all ten seeds. On Size, λ=0.3 yielded 64.50 ±0.79 test AUC, but λ=0.01 had the highest independent-validation mean. We therefore treat 0.3 as exploratory, not a validation-selected winner. ERM has not been independently tuned.

## Are the views different?

Yes. Across 60 Assay checkpoint diagnostics (2,572 validation molecules each), independent voting beat the best individual view in 50/60 evaluations. These evaluations share seeds/data and are not independent statistical replicates.

For λ=0.1/shared-selected checkpoints, average pairwise selected-edge Jaccard overlap was 60.0%, prediction correlation 0.785, and class disagreement 13.4%. With λ=0, overlap was 58.0%: views did not collapse even without contrastive or diversity losses. Jaccard means intersection divided by union of selected edge sets.

## Molecule visualizations

Open [normal inference](visualizations/evaluation/index.html) or [sampling demonstration](visualizations/sampling/index.html). Each includes five validation molecules, the input molecule, each view's eligible connections and selected atom/bond highlights, probabilities and machine-readable traces.

Normal inference does not randomly drop fragment edges. The sampling demonstration enables only the sampler while freezing dropout/BatchNorm; it is not a full training-step recording. Unhighlighted molecular structure is context. The first five molecules are illustrative, not representative; their predictions are all strongly positive. Selections are not proof of causal chemistry.

## Direction

1. Freeze this evidence and code before changing architecture.
2. Tune a full-molecule baseline with a bounded validation-only budget.
3. Compare learned selection with random selection at the same graph budget.
4. Evaluate a protocol fixed in advance on Scaffold before expanding architectural searches.

All 100 recent runs are retained in [per-seed tables](PER_SEED_RESULTS.md) and [machine-readable results](results.json). Test results have been inspected repeatedly during development; these are exploratory comparisons, not an untouched-test confirmation.
