# Bounded SuGAr validation search

Goal: establish a validation-tuned reference recipe before five-seed replication.
This is released-training SuGAr with the documented dependency/data adapters,
not corrected-indexing training or a fragment adaptation. The original pinned
source is unchanged. Paper: https://arxiv.org/html/2410.22228v2, Appendix A.

## Frozen scope

- EC50 Assay; raw source hash pinned in the JSON config.
- 18 predeclared combinations, sequential; 40 epochs and ten views each.
- Initialization 1 throughout. Ten views are not independent experiment repeats.
- Contrastive coefficient covers .5, 1, 2, 4, 8, 16, 32.
- Released diversity penalty covers .01, .1, 1, 2, 4, 8 (three trials each).
- Released perturbation sampling covers 85%, 90%, 95% (six trials each).
- Exact tuples/order: `configs/sugar_overnight_20261017.json`.
- Selection: highest all_mean OOD-validation AUC; earlier trial wins exact ties.
- Each view restores its own highest-validation-AUC checkpoint before ensemble
  inference, matching the completed reference path.
- all_mean = all models, mean raw scores, common hard top-k, equal probability
  voting. The evaluator also records all_max; that is not a selection objective.
- No greedy subset search, independent inference, test evaluation, early stopping,
  automatic retry, parallel trials, corrected training or automatic next study.
- A failed trial stops the batch and preserves its logs and prior completed trials.
- Estimate 6-10 hours based on the observed 20-33 minute 40-epoch runs, not a
  timeout. An overrun does not authorize stopping or expanding the approved batch.

## Important semantics and limits

These are 18 fixed combinations, not the full 126-combination factorial grid,
not adaptive Optuna, and not proof that the best possible recipe was found.
The coefficients are upstream arguments, not our native project's loss lambdas.
The released diversity term additionally multiplies `penalty` by 1e-2 and uses
its own normalization and summation; this implementation is preserved.
Sampling is the released `delete_edge` perturbation, not our fragment sampler.

Other recipe values are unchanged: 39/10 features, hidden32, three layers,
batch128, Adam .001, dropout .1, temperature1, selected edge ratio .25,
feature classifier/contrastive representation and mean classifier pooling.
The known released feature-contrastive indexing behavior remains, deliberately.
Dataset hashing verifies our exact files, not equivalence to the authors' files.
The initial recipe's test scores have already been observed; retain them and
do not describe subsequent testing as an untouched test benchmark.

The runner's default is dry-run. `--execute` is required; remote execution also
requires exact-command approval. All trial configs, checkpoint replay, predictions,
logs and validation scores are retained. The output leaderboard cannot select
using test AUC because this runner has no test-evaluation path.

## Software checks

The tiny two-trial synthetic queue completed training, checkpoint replay and
validation aggregation. Artificial scores provide no scientific evidence.
The test suite covers frozen search scope, paper-range membership, coverage,
mask semantics, indexing modes and shared-graph inference.
