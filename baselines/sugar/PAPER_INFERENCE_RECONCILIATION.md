# SuGAr ENS: paper, release, and reference evaluator

Date: October 17, 2026. Scope: ensemble inference (ENS), not weight averaging (WA).
Paper: https://arxiv.org/html/2410.22228v2, Appendix A, Algorithms 1 and 3.
Released revision: `b2597710e56cd15370d86794a6ea3e1a61c4bb6a`.

## Primary decision

Use the existing **released-training** checkpoints. Do not retrain, apply the
indexing correction, or compare independent graphs for this reproduction check.
Primary inference is all ten models, mean raw edge scores, common hard top-k,
and the equal-weight average of positive-class probabilities. For ROC-AUC use
the probability score, not the thresholded class decision.

The earlier claim that equal-weight probability voting was merely our invention
was too cautious: Algorithm 3 explicitly specifies equal-weight soft voting.
Independent per-view subgraphs remain a separate project experiment.

| Component | Paper | Released code | Evaluator decision |
|---|---|---|---|
| Edge aggregation | Average or maximum (Algorithm 3) | `ens.py::sub_avg`, `sub_max` | Primary average; report maximum separately |
| Scores supplied to aggregation | Predicted edge weights | `return_data='attention'` returns logits modified by `split_graph` | Read raw scorer output before selection; documented paper-oriented correction |
| Shared subgraph | Top-k of aggregate, used by every classifier | Same tensor passed repeatedly into a mutating selection function | Clone the aggregate for each classifier, preserving equal input |
| Probability vote | Equal weights (Algorithm 3, line 20) | ROC-AUC `soft_vote` weights by AUC on evaluated labels | Equal weights as specified by paper |
| Ensemble membership | All models or greedy validation selection (Algorithm 1) | Both paths exist | Primary all models; greedy secondary |
| Greedy acceptance | Accept if candidate validation score >= previous | `ens.py` uses strict `>` | Use paper's >=; deterministic view-ID tie breaking |
| Split used for greedy selection | Validation | Validation in greedy outer loop | Validation only; never choose using test scores |

Raw scorer output means the released `edge_att` output, without silently
adding sigmoid normalization before score aggregation. Within classifiers,
the released mask semantics still apply sigmoid through the compatibility
bridge. This follows the released score representation; the paper alone does
not fully specify this numerical convention. Directed edges, ceil top-k,
feature encoder output, and GNN classifiers remain those of the release.

## What is implemented and checked

`adapters/evaluate_reference.py` is a no-training, validation-only evaluator.
It verifies pinned source hashes, raw data hash, and each stored view's selected
validation AUC within 1e-4 before aggregation. It refuses checkpoints marked
as contrastive-indexing-corrected. It never constructs a test loader.

It reports all_mean (primary), all_max, and optionally greedy_mean/greedy_max,
with full selection traces and selected view IDs. Greedy results are fitted
and scored on the same validation set: they are selection statistics, not
unbiased generalization estimates. No best variant is automatically promoted.

Tests compare the two shared aggregations against direct calculations using
actual upstream CIGA models, verify shared edge identity, unchanged aggregate
inputs, label independence, view-order invariance, and greedy tie/rejection
behavior. A tiny checkpoint-reload end-to-end smoke evaluation passed; its
artificial AUC values are not benchmark evidence.

## Limits on claims

This is a paper-oriented ENS inference implementation with documented
departures from problematic released behavior. It is not bit-identical execution
of `ens.py`. It does not establish that published results used label-dependent
voting, nor certify that our raw dataset is identical to the authors' files.

One fixed recipe, one initialization, and validation-only evaluation are not
the paper's hyperparameter search, five-seed benchmark or final test result.
Training remains the released behavior, including the known feature-contrastive
indexing issue. The corrected training run is a separate scientific variant.
WA, validation-weighted score aggregation (`sub_softavg`) and independent
subgraph voting are outside this evaluation. No four-run replication or
Ki/GOOD continuation is authorized by this evaluator.

## Frozen test protocol (prepared October 17)

`configs/sugar_reference_test_20261017.json` pins the completed validation record,
checkpoint and raw data hashes, primary all_mean, and all four view subsets.
The evaluator's `--test-protocol` plus `--frozen-validation` mode first replays
individual and frozen ensemble validation AUCs, then evaluates OOD test once.
`--greedy` is rejected in this mode. It never searches or changes membership on
test. All four results must be reported; the primary remains all_mean.

Zero new training is required. This is the released-training initialization-1
reference, not the corrected-indexing variant and not a five-seed benchmark.
Preparing this path does not authorize launching a new remote evaluation.
