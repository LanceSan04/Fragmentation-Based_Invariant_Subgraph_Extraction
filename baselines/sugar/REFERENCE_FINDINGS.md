# SuGAr reference checks and next validation run

The upstream snapshot remains unchanged at commit
`b2597710e56cd15370d86794a6ea3e1a61c4bb6a`.
Run `python baselines/sugar/adapters/audit_reference.py` for small executable
counterexamples. These use the actual pinned functions extracted through AST,
not rewritten approximations. No dataset or training is involved.

## Confirmed behaviors

1. `upstream/ens.py::soft_vote` (lines 267-282, 409-424): model weights are
   computed from AUC on the evaluated loader. Changing labels with predictions
   held fixed changes the aggregate prediction. Do not use this path for test
   benchmarking. This does not establish which path produced published numbers.
2. `upstream/models/ciga.py::split_graph` (line 275): dropped scores are
   multiplied by -10 in place. Reusing the vector can change subsequent selected
   edges (demonstrated with negative scores). `forward(..., return_data='attention')`
   returns that modified vector. The shared aggregation path must not assume
   immutable, raw scores. This mutation also feeds the released training's
   diversity computation; removing it would change training semantics.
3. `upstream/models/ciga.py::forward` (line 352): feature contrastive pooling
   relabels original embeddings using already-remapped edge indices. A retained
   edge between original atoms 2 and 3 becomes 0 and 1, and the second relabel
   reads original embeddings 0 and 1. This affects the pilot's `c_rep=feat`
   configuration. Its benchmark effect size is unknown.

## Decision

First obtain one full validation-only released-training trajectory as a
diagnostic reference, preserving the pilot's recipe and source behavior.
This is not an endorsed bug-free model or a completed paper reproduction.
Changing the feature indexing, diversity scores and aggregation together would
prevent attribution of differences to specific corrections.

Proposed scope: EC50 Assay, initialization 1, ten jointly trained views, 40
epochs; same 39/10 features, split membership, batch 128, hidden 32, three
layers, Adam .001, dropout .1, contrast 4, temperature 1, penalty 1,
sampling 95%, selected edge ratio .25, `c_in=feat`, `c_rep=feat`.
Per-view best validation checkpoints; no test data evaluation, no ensemble
evaluation, no tuning, retries or subsequent jobs. Ten views are not ten
independent repeat seeds. The paper specifies 40 epochs by default, but this
single fixed recipe does not reproduce its validation hyperparameter search.

The two-epoch pilot took 61 seconds including checkpoint replay and allocated
519 MiB GPU memory. A linear projection is about 20 minutes for 40 epochs;
plan for 20-40 minutes, without treating the estimate as a kill limit.

After this reference, prioritize an opt-in correction to the feature indexing
with a forward/gradient test and matched initialization comparison. Before any
test benchmark, specify a voting rule whose predictions cannot depend on test
labels. Uniform probability voting is a useful independent-view control, but
does not by itself implement the paper's shared-subgraph aggregation. Do not
equate the two or claim paper replication from it.

Paper: https://arxiv.org/html/2410.22228v2 (optimization and model selection;
Algorithms 1 and 3).

## Implemented opt-in corrections and inference

`adapters/run.py --contrastive-indexing corrected` replaces only the final
feature-contrastive relabel call with the original retained atom indices.
Default `released` remains unchanged. The pinned snapshot is never edited.
`tests/test_sugar_corrections.py` verifies selected noncontiguous atom embeddings,
gradient destinations, and identical prediction logits before training.

`--report-ensemble` evaluates validation only after restoring all per-view
best checkpoints. Both policies use uniform probability voting:

- `independent_uniform`: each view predicts from its own selected graph.
- `shared_raw_uniform`: average raw pre-selection edge logits; each classifier
  receives a fresh copy of that average for its own released selection routine.

No label is passed into either prediction policy, no score is weighted by the
evaluated labels, and no policy is chosen automatically. The shared policy
implements the raw-score average/top-k idea, while replacing the released
modified-attention and label-dependent voting behavior. It is a documented
adaptation, not bit-identical released ensemble execution. The selected edge
identities coincide only when models use the same ratio and tie handling.

`--reference-checkpoint PATH` verifies identical recipe and dataset metadata,
replays the original per-view validation AUCs, then evaluates the saved released
models with these same two policies. This avoids retraining the control and
does not use test data. Both policies and every view must be reported, rather
than selecting whichever shows the largest correction benefit.

The matched synthetic training/reload/ensemble check passes. Its artificial
AUCs are software checks and provide no performance evidence. Diversity-score
mutation and other released training behavior remain preserved to isolate the
indexing correction. Real-data impact still requires an approved matched run.
