# SuGAr reference — preparation status

Official source: https://github.com/Nanolbw/SuGAr

Pinned commit: `b2597710e56cd15370d86794a6ea3e1a61c4bb6a`.

`upstream/` is an unmodified source snapshot, with file hashes in
`provenance.json`. Generated artifacts, IDE files and data were omitted.
No top-level license was present in the pinned checkout. Keep this local until
redistribution permission is clarified; do not assume public means licensed.

## Status: training runs; full paper-result reproduction remains incomplete

The two-epoch pilot and one 40-epoch released-training run completed on EC50
Assay with successful per-view validation replay. A corrected-indexing training
comparison is separate. See [PAPER_INFERENCE_RECONCILIATION.md](PAPER_INFERENCE_RECONCILIATION.md)
for the October 17 paper-oriented shared-graph ENS evaluator, its tests and
explicit departures from the released ensemble. Independent voting and the
proposed four-run replication are paused. No SuGAr test benchmark is complete.

Run `python baselines/sugar/check_reproduction.py` for integrity and dependency
inventory. It does not run upstream training or certify compatibility.

The published checkout is incomplete for DrugOOD: configs are absent,
`main.py` passes a nonexistent `seed` keyword to its DrugOOD adapter, and the
ensemble DrugOOD branch references commented-out imports. Its legacy PyG
explanation hooks also require behavior verification, not just import fixes.

The upstream DrugOOD branch declares **39 atom and 10 bond features**. Our
46/14 tensors must not be substituted and called a faithful reproduction.
The original DrugOOD featurization and exact split membership must be verified.

## Adapter status (October 16)

The original DrugOOD featurizer is now pinned under `drugood_reference/` at
`61a70e4ad1fb227e4a264ed5ba87d4c78fdb4ae7` (see its provenance file for
the authoritative full revision and file hashes). The training adapter uses its
39/10 features and preserves raw split membership. It does not use our 46/14 cache.

`adapters/run.py` executes the released training body with narrowly asserted
AST substitutions for data loading, output paths, validation recording and test
reporting suppression. Modern PyG message-mask hooks are supplied separately.
The reference source remains unchanged. The run specification records versions,
arguments and source hashes. No full benchmark reproduction is certified yet.

Local checks passed: modern message gating matches the manual GINE expression;
suppressing per-epoch test reporting preserves selected model tensors bit for
bit on a synthetic fixture, including the DataLoader RNG draw. Every saved view
is reloaded and its selected validation AUC replayed. Synthetic checks establish
compatibility behavior, not real-data accuracy.

Important unresolved reference behavior: the ROC-AUC branch of `ens.py`'s
`soft_vote` uses AUC on the evaluated loader to weight its models. Applied to a
test loader, this makes ensemble weights depend on test labels. Our pilot is
validation-only and does not invoke that ensemble path. This observation does
not establish how the paper's reported numbers were produced. A later benchmark
needs an explicitly documented label-free test voting rule.

Other source behaviors requiring reconciliation include in-place modification
of dropped attention scores in `models/ciga.py::split_graph` and relabeling
contrastive features with already-remapped causal edge indices in `forward`.
They are preserved in this adapter; their performance impact is not established.

The proposed server pilot is two epochs and ten views on EC50 Assay, with no
test evaluation. Dependencies go into a fresh project-local overlay environment;
the existing training environment is only referenced, never modified. This is
a modern compatibility environment, not the original paper environment.

## Remaining compatibility work (historical checklist; partly completed above)

1. Pin the external DrugOOD source and matching configs/featurizer.
2. Create an isolated environment, with compatible resolved versions recorded.
3. Make minimal compatibility patches in a separate adapter/patch layer; leave
   the reference snapshot untouched and record patch hashes and justifications.
4. Verify edge masking changes predictions and gradients, deterministic
   perturbation behavior, per-view contrastive/diversity normalization,
   checkpoint save/load, and ensemble validation replay on tiny graphs.
5. Adapt experiment output/checkpoint locations; avoid evaluating the test set
   each epoch in the reproduction harness, recording this reporting-only change.
6. Run an approved representative resource pilot before defining its sweep.
7. Add the independently validated upstream branch to the queue with its own
   explicit budget. Our native queue does not count it as completed or run it.

The native atom-only control is **not** this reference. A faithful fragment
adaptation also needs matched feature conventions, tuning budgets and inference.
