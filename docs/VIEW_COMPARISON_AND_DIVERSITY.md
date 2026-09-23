# View comparison and diversity warmup

Implemented on `fix/generalization-audit-and-training-pipeline`. No existing
experiment outputs changed. Existing training configs keep their prior behavior:
schedule delay/ramp defaults are zero. The new comparison configs enable early
stopping; it was not silently enabled for historical configs.

## Dumb It Down

We can now tell whether five views are effectively duplicates, whether they
select different molecular information, and whether those differences help
validation predictions. We also have a gradual diversity schedule that lets
prediction learning begin before encouraging views to differ.

## What the comparison does

`scripts/compare_views.py` loads a saved checkpoint, puts every view in evaluation
mode, and loads only its configured validation split. No training or test access.
Currently it supports at least two non-ERM views using one fragmentation method,
which matches our five GroupGraph views. Mixed fragmentation methods fail clearly.

It reports:

- Each view's AUC/accuracy on its own selected graph.
- Pairwise positive-class probability correlation, prediction disagreement, and
  the fraction of examples both views classify incorrectly.
- Mean per-molecule selected-edge Jaccard overlap and score correlation. Empty
  edge unions and undefined correlations are excluded with valid counts reported.
- Each classifier's AUC on the full ensemble's shared selected graph.
- Current full shared-graph ensemble AUC.
- Independent-graph soft-voting AUC: each view predicts its own rationale first,
  then probabilities are averaged.
- Leave-one-vote-out AUC, holding the full shared graph fixed.
- Leave-one-entire-view-out AUC, recalculating both graph selection and voting.

Jaccard 1 means identical selected edge sets; lower values mean less overlap.
Probability correlation measures similar scores, not identical mistakes.
High overlap alone does not prove waste; diversity alone does not prove usefulness.
The tool verifies that explicitly averaging shared classifier probabilities
matches the production ensemble probabilities.

These are frozen-checkpoint comparisons. An individual view extracted from a joint
five-view model is not equivalent to a separately trained single-view baseline.
Likewise, an alternative inference path did not select this checkpoint's epoch.
Differences should motivate a matched training/selection experiment.

## Technical Explanation: schedule and stopping

`lambda_diversity` is the target coefficient. The effective coefficient is:

```text
epoch <= delay: 0
after delay: target * min(1, (epoch - delay) / ramp)
zero ramp: switch directly to target after delay
```

Epochs are one-based. With delay10/ramp10/target0.1, epochs1–10 use zero,
epoch11 uses .01, epoch20 reaches .1, later epochs stay at .1.
Training and validation loss reporting use the same coefficient for each epoch.
CSV and checkpoint history include `effective_diversity_weight`.

We retain the highest individual validation AUC checkpoint from the entire run.
An early pre-diversity checkpoint can still win; that would not demonstrate a
benefit from activated diversity. Selection uses no test results.

Early stopping starts spending patience only after both the configured minimum
epoch and the end of the diversity ramp. All three comparison configs use:

- 100 maximum epochs, patience20, minimum30 epochs.
- Five views, contrastive .1, IRM0, best individual validation epoch.
- Same preprocessed train/validation data and seeds101/102.

With no later improvement, the earliest stop is epoch50. Improvements reset
patience. This gives the ramp ten full-strength epochs before patience starts.

Schedules compared: diversity off; constant .1; delay10/ramp10 to .1.
The target is deliberately smaller than our previous .5 probe. Off remains the
reference. The constant .1 case distinguishes lower weight from delayed activation.
Scheduling changes total regularization exposure as well as timing, so this is
not an equal-integrated-penalty experiment.

Exact schedule resume is not implemented. Resuming a scheduled training checkpoint
raises an explicit error rather than silently restarting its schedule at epoch1.
Evaluation and comparison of those checkpoints remain supported.

## Server commands

Upload current `train_sugar.py`, `src/training/sugar_trainer.py`,
`src/training/diversity_schedule.py`, `scripts/compare_views.py`,
`scripts/run_diversity_schedule.py`, all three `configs/diversity_schedule_*.yaml`,
and the updated tests. Keep the existing processed cache and checkpoints.

From the project root inside tmux, compare one checkpoint:

```bash
.venv-reu-server/bin/python scripts/compare_views.py outputs_overnight_losses_20260913_102501/eval/runs/artifacts/drugood_lbap_core_ec50_assay/group_graph/eval_run00_seed1001/checkpoints/best_model.pt --device cpu
```

Output: `outputs_view_comparison_<timestamp>/comparison.json`.
Use `--max-batches 2` only for execution checks; partial AUC is not representative.
Use `--processed-dir PATH` if the saved cache path has moved.

Check the matched schedule experiment commands without training:

```bash
.venv-reu-server/bin/python scripts/run_diversity_schedule.py --dry-run
```

Then run all six training runs sequentially:

```bash
.venv-reu-server/bin/python scripts/run_diversity_schedule.py --device cuda
```

This is foreground execution with visible logs. Start it once. In tmux, detach
with Ctrl+B then D. Each launch gets new timestamped run names; do not launch
duplicates. Outputs are under `outputs_diversity_schedule/<dataset>/group_graph/`.
No automatic test evaluation occurs in this schedule comparison.

## Verification

122 tests pass, including real synthetic scheduled training, effective-loss scaling,
an intentionally identical-view control, delayed patience, and restoration of an
earlier best checkpoint. A zero-variance probability vector produces undefined
correlation rather than a false numeric result. Full validation checkpoint
comparisons were also run locally; no real-data training was launched locally.

## Recommended next steps

Measure existing views before assuming collapse. Compare off/constant/ramp on
validation, then decide whether diversity belongs in the next search. Investigate
independent versus shared-graph inference if the saved-checkpoint comparisons
support it, with checkpoint selection aligned to the inference path. Keep IRM off.
Then tune ordinary training/retention parameters on EC50-Assay. Prepare other
DrugOOD variants and an explicitly chosen OGB dataset in parallel, preserving
official split and metric semantics. Retain a matched ERM baseline in the study.

## Full validation findings from existing checkpoints

All three selected contrastive-only checkpoints were compared on all 2,572
validation molecules, on CPU. No test evaluation or retraining was performed.

| Seed | Current shared graph AUC | Own-graph probability vote AUC | Mean pairwise edge overlap |
|---|---:|---:|---:|
| 1001 | 0.7866 | 0.7982 | 0.485 |
| 1002 | 0.7825 | 0.7989 | 0.364 |
| 1003 | 0.7702 | 0.7955 | 0.447 |

These models have not collapsed into identical views. Independent-graph voting
improved validation ranking in all three checkpoints. Current CPU shared-path
AUCs differ slightly from saved CUDA reports (less than 0.00005); no bit-identical
cross-device replay is claimed. Alternative inference was not used to select
these checkpoint epochs, and the three seeds share the same validation dataset.

Recommended priority: compare inference policies with validation checkpoint
selection aligned to each policy, before large hyperparameter tuning. Keep the
new diversity schedule as a controlled experiment rather than an assumed fix.
No claim about alternative-inference test AUC follows from this comparison.
