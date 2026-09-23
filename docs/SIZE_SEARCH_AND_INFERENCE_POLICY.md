# Size search completion and inference policy

All24 trials COMPLETE;13 unique configurations. Reference trial0 validation mean .742569. Best trial14 .760579 (seeds401/402 .760937/.760220). Winning configuration appeared8times; average trial mean .755928, range .750034–.760579. Repeated common seeds are not independent additional seeds. No test results. Winner: lr .001, decay .0001, dropout0, edge ratio .7, fragment sample ratio .7; contrast .1/diversity0/IRM0. These settings were selected for SHARED inference.

## New setting

```yaml
inference_policy: independent
```

Allowed values: shared (default, including old configs) or independent. This controls trainer validation, early stopping/best-checkpoint selection, final validation/test evaluation, and Optuna objectives through the same ensemble implementation. Independent applies each view to its own selected graph and averages probabilities via stable log-sum-exp. Per-view top-k stays active; training losses/updates unchanged. Explicit policy is saved in effective configuration/checkpoints and final report metadata. Checkpoint evaluation inherits the saved policy when unspecified; explicit overrides permit exploratory evaluation. Resume with a different saved policy is rejected. Historical compare scripts keep their explicit shared baseline via the default ensemble constructor.

This implements ONE selected policy per run. It does not yet implement a fixed-trajectory dual-policy checkpoint-selection experiment. Previous independent comparisons used checkpoints selected for shared inference; this setting permits fresh training selected for independent inference. Existing results/checkpoints are not changed. No real training launched locally.

## Upload

Upload current train_sugar.py, src/models/sugar_ensemble.py, src/training/sugar_trainer.py, scripts/run_parameter_search.py, and scripts/run_ec50_followup.py. Keep existing current source/config dependencies. Last script also prevents pooling result configurations with different policy fields.

From server repository root inside tmux:

```bash
.venv-reu-server/bin/python scripts/run_parameter_search.py --dataset size --inference-policy independent --dry-run
.venv-reu-server/bin/python -u scripts/run_parameter_search.py --dataset size --inference-policy independent --smoke --device cuda
```

If choosing to launch a fresh independent-policy search after smoke:

```bash
.venv-reu-server/bin/python -u scripts/run_parameter_search.py --dataset size --inference-policy independent --device cuda
```

Same bounded space/budget:24 trials x2 seeds401/402,100epochs maximum with early stopping, validation only. Initial reference remains the Assay-derived Size configuration, not the finished shared-search winner. New timestamp output; no old-study continuation or automatic test evaluation. No diversity/IRM changes.

Recommended scientific follow-up: first compare policy-specific selection under a fixed training budget, then run a new independent-objective search. That controlled comparison runner remains separate work.

Verification:130 tests pass; synthetic independent training, checkpoint reload, direct probability parity with manual own-view voting, selected-view voting, invalid-policy rejection, and a real two-trial/two-seed synthetic Optuna smoke passed. CUDA server smoke still required.
