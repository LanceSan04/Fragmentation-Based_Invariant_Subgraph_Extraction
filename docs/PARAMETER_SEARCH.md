# Diversity decision and next parameter search

## Simple conclusion

Keep diversity off for the next parameter search. Constant diversity hurt both
seeds. The ramp has a higher average, but the run responsible for that advantage
selected epoch5, when diversity was still zero. We have not demonstrated that
turning on diversity helped. This is a conservative working choice, not proof
that every diversity schedule is harmful.

## Results

| Schedule | Seed101 validation AUC | Seed102 validation AUC | Mean |
|---|---:|---:|---:|
| Off | .7798 | .7818 | .7808 |
| Constant .1 | .7745 | .7757 | .7751 |
| Delay10/ramp10 to .1 | .7965 | .7734 | .7850 |

Off selected epochs34/28 and stopped at54/50. Constant selected24/20 and stopped
at50/50. Ramp selected5/48 and stopped at50/68. The effective coefficients follow
the intended schedule. Final metrics contain validation only.

Reproducibility caveat: same-seed off/ramp curves already differ before the ramp
activates. Effective settings match except the intended schedule/output fields,
but this does not prove identical execution, source, or random trajectories.
CUDA nondeterminism is a possible contributor; these artifacts do not establish
the cause. Do not attribute ramp's early advantage to later diversity activation.
The new off results also differ from earlier overnight runs on those seeds.

## Technical Explanation

The next search freezes contrastive .1, diversity0, IRM0, five GroupGraph views,
shared-graph inference, hidden128, three GNN layers, batch64 and validation-based
best individual epoch selection. Early stopping remains patience20 after minimum
epoch30, with maximum100. Comparisons concern the best checkpoint found within
that stopping protocol, not necessarily the best of a fixed 100-epoch trajectory.

Optuna TPE searches categorical candidate values:

| Parameter | Candidates |
|---|---|
| Learning rate | .0001, .0003, .001, .003 |
| Weight decay | 0, .00001, .0001, .001 |
| Dropout | 0, .1, .3, .5 |
| Edge ratio | .2, .3, .5, .7 |
| Fragment sample ratio | .7, .9 |

24 trials including one enqueued reference configuration, two common seeds201/202
per trial: 48 training runs, at most4800 epochs. This is a bounded adaptive search,
not all 512 combinations. The reference trial uses the original parameter settings.
Use its fresh score for comparison, not the best historical validation number.
Runtime may span more than one night.

Each trial's objective is the mean best validation AUC across its two seeds.
All repeats must produce valid scores. Test/report evaluation stays disabled and
the loader only opens train/ood_val. Fresh output directories prevent mixing with
old studies. Config snapshots, source hashes, study database and per-run histories
are saved. This wrapper does not automatically resume or test its winner.

## Server steps

Upload `scripts/run_parameter_search.py`, updated `scripts/grid_search.py`, and
current project source/configs, including `configs/diversity_schedule_off.yaml`.
The real search reads the existing processed data at the path saved in that config.
It does not regenerate it. Do not upload a local virtual environment over the
server environment.

From the project root inside tmux:

```bash
.venv-reu-server/bin/python scripts/run_parameter_search.py --dry-run
.venv-reu-server/bin/python scripts/run_parameter_search.py --smoke --device cuda
```

After the synthetic smoke succeeds, launch once:

```bash
.venv-reu-server/bin/python -u scripts/run_parameter_search.py --device cuda
```

Logs appear in the terminal. Detach tmux with Ctrl+B then D. Re-running launches
another study; do not start duplicates. Output is `outputs_parameter_search_<time>`.

Review `search/leaderboard.csv`, `search/best_params.json`, and `search/runs`.
The script returning does not necessarily mean every trial completed if interrupted;
check leaderboard states/counts. A smoke output is synthetic and not a benchmark.

## After this search

Review the winner against the fresh reference and runner-up configurations. Confirm
promising choices on additional validation seeds before final test reporting.
Keep the deferred independent-inference proposal separate. Prepare other DrugOOD
variants and a specific OGB dataset in parallel, and retain a matched ERM baseline.

No real training was launched locally. Synthetic Optuna integration was executed;
the source regression suite passed. This work does not claim improved real AUC yet.
