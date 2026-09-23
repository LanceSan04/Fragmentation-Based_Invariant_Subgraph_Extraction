# Focused overnight tuning and test evaluation

Updated September 13, 2026. This is an exploratory experiment, not a guarantee of
better performance or completion before morning.

## Dumb It Down

We are keeping most of the recipe unchanged and trying combinations of all three auxiliary losses. Each recipe gets the same two random starting conditions. Validation
scores choose the recipe. We then train that recipe three more times and measure
each on the held-out test molecules. We report all three, not just the best one.

IRM encourages a predictor to behave consistently across groups of training
molecules. Its coefficient controls how strongly that goal competes with ordinary
prediction loss. Zero switches the IRM penalty off.

## Technical Explanation

`scripts/run_overnight.py` uses the corrected `scripts/grid_search.py` functions.
It reads `configs/multiview_irm.yaml`, then creates a frozen effective base config
and a fresh preprocessed cache inside its output directory. It includes train,
ood_val and ood_test. Preprocessing the held-out split does not train on it.
Search explicitly disables test/report evaluation and test curves.

Protocol:

| Setting | Value |
|---|---|
| Fragmentation | Five group_graph views |
| Checkpoint selection | Highest single-epoch ensemble validation ROC-AUC |
| Training budget | 100 epochs; early stopping disabled |
| Contrastive candidates | 0, 0.1 |
| Diversity candidates | 0, 0.5 |
| IRM candidates | 0, 0.5, 1, 2 |
| Environment construction | Two buckets ordered by domain value, existing trainer logic |
| Search seeds | 101 and 102 for every candidate |
| Search objective | Mean of each seed's best validation AUC |
| Final retraining seeds | 1001, 1002, 1003 |
| Final reporting | Each selected checkpoint's ood_test AUC, then mean and sample SD |

"Best individual" means one epoch, not one of the five GNN views. Each checkpoint
contains the jointly trained five-view ensemble. Final runs use training data to
fit weights and validation data to select epochs. They do not combine train and
validation into a larger training split.

This is an Optuna-managed exhaustive grid over three loss coefficients, rather than an
adaptive TPE search. All 16 combinations are evaluated, allowing matched comparisons at fixed other-loss settings.
[Optuna 3.6.2 GridSampler documentation](https://optuna.readthedocs.io/en/v3.6.2/reference/samplers/generated/optuna.samplers.GridSampler.html).

There are 32 search runs and 3 final runs: 35 runs / 3,500 epochs total, plus
preprocessing. Estimate runtime using the duration of your previous five-view
100-epoch runs. Two search seeds are a practical budget choice, not strong proof
that a small difference is real. New evaluation seeds change training randomness;
they do not create a new validation or test dataset.

## Server commands

Upload the current project code (especially both scripts below and the current
multiview config). Keep your existing data and server virtual environment.
Changed/new files for this turn: `scripts/grid_search.py`,
`scripts/run_overnight.py`, `tests/test_overnight_search.py`, this guide.
The runner also depends on the earlier fixes already in `src/`, `train_sugar.py`
and `configs/multiview_irm.yaml`; upload them if your server copy is older.

From the project root on the Linux server, activate the environment using its
actual location. If it is inside the repo:

```bash
source .venv-reu-server/bin/activate
python -c "import torch, optuna; print('CUDA:', torch.cuda.is_available()); print('Optuna:', optuna.__version__)"
python scripts/run_overnight.py --device cuda --dry-run
python scripts/run_overnight.py --device cuda --smoke
```

The smoke run uses artificial labels and tiny synthetic splits (including
overlapping molecules for execution testing). Its AUC has no scientific meaning.
It performs 18 two-epoch runs with real Optuna, preprocessing, training and test
reporting. Its output directory starts with `outputs_overnight_smoke_`.

After the smoke succeeds, launch the real run so disconnecting SSH does not stop it:

```bash
RUN="outputs_overnight_losses_$(date +%Y%m%d_%H%M%S)"
nohup python -u scripts/run_overnight.py --device cuda --output-dir "$RUN" > "${RUN}.log" 2>&1 &
echo "PID=$! OUTPUT=$RUN LOG=${RUN}.log"
```

Save the printed output directory name. Check progress:

```bash
tail -f "${RUN}.log"
```

Ctrl+C exits `tail`; it does not stop the background training process. After
reconnecting, set RUN to the printed directory name again to reuse these commands.
The output directory must not already exist. Do not create it before launching.

## Where the actual test result is

`<RUN>/eval/eval_results.json`:

- `summary.ood_test.auc.mean`: mean final test ROC-AUC.
- `summary.ood_test.auc.std`: variation across the three training seeds.
- `summary.ood_test.auc.n`: should be 3.
- `runs`: each seed's validation and test metrics.

For a quick terminal summary:

```bash
python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["summary"]["ood_test"]["auc"])' "$RUN/eval/eval_results.json"
```

`<RUN>/COMPLETE.json` appears only after both stages succeed. Without it, inspect
the log before assuming completion. Download the log and output files through
WinSCP. The `processed/` folder can be large; it need not be downloaded for result
review. Keep it on the server if you want to reload checkpoints there.

Other saved files:

- `search/leaderboard.csv` and `search/best_params.json`: validation-only selection.
- `search/runs/*.result.json`: each search run's history and validation metrics.
- `eval/runs/*.result.json`: each final run's metrics/history, saved as it finishes.
- `eval/runs/artifacts/<dataset>/group_graph/<run>/checkpoints/best_model.pt`:
  final run's selected model. Neighboring files include epoch losses and manifest.
- `base_config.yaml`, `search_spec.json`, `provenance.json`: effective settings,
  runtime versions and hashes of source files, including uncommitted edits.
- `environment_summary.json`: actual processed training-domain bucket counts.
- `preprocessing_failures.json`: skipped preprocessing records, if any.

The runner stops if training environment metadata are missing or do not populate
both buckets. This prevents the trainer's random-environment fallback from silently
changing this experiment. The report counts full-dataset buckets; it does not
guarantee every minibatch contains both buckets. Preprocessing remains non-strict
for real DrugOOD, matching the ablations; review any recorded skipped molecules.

Interrupted/incomplete search does not trigger test evaluation. This wrapper has
no automatic resume: preserve the output and ask for a recovery plan. Do not delete
a useful study or restart repeatedly just to obtain a better score.

## Why keep the architecture fixed?

The existing broad config has ten varying parameters and 259,200 possible
combinations. It requests 100 trials x 3 repeats = 300 training runs, versus 32
search runs here. Optuna can explore a space without exhaustively visiting it,
but it cannot remove noise from the measurements supplied to it.

Recent three-seed validation means were about 0.7785 for five-view prediction,
0.7802 with contrastive loss, 0.7519 with diversity, and 0.7859 with IRM. IRM is a
promising lead; its approximately 0.0074 mean improvement is still small and varied
by seed. Diversity at the tested weight hurt all three seeds. That does not prove
all diversity settings are bad or that interactions cannot help.

A broad search changes many causes simultaneously. We could spend much longer and
still be unsure whether an improvement came from IRM, model size, dropout, or a
lucky initialization. Looking at many candidate maxima also increases selection
optimism: the winner's validation score can overstate its generalization. Fixed
seeds help paired comparisons but do not eliminate this bias.

Broad tuning makes sense after this controlled check, with a defined compute
budget and a clear goal. Once we inspect these test scores, subsequent decisions
influenced by them make further test evaluation exploratory. Do not use test AUC
as the Optuna objective or pick the best of the three test seeds.

## Optuna code re-review: actual issues and status

1. **Test split propagation (fixed this turn).** `run_eval` used the spec's test
   split to label its report but did not inject it into the effective training
   config. With a validation-only base, training could omit test evaluation and
   the report would contain null test metrics. It now enables that split and
   raises if its AUC is absent or nonfinite. This was a reporting-path bug; it
   does not explain low historical scores when those runs really evaluated test.
2. **Unequal repeat counts (fixed this turn).** Interrupting a trial, or having an
   invalid repeat, could leave it eligible using fewer valid seeds. Every requested
   repeat must now yield a finite score, otherwise the trial is pruned (excluded
   from winning). Snapshot ranking now explicitly requires COMPLETE trials.
3. **Different seeds per candidate (addressed for this runner).** Historical code
   uses `base_seed + trial_number * repeats + repeat`. This is not intrinsically
   invalid optimization, but adds noise to comparisons. Optional common seeds now
   let all overnight candidates use 101/102. Generic search keeps its old default
   so an old persistent study is not silently reinterpreted.
4. **Moving-average objective vs checkpoint semantics (earlier fixes retained).**
   Commit `201a5b8` changed the objective from returned validation metrics to the
   best moving average in validation history. An average across epochs is not the
   performance of a single checkpoint. Smoothing is a valid design choice, but
   scoring, checkpoint selection and final evaluation must agree on the policy.
   Earlier fixes shared complete-window logic; our chosen window of 1 now aligns
   the objective with the best single-epoch ensemble snapshot. This is not evidence
   that Optuna itself caused the original regression.
5. **Insufficient artifacts (improved this turn).** Search previously saved configs
   but discarded histories; eval suppressed checkpoints too. Histories/metrics
   now persist per completed run, and this runner retains final checkpoints.
6. **Resume protocol risks (avoided by this runner; not generally redesigned).**
   Generic search counts all stored trial states against its budget and does not
   reject an old study just because source/base settings changed. The new wrapper
   requires a fresh directory and verifies all 16 combinations completed before
   evaluating test. Do not resume an old broad study after changing its protocol.

These are code findings. The unavailable historical Optuna artifacts cannot tell
us which issues actually affected those old runs. Tuning cannot compensate for a
broken model path, inconsistent selection policy, or incorrectly configured data.

## Matched comparisons

`search/irm_comparisons.json` compares each positive IRM coefficient against zero
at the same contrastive weight, diversity weight and training seed. Positive
differences mean IRM helped validation AUC. Comparing these differences across
other-loss settings exposes possible interactions. Two seeds are descriptive,
not a strong statistical conclusion. Each run selects its own best epoch using
the same policy. The ordinary prediction loss remains active in every run.

## Verification

All 117 unit tests passed locally. A real Optuna 3.6.2 synthetic CPU integration
completed 16 search runs and two final runs, saved both final checkpoints, and
produced finite test AUC. Reloading a saved checkpoint reproduced its test AUC;
search outputs contained no test metrics. No real DrugOOD training was run locally.
CUDA and server runtime remain to be checked using the commands above.
