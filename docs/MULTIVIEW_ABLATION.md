# Next experiments: best individual validation AUC

Policy adopted by the user on September 13, 2026: select the highest individual
ensemble OOD validation ROC-AUC and restore all views together. Window 1 implements
this policy. The main config and future search spec now use window 1. Historical
selection-comparison configs retain window 5 intentionally. No historical outputs
have been rewritten. Smoothed curves can still be calculated from saved epoch AUCs;
they no longer determine selection in the new configs.

`early_stopping: false` now explicitly disables patience-based stopping in ordinary
training. The new experiments and main config use a fixed 100-epoch budget. Existing
configs without that field retain prior early-stopping behavior. The ordinary
checkpoint and `final_evaluation_metrics.json` from the new runs now describe the
best individual epoch, not a moving-average endpoint. History identifies its epoch.

## Stage 1: do multiple views help?

Run five Group Graph views with prediction loss only. Compare against the existing
one-view fragment `best_raw` results from the fixed-budget study (mean validation
AUC .7764). The existing one-view ERM raw results remain a reference. This measures
the package of extra views and shared-subgraph/soft-vote inference, not an isolated
change to one GNN layer. More views also cost more compute. The selected validation
scores are development evidence, not unbiased independent-test estimates.

## Upload and execute

Upload updated `train_sugar.py`, `configs/experiment.yaml`, `configs/grid_search.yaml`,
the four `configs/multiview_*.yaml` files, `scripts/run_multiview_ablation.py`,
`scripts/smoke_test.py`, and `tests/test_multiview_ablation.py`. Retain the other fixes
already on the server. Leave the server environment/data/results intact.

From the repository root in the server's existing environment:

```bash
source .venv-reu-server/bin/activate
python -m unittest discover -s tests -q
python scripts/smoke_test.py --device cuda --ablation
python main.py preprocess --config configs/multiview_prediction.yaml --validate-loaders
python scripts/run_multiview_ablation.py views --dry-run
python scripts/run_multiview_ablation.py views
```

Local verification: 111 tests pass. Four synthetic CPU five-view cases each complete
two epochs and checkpoint reload parity. CUDA must be checked on the server. Stop
on test/preprocessing errors before expensive runs.

Preprocessing IS needed once for this stage: the recipe changes from one to five
views, and the current cache hash includes those counts. The new cache uses
`data/processed/multiview_ablation_v1/` to preserve the old control cache. All four
new configs have matching preprocessing hashes. If the server uses a different raw
data path, edit all four consistently before preprocessing. Inspect failures and
split counts; stop on unexpected differences from the earlier control dataset.

The views command launches exactly three 100-epoch runs sequentially. It refuses
to overwrite existing destinations. Use a persistent session or scheduler if
disconnecting SSH. Do not run the main full config for this controlled stage: it
still has auxiliary objectives and optional greedy selection enabled.

## Stage 2: auxiliary objectives, after reviewing Stage 1

Nine runs are prepared, not automatically launched:

| Configuration | Contrast | Diversity | IRM | Environments |
| --- | ---: | ---: | ---: | ---: |
| multiview_prediction | 0 | 0 | 0 | 1 |
| multiview_diversity | 0 | 0.5 | 0 | 1 |
| multiview_contrastive | 0.1 | 0 | 0 | 1 |
| multiview_irm | 0 | 0 | 2.0 | 2 |

Each has five views, seeds 11/22/33, the same 100-epoch budget and best-raw policy.
The diversity/IRM coefficients probe values already in the main config; contrast
0.1 is an exploratory setting. These are not tuned optima. IRM sums across views
and environments, so 2.0 may be strong; a negative result at this value does not
prove IRM is inherently unhelpful. Verify environment composition and batch coverage
before running/interpreting the IRM arm. Missing keys can trigger a random fallback.

After Stage 1 review, the separate command is:

```bash
python scripts/run_multiview_ablation.py auxiliary
```

No additional preprocessing is needed for this auxiliary stage. It performs nine
100-epoch runs. It tests each term alone, not combinations or interactions. If Stage
1 exposes a problem, diagnose it first rather than automatically launching Stage 2.

## What to send back

Download `outputs_multiview_ablation/drugood_lbap_core_ec50_assay/group_graph/`.
Stage 1 folders are `multiview_prediction_s11`, `multiview_prediction_s22`, and
`multiview_prediction_s33`. Include effective configs, manifests, loss CSVs and final
metrics. The primary metric is `metrics.ood_val.auc` in `final_evaluation_metrics.json`.
Checkpoint files allow follow-up checks. Preserve source revision/patch or hashes,
server versions, GPU model and preprocessing statistics alongside the experiment.

All new configs are validation-only: no test split, report splits, per-epoch test
curve, greedy subset selection or weight averaging. The next action after these
stages is to interpret the controls and decide whether a narrow search is warranted.
Optuna is not part of the current run script. Its future spec uses a fresh study
name/output directory so the changed policy does not mix with an old study.
