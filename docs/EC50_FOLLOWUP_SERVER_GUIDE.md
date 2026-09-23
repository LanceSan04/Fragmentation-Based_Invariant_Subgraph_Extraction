# EC50 follow-up: exact server instructions

## What changed

Two new scripts; production training/inference functions are unchanged.

- `scripts/compare_inference.py`: reloads saved fragment checkpoints and compares shared-graph inference with independent per-view top-k probability voting on validation AND test. No retraining. Saves each example's label/probabilities and reports shared AUC parity against the original result (absolute tolerance .0001). Stops on failed parity; inspect before interpreting differences. Existing checkpoints selected by shared validation remain fixed, so this is exploratory, not a policy-specific checkpoint selection experiment.
- `scripts/run_ec50_followup.py`: two extra Assay seeds, five-seed result summary, audited preprocessing for Scaffold/Size, sequential five-seed fragment and ERM comparisons, and a synthetic smoke workflow. Fresh output/cache locations are required. No background launching, deletion, dependency installation, automatic resume, or overwriting existing run directories.

## Technical Explanation

Both inference policies retain top-k. Shared: average edge scores, choose one graph, run every classifier on it, average probabilities. Independent: each scorer selects its own graph, its corresponding classifier predicts on it, then probabilities are averaged. Training-only edge sampling is disabled for both evaluation paths.

For new datasets the frozen Assay settings are lr .0003, decay0, dropout .5, edge ratio .5, fragment sampling .7; hidden128, three layers, batch64, five GroupGraph views, contrast .1/diversity0/IRM0. Max100 epochs, min30/patience20 early stopping; best individual validation AUC selects checkpoints. Test is evaluated after training/restoration, never used for checkpoint selection.

The ERM control is explicitly a FIVE-CLASSIFIER FULL-GRAPH ENSEMBLE. It matches ensemble size, GNN architecture, training settings, seed list and stopping protocol. Fragment scoring/selection is bypassed and all auxiliary losses are zero. This contrasts the entire fragment+contrastive method against ERM; it does not isolate fragmentation alone, and it is not a separately tuned or single-GNN ERM benchmark. Preprocessing still builds shared cached fragment tensors, but ERM prediction ignores them.

Scaffold and Size train new models on their own training splits. This evaluates transfer of the Assay-tuned configuration, not application of the trained Assay weights to other datasets. Keep these as fixed initial benchmarks before any dataset-specific tuning. Report all seeds, not the best test seed.

## Dumb It Down

First reuse what we trained to test a different way of combining predictions. Then finish five Assay runs. Finally check whether the same recipe works on different molecular splits and whether it beats a comparable ensemble that sees entire molecules.

## Upload using WinSCP

Into `/home/lsantiago4/REU_Summer_2026/`, upload:

1. `scripts/compare_inference.py`
2. `scripts/run_ec50_followup.py`
3. `configs/parameter_winner_test_base.yaml` (the current frozen winner config)
4. `docs/EC50_FOLLOWUP_SERVER_GUIDE.md` (optional on server)
5. `data/raw/drugood/lbap_core_ec50_scaffold.json`
6. `data/raw/drugood/lbap_core_ec50_size.json`

Keep the existing current source, `scripts/compare_views.py`, `scripts/grid_search.py`, and `scripts/run_overnight.py`. The new scripts import these. Do not replace the server venv with the Windows venv.

Keep `outputs_parameter_winner_test` including its `eval/runs/artifacts/.../checkpoints/best_model.pt` files, and the existing Assay cache `outputs_overnight_losses_20260913_102501/processed`. The config currently points to that cache under your `/home/lsantiago4` repository.

## 1. Enter tmux and smoke test

```bash
cd /home/lsantiago4/REU_Summer_2026
tmux new-session -A -s sugar_run
```

Then, inside tmux:

```bash
cd /home/lsantiago4/REU_Summer_2026
.venv-reu-server/bin/python -u scripts/run_ec50_followup.py smoke --device cuda
```

Wait for `SMOKE PASSED`. This trains tiny synthetic fragment and ERM models and evaluates both inference policies. Synthetic AUC is not a research result. This does not need the new raw datasets.

## 2. Seed 3003 inference comparison — no training

```bash
.venv-reu-server/bin/python -u scripts/compare_inference.py outputs_parameter_winner_test --seeds 3003 --device cuda --output-dir outputs_inference_seed3003
```

Then compare the other two existing seeds without rerunning 3003:

```bash
.venv-reu-server/bin/python -u scripts/compare_inference.py outputs_parameter_winner_test --seeds 3001 3002 --device cuda --output-dir outputs_inference_seed3001_3002
```

Each directory contains `comparison_all.json` plus per-run `comparison.json` and `ood_val_predictions.npz` / `ood_test_predictions.npz`. In JSON, `auc_delta` means independent minus shared; positive favors independent. NPZ row indices refer to the corresponding processed split's sequential example order, with labels and both probability vectors.

Seed3003 was identified after observing its test performance. Do not report only that seed's comparison. Both policies are fixed in advance; do not repeatedly adjust them to improve this test set.

## 3. Add Assay seeds 3004 and 3005 — two training runs

```bash
.venv-reu-server/bin/python -u scripts/run_ec50_followup.py assay-extra --device cuda --output-dir outputs_parameter_winner_extra
```

This uses the same configuration/stopping policy as seeds3001–3003. It saves the additional results in `outputs_parameter_winner_extra/eval/eval_results.json`.

After it completes, combine ALL FIVE seeds:

```bash
.venv-reu-server/bin/python scripts/run_ec50_followup.py summarize --results outputs_parameter_winner_test outputs_parameter_winner_extra --output-dir outputs_parameter_winner_five_seed_summary.json
```

This command's `--output-dir` argument names the output JSON file. It rejects duplicate seeds and mismatched key configurations. The `summary.ood_test.mean/std/n` fields report the five-run AUC summary. Standard deviation measures run variation, not a confidence interval.

Compare inference for the two new checkpoints too:

```bash
.venv-reu-server/bin/python -u scripts/compare_inference.py outputs_parameter_winner_extra --device cuda --output-dir outputs_inference_seed3004_3005
```

## 4. Prepare each new dataset — separate caches

```bash
.venv-reu-server/bin/python -u scripts/run_ec50_followup.py prepare --dataset scaffold --device cpu
.venv-reu-server/bin/python -u scripts/run_ec50_followup.py prepare --dataset size --device cpu
```

These audit the supplied train/ood_val/ood_test records, canonical molecular overlap, labels and domain IDs; preserve native splits; preprocess them; and verify basic loader batching. They write independent caches under `data/processed/ec50_followup_v1/scaffold` and `/size`. Each cache includes `raw_audit.json` and, after success, `ready.json` with counts and raw-file hash. Training checks that the source matches the prepared cache. Read the printed processed counts and preprocessing failure reports; raw and usable sample counts may differ.

Local RAW audits (not a guarantee that every molecule fragments successfully):

| Dataset | Train | OOD validation | OOD test |
|---|---:|---:|---:|
| Scaffold | 2570 | 2532 | 2533 |
| Size | 4684 | 2313 | 2398 |

Both files had zero invalid SMILES and zero canonical isomeric-SMILES overlap across these three splits. This check does not establish independence under every chemical-equivalence definition. Scaffold training is about94% positive, versus74% validation/72% test. Size is about86%/72%/68%. Domain IDs are present; IRM remains off.

If a cache already exists, preparation stops rather than replacing it. For a deliberate fresh cache, pass the SAME `--cache-dir NEW_PATH` to both prepare and train. Do not reuse an Assay cache for another split benchmark.

## 5. Train new benchmarks — sequentially

```bash
.venv-reu-server/bin/python -u scripts/run_ec50_followup.py train --dataset scaffold --device cuda --output-dir outputs_ec50_scaffold_benchmark
.venv-reu-server/bin/python -u scripts/run_ec50_followup.py train --dataset size --device cuda --output-dir outputs_ec50_size_benchmark
```

Each command runs five fragment seeds3001–3005, then five matched ERM seeds3001–3005: TEN training runs per dataset, TWENTY total. There are at most2000 training epochs across the two datasets; early stopping usually shortens this. No runtime promise. To split the budget into separate jobs use `--model fragment` or `--model erm` with different fresh output directories.

Final files:

```text
outputs_ec50_scaffold_benchmark/fragment/eval/eval_results.json
outputs_ec50_scaffold_benchmark/erm/eval/eval_results.json
outputs_ec50_size_benchmark/fragment/eval/eval_results.json
outputs_ec50_size_benchmark/erm/eval/eval_results.json
```

Within each: `summary.ood_test.auc.mean/std/n`. Checkpoints and histories remain under `eval/runs`.

## Running and returning results

Commands run in the foreground and display logs in tmux. Launch once. Detach with Ctrl+B then D; reconnect with `tmux attach -t sugar_run`. Do not start duplicate GPU jobs. Existing output directories cause an error; intentional retries require fresh output names. An interruption leaves partial artifacts, not a completed experiment or automatic resume.

Return all inference-comparison output folders, the five-seed summary, the extra Assay output, both benchmark output folders, and cache audit/ready JSON files. Keep all checkpoints on the server. We will compare paired policy deltas and paired fragment-vs-ERM results, including weak seeds.

Local verification: synthetic fragment+ERM training and checkpoint comparison passed; 127 regression tests passed. No real training or new real test evaluation was run locally for this implementation.
