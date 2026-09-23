> September 23 update: historical/generated folders were archived. Current locations and recovery instructions are in [CLEANUP.md](CLEANUP.md); older location statements below are historical.

# Consolidated codebase and server guide

## Decision

Main.py is the user-facing entry point. Workflow implementation lives in src/experiments. Legacy script names delegate to those same modules and remain import-compatible. No model, loss, sampling, checkpoint-selection or dataset semantics were intentionally changed. Existing result schemas remain readable. No historical folders or caches were moved/deleted.

## Command map

| Task | Command prefix |
|---|---|
| Preprocess / train / saved-checkpoint evaluation | `python main.py preprocess`, `train`, `eval` |
| Config-driven general search / repeated final evaluation | `python main.py search`, `search-eval` |
| Bounded Assay/Size parameter search | `python main.py experiment tune` |
| New-dataset preparation, benchmarks, extra Assay seeds, summary | `python main.py experiment benchmark` |
| Fixed-epoch dual-policy comparison or matched ERM | `python main.py experiment select` |
| Compare shared/independent on saved checkpoints | `python main.py experiment compare-inference` |
| Per-view overlap and prediction diagnostics | `python main.py experiment compare-views` |
| Domain/gradient diagnostics | `python main.py experiment diagnose` |
| Historical overnight loss-grid workflow | `python main.py experiment overnight` |
| Historical multiview / diversity schedules | `python main.py experiment ablation`, `diversity` |

Use `--help` after a command for its unchanged workflow options. The main.py `search` and `search-eval` commands preserve their existing preprocessing defaults: use `--skip-preprocess` to reuse prepared data. Legacy recipe commands keep their historical configured output behavior; do not launch them casually as new benchmarks.

## Where implementation belongs

```text
main.py                         user entry point; dispatch only
train_sugar.py                  ordinary training/checkpoint evaluation API
src/experiments/
  cli.py                        experiment command dispatch + artifact index
  search.py                     Optuna and repeated final evaluation
  parameter_search.py           bounded search orchestration
  selection.py                  fixed-budget checkpoint-policy comparison + ERM
  benchmark.py                  dataset audit/prepare/repeated benchmarks
  inference.py, views.py        offline checkpoint comparisons
  diagnostics.py                domain/gradient probes
  overnight.py                  historical factorial workflow
  ablation.py, diversity.py      historical controlled recipes
  plans.py                      shared configs and synthetic plan construction
  common.py                     shared RNG setup and mean/std helpers
  metrics.py                    probability metrics
configs/
  workflows.yaml                search space/budget/seeds and selection defaults
  parameter_winner_test_base.yaml  current frozen benchmark training settings
scripts/                        thin legacy wrappers + utility tools
```

Ordinary training and fixed-budget dual selection remain intentionally distinct protocols. The latter calls the same trainer's train_epoch rather than duplicating the learning algorithm. It now has one maintained implementation in src/experiments/selection.py. We did not merge the protocols merely to reduce file count.

## Output organization

Automatic output paths for the new experiment entry point:

```text
outputs/assay/select/<timestamp>/
outputs/size/tune/<timestamp>/
outputs/scaffold/benchmark/<timestamp>/
outputs/analysis/compare-inference/<timestamp>/
outputs/smoke/<workflow>/<timestamp>/
data/processed/ec50_followup_v1/<dataset>/
```

Explicit `--output-dir` remains respected. The legacy script CLI retains its previous defaults. Benchmark preparation writes its reusable cache; summarize writes the explicitly requested JSON file.

Each newly created experiment output directory launched through main receives experiment.json with command arguments and paths to native result files. Status `returned` means the workflow returned, NOT proof all Optuna trials completed; inspect native run/trial states. Exceptions are marked failed when a new output directory exists. Existing metadata is never overwritten. This index unifies discovery without rewriting historical metric schemas.

Look at:
- Search: search/leaderboard.csv, search/best_params.json, search/runs/.
- Dual/ERM selection: summary.json, seed*/history.json and best_*.pt.
- Benchmark: fragment/eval/eval_results.json and erm/eval/eval_results.json.
- Offline inference: comparison_all.json and per-checkpoint comparison.json/predictions.

Generated outputs/logs/temp are ignored by Git for future additions. Already tracked files remain tracked. No files were deleted by this change. Separate experimental outputs are necessary evidence; their root-level sprawl is solved for new CLI runs rather than by deleting old experiments.

## Server migration

Upload using WinSCP, preserving paths:
1. Updated main.py.
2. Entire NEW src/experiments directory (all .py files; not __pycache__).
3. Updated scripts wrappers: grid_search.py, run_parameter_search.py, run_ec50_followup.py, run_dual_policy_selection.py, run_overnight.py, compare_inference.py, compare_views.py, diagnose_generalization.py, run_multiview_ablation.py, run_diversity_schedule.py.
4. NEW configs/workflows.yaml.
5. README.md and this guide if desired. Keep all other existing source/config dependencies and server venv/cache.

No global or local package installation is needed. Do not upload the Windows venv, move server caches, or delete old result folders. Upload between jobs, not while running experiments are importing code.

Inside tmux, from server root:

```bash
cd /home/lsantiago4/REU_Summer_2026
.venv-reu-server/bin/python main.py experiment --help
.venv-reu-server/bin/python -u main.py experiment select --dataset assay --smoke --evaluate-test --device cuda
.venv-reu-server/bin/python -u main.py experiment tune --dataset size --inference-policy independent --smoke --device cuda
.venv-reu-server/bin/python -u main.py experiment benchmark smoke --device cuda
```

These are synthetic verification jobs only. Outputs automatically go under outputs/smoke. Do not launch another full research experiment just to test this refactor. Existing old commands remain valid once wrappers AND src/experiments are uploaded together.

Examples for later, when the experiment is agreed:

```bash
# Same historical fixed-budget protocol, new front door; explicit seeds avoid ambiguity.
python main.py experiment select --dataset assay --model erm --seeds 5001 5002 5003 --epochs 100 --evaluate-test --device cuda
# Shared versus independent selection remains an explicit scientific choice.
python main.py experiment tune --dataset size --inference-policy independent --dry-run
```

## Verification and provenance

A ZIP snapshot of all pre-refactor Python source, configs, tests, requirements and Git status was saved in Obby/REU_Code_Snapshots/before_consolidation_20260918_221302.zip. It includes source SHA256 hashes; no datasets/checkpoints/venv were bundled.

Before refactor:131 tests passed. After refactor:135 tests passed, including legacy-import identity, main dispatch, output routing, and metadata preservation. Synthetic main-entrypoint search and fragment/ERM benchmark smokes passed. A deterministic CPU dual-policy experiment run from the snapshot versus refactored source matched EXACTLY in histories, selection epochs, final metrics and all saved model tensors. No real training launched. This confirms that tested fixture, not every conceivable execution or CUDA determinism.

Project edits were already extensive and uncommitted before this refactor. The snapshot preserves that state; no project push was performed. Historical server source hashes will differ from the consolidated source even when behavior is unchanged, so keep both provenance records.
