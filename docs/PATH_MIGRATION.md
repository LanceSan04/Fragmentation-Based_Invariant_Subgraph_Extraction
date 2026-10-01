# Output and cache migration — September 24, 2026

The active workspace now has one result root: outputs/. Reusable caches belong in data/processed/. Explicit user --output-dir overrides are still honored; defaults no longer create root outputs_* or logs folders.

## Local moves

| Previous | Current |
|---|---|
| outputs_diversity_schedule | outputs/assay/diversity_schedule |
| outputs_policy_comparison | outputs/assay/policy_comparison |
| outputs_overnight_losses_20260913_102501/processed | data/processed/ec50_followup_v1/assay |
| remaining outputs_overnight_losses_20260913_102501 | outputs/legacy/outputs_overnight_losses_20260913_102501 |
| logs | outputs/logs |

All local moved files passed SHA256 comparisons. Map: outputs/migrations/local_20260923/moves.json. Historical saved configs/results/checkpoint contents were not rewritten. The September23 research freeze remains unchanged outside the repository.

## Updated runnable paths

Config output roots for diversity, policy comparisons, ablations and preliminary/winner evaluations are under outputs. Search default and configured search destinations are under outputs/search. Legacy experiment wrappers, the background overnight launcher and the fallback loss log use outputs. New real overnight preprocessing uses data/processed/overnight/<run name>; tiny synthetic fixtures remain self-contained under outputs/smoke. Runnable Assay presets use data/processed/ec50_followup_v1/assay rather than a hardcoded Linux home path.

A missing historical Assay cache path ending in outputs_overnight_losses_20260913_102501/processed resolves with a warning to the migrated local cache when that cache exists. An existing path or unrelated path is never redirected. Config loading does not alter historical files. For a different cache, pass --processed-dir explicitly.

The .gitignore data rule is now root-anchored (/data/) so new source files under src/data are not accidentally ignored.

## Server migration completed

On 2026-09-24, an idle server was inventoried, the migration script was verified and dry-run, and 28 root-level entries were moved into `outputs/` or `data/processed/`. All 28 moves passed SHA256 before/after checks. The recovery map is `outputs/migrations/20260924_041342_199264/moves.json`. There are no remaining root-level `outputs_*` entries. Five named Assay studies are under `outputs/assay/`; older studies retain their original names under `outputs/legacy/`; the old reusable Assay cache is `data/processed/ec50_followup_v1/assay`.

The two records under `/home/lsantiago4/reu_runs` were separately moved and file-hash verified: the failed pilot is under `outputs/assay/baseline_study_v1/failed_pilot_20260923_01`, and the synthetic smoke is under `outputs/smoke/remote_validation/smoke_release_20260923_01`. The recovery map is `outputs/migrations/reu_runs_20260924.json`. The empty `/home/lsantiago4/reu_runs` directory was removed. No research result or checkpoint was deleted or overwritten.

The migration script preflighted destination collisions and symlinks, refused visible project Python jobs, and verified SHA256 before/after each move. It did not rewrite historical saved configs, results, or checkpoints; old absolute paths embedded in those records may refer to their former locations. Future jobs should use the canonical paths directly.

Windows local moves were performed with native PowerShell after resolving paths. The server script's --apply is deliberately Linux-only.

## Validation

135 existing tests and3new migration/cache tests passed. Real historical Assay config resolved to the new cache and loaded2572validation molecules. Synthetic CPU select completed all four policy combinations. Legacy parameter-search dry run uses outputs/smoke/tune. Source compilation and diff whitespace checks passed. CUDA checks remain for the server. No model parameters, scientific settings or data contents changed; no commit/push performed.
