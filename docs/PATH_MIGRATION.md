# Output and cache migration — September 23, 2026

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

## Server steps

Stop project training/diagnostic jobs first. Extract REU_Path_Migration_Update.zip into the existing server project, preserving directories and replacing matching code/config files. This is a partial update, not a full installation.

Preview:

```bash
cd /home/lsantiago4/REU_Summer_2026
.venv-reu-server/bin/python scripts/migrate_workspace_paths.py
```

Apply:

```bash
.venv-reu-server/bin/python scripts/migrate_workspace_paths.py --apply
```

Then verify:

```bash
.venv-reu-server/bin/python main.py experiment select --dataset assay --smoke --evaluate-test --device cuda
```

The migration preflights destination collisions and rejects symlinks. It refuses visible project Python jobs, does not merge/delete/overwrite results, and verifies SHA256 before/after each move. It records a recovery manifest under outputs/migrations. Missing legacy folders are skipped. Unknown root outputs_* items go under outputs/legacy with original names. A collision is a stop condition requiring inspection; do not delete either directory to bypass it. Partial interrupted migrations leave recorded moves; inspect the manifest before retrying. Renames require a single filesystem; a cross-filesystem rename error stops the process without copy/delete fallback.

Windows local moves were performed with native PowerShell after resolving paths. The server script's --apply is deliberately Linux-only.

## Validation

135 existing tests and3new migration/cache tests passed. Real historical Assay config resolved to the new cache and loaded2572validation molecules. Synthetic CPU select completed all four policy combinations. Legacy parameter-search dry run uses outputs/smoke/tune. Source compilation and diff whitespace checks passed. CUDA checks remain for the server. No model parameters, scientific settings or data contents changed; no commit/push performed.
