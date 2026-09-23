# Parameter search review and test evaluation

All 24 trials COMPLETE, 48 training runs, 17 unique configurations. Search artifacts contain validation only; recorded source hashes match current local sources.

Reference trial0 mean validation .785680 (seed201 .783372, seed202 .787989). Best trial21 .804744 (seed201 .808923, seed202 .800566), +.019064 absolute AUC. Winning parameters repeated in trials16/20/21/22/23, means .803954/.799427/.804744/.793853/.799905; average .800377. These are repeated executions of the same two seeds, not ten independent seeds. Variation prevents treating the best .8047 as expected performance. TPE categorical proposals duplicated settings; only17 distinct combinations were explored. No individual parameter causal effects established.

Dumb It Down: promising enough to test; improvement on validation does not guarantee improvement on unseen domains. Freeze the chosen settings before reading test results.

Technical Explanation: lr .0003, weight decay0, dropout .5, edge ratio .5, fragment sample ratio .7. Contrast .1, diversity0, IRM0, five views/shared inference retained. Three fresh seeds3001–3003; max100epochs, existing early stopping, best individual validation AUC checkpoint. Report validation and test once after training each seed. Save checkpoints and histories. This retrains models because search did not retain checkpoints. Prior test mean .7456 over three different seeds is historical context, not a paired comparison.

Upload configs/parameter_winner_test_base.yaml, configs/parameter_winner_test.yaml, configs/parameter_winner_params.json, and retain current scripts/grid_search.py and source.

From server repository root inside tmux:

```bash
.venv-reu-server/bin/python scripts/grid_search.py eval --config configs/parameter_winner_test.yaml --params configs/parameter_winner_params.json --device cuda --dry-run
.venv-reu-server/bin/python -u scripts/grid_search.py eval --config configs/parameter_winner_test.yaml --params configs/parameter_winner_params.json --device cuda
```

Launch once; detach Ctrl+B then D. Output directory is fixed: outputs_parameter_winner_test. Do not relaunch into it after a run without choosing a fresh output_dir in the spec. Final summary: eval/eval_results.json, summary.ood_test.auc.mean/std/n and per-seed runs. Dry-run checks config assembly only. Local synthetic two-epoch evaluation passed, including finite validation/test metrics and saved checkpoint; no real test run performed locally.
