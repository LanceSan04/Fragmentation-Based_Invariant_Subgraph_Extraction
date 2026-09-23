# Assay: controlled shared/independent checkpoint selection

## What this tests

One training trajectory per seed, with both validation policies evaluated after every epoch. Save best_shared.pt and best_independent.pt separately. Use earliest epoch on tied AUC. Fixed100 epochs, no early stopping, no weight averaging, no greedy view selection. Existing Assay settings:lr .0003, decay0, dropout .5, edge ratio .5, fragment sampling .7, contrast .1/diversity0/IRM0, five views. Fresh seeds5001–5003. Training loss unchanged; independent still uses per-view top-k.

Evaluation preserves Python/NumPy/Torch RNG state so these extra evaluation passes do not advance training random generators. CUDA may still be nondeterministic across separate executions. Both policy selections within a run see the same actual training trajectory. Checkpoint files omit optimizer states: this runner does not implement resume.

Final comparison is a2x2 table: each policy-selected checkpoint evaluated under each inference policy. Validation is always reported; optional --evaluate-test loads test only after that seed's checkpoint selection has finished. No test-guided decisions. The experiment is exploratory on previously inspected test sets. Historical results also differ in early stopping, seeds, and potentially code revision; use the within-run comparison for conclusions.

## Upload

Upload scripts/run_dual_policy_selection.py to the same scripts directory on the server. Retain all current source updates from the inference-policy implementation, scripts/run_ec50_followup.py, scripts/run_overnight.py, scripts/grid_search.py, and configs/parameter_winner_test_base.yaml. Assay uses the existing cache configured there; --processed-dir can override its location without rebuilding data.

## Commands, inside tmux

```bash
cd /home/lsantiago4/REU_Summer_2026
.venv-reu-server/bin/python -u scripts/run_dual_policy_selection.py --dataset assay --smoke --evaluate-test --device cuda
```

Wait for COMPLETE. Synthetic metrics only. Then inspect the planned budget:

```bash
.venv-reu-server/bin/python scripts/run_dual_policy_selection.py --dataset assay --dry-run --evaluate-test
```

Launch once:

```bash
.venv-reu-server/bin/python -u scripts/run_dual_policy_selection.py --dataset assay --device cuda --evaluate-test --output-dir outputs_dual_policy_assay
```

This is3 training runs x100 epochs, with two validation forward paths per epoch. It may take longer than an equivalent single-policy run; no time estimate guaranteed. Detach Ctrl+B then D. Fresh output directory required; rerunning into an existing directory is rejected. Partial outputs are not resumable. Omit --evaluate-test for validation-only operation.

## Read results

Download the entire outputs_dual_policy_assay directory. Each seed folder contains config.json, history.json, best.json, best_shared.pt, best_independent.pt, and results.json. Root provenance.json captures source hashes. summary.json exists after all seeds complete.

Summary layout: summary[checkpoint-selection policy][inference policy][split], followed by mean/std/n (AUC across seeds).

- shared → shared → ood_test: current inference with its chosen checkpoint.
- shared → independent → ood_test: inference-only change at shared-selected checkpoint.
- independent → independent → ood_test: independent voting with its own chosen checkpoint.
- independent → shared → ood_test: cross-check shared inference at the other checkpoint.

Compare the second and third entries to isolate the added effect of checkpoint selection under independent validation. Compare first and second to isolate the prediction-policy change. The same interpretation applies to ood_val. Selection-based validation improvements are expected and do not themselves prove better test generalization.

## Verification

131 tests pass, including a controlled test forcing different winning epochs for the two policies and confirming both saved checkpoints. A real synthetic3-epoch run exercised both policies and all four final evaluations on validation/test. No full real-data training or server jobs launched locally.
