# Assay ERM matched to fixed-epoch fragment experiment

Upload updated scripts/run_dual_policy_selection.py. Existing source/config dependencies remain required.

```bash
cd /home/lsantiago4/REU_Summer_2026
.venv-reu-server/bin/python -u scripts/run_dual_policy_selection.py --model erm --dataset assay --smoke --evaluate-test --device cuda
```

After COMPLETE, launch once:

```bash
.venv-reu-server/bin/python -u scripts/run_dual_policy_selection.py --model erm --dataset assay --seeds 5001 5002 5003 --epochs 100 --evaluate-test --device cuda --output-dir outputs_assay_erm_matched_5001_5003
```

Three new training runs, five full-molecule classifiers each, prediction loss only.100epochs fixed; noearlystop. Classifier hyperparameters match current Assay fragment config. Same seeds as original outputs_dual_policy_assay. Matching seeds does not imply identical optimizer trajectories across architectures. One validation policy suffices because shared/independent ERM predictions are equivalent. Best validation checkpoint saved, test evaluated after selection.

Final results: outputs_assay_erm_matched_5001_5003/summary.json, summary.shared.shared.ood_test.mean/std/n. Histories and best_shared.pt under each seed folder. Fresh directories required; no automatic resume. This is an ensemble-size/settings-matched control, not separately tuned optimal ERM. Fragment method additionally uses contrastive loss; interpretation is combined method versus prediction-only full graph.

Local synthetic ERM end-to-end training/evaluation passed and existing dual-selection regression test passed. Run server CUDA smoke before full jobs. No real training launched locally.

Research protocol: choose seeds/budget before results, keep every valid run, report mean/SD and per-seed scores. Do not stop once a favorable test result appears or exclude valid weak runs. Improve methods using validation and predefined comparisons. Selecting or repeatedly tuning against test outcomes makes that test set part of development; acknowledge existing exposure.
