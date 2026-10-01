# Retained-atom training study — 2026-09-27

Question: does controlling reconstructed atom count improve validation generalization, and does learned edge ranking help when both arms use that rule?

| Arm | Hard selection | Message weights |
|---|---|---|
| reference | Learned top 50% fragment edges | Learned |
| atom_learned | Learned-ranked prefix closest to 65% original atoms | Learned |
| atom_random_rank | Random-ranked prefix closest to 65% original atoms | Learned |

Whole fragments are indivisible. For each nonempty eligible prefix we count distinct atoms; choose minimum distance to ceil(0.65 × original atoms), then fewer atoms, then fewer edges. This is an approximate target, not exact matching. Original edgeless molecules retain the existing full-graph fallback. Reconstruction does not add missing boundary bonds. Training eligibility still samples 70% of fragment edges; validation uses all eligible edges. Random priorities have an isolated training RNG and molecule-stable evaluation draws. The old random-selection baseline remains a separate, unchanged mode.

Five paired seeds 8401–8405, 100 epochs each, three arms (15 trajectories), no early stopping. Five views, 128 hidden units, three layers, batch 64, learning rate 0.0003, dropout 0.5, weight decay 0. Pooled contrastive coefficient 0.1; diversity and IRM zero. Ordered scorer. Independent voting for validation/checkpoint selection and final validation. No test split is loaded or evaluated. Ties retain the earliest checkpoint. Shared inference rejects the new selector rather than silently changing its meaning.

Preflight: synthetic three-epoch smoke per arm; real one-epoch pilot per arm. One GPU, one sequential pipeline, existing sugar_run session. No retry or expansion. Estimated full runtime 4–8 hours, to refine from pilot; time estimates are not termination limits.

Success screen: mean paired validation AUC gain at least 0.005 versus reference, positive in at least four of five seeds. This is a screening rule, not statistical significance or test-set evidence. Record all runs, including unfavorable outcomes. Do not pick seeds. Comparing two atom-budget arms still requires checking actual retained sizes: a mean fraction gap over 0.02 flags poor size matching.

Each seed stores config.json, runtime_spec.json, history.json, best.json, best_independent.pt, results.json, resources.json, retention.json. Retention covers the same 100 size-stratified validation molecules, five views each: original/selected atoms and bonds, fragment edges, connected components, overlaps, target residual and edgeless fallback. These sample summaries are descriptive, not population-weighted. Full run summary and retention_summary.json compare all seeds. Source and train/validation cache hashes are preserved. Fresh training is reproducible; resuming interrupted training including private sampling counters is not implemented.

Entrypoint: main.py experiment atom-budget-study. Local code: src/models/atom_budget_selection.py, FragmentCIGA.select_fragment_edges, src/experiments/atom_budget_study.py. Default edge selection and old checkpoints retain their parameter layout.

Remote outputs: /home/lsantiago4/REU_Summer_2026/outputs/assay/atom_budget_study_v1/{smoke,pilot,training}/results.

## Launch and verification

Implemented and launched on GPU 1 in the existing sugar_run session, September 27, 2026. The full study runs independently of the laptop. Source release: REU_Atom_Budget_20260927_v1; manifest a2432baf175f2437770d272c3dadd708c4358f6216decded06a03fbb1d1d0c7a.

178 local tests passed; CPU and server GPU synthetic smoke passed; three real-data one-epoch pilots passed, including restoring the selected checkpoint and reproducing its validation AUC. Pilot training/validation/checkpoint times: reference 13.61 s, atom_learned 12.67 s, atom_random_rank 13.39 s. Peak PyTorch allocated tensor memory was 253–265 MB (excludes CUDA context and other GPU allocations).

Pilot descriptive validation sample: 64.58% atoms retained by reference, 65.77% by learned atom target, 66.19% by random atom target. Learned/random gap 0.42 percentage points. These are size checks, not performance findings. Full results remain pending.

Monitor on server:

```bash
tail -f /home/lsantiago4/REU_Summer_2026/outputs/assay/atom_budget_study_v1/training/console.log
```

At completion, training/results/summary.json contains all five validation AUCs per arm and paired differences. training/results/retention_summary.json reports actual size matching. Do not restart the launch command while the study is running. Agent agreement and SSH skill files were unchanged.

## Results check: study stopped after the first run

The launcher exited with code 1 after approximately 12 minutes (07:24:48–07:36:58 UTC on September 27). Only reference seed 8401 finished its 100 training epochs. The remaining 14 trajectories did not start.

- Selected epoch: 23.
- Validation ROC-AUC at selection: 0.8182459174769385.
- Restored-checkpoint validation ROC-AUC: 0.8182490339067565.
- Absolute difference: 0.0000031164298180.
- Final epoch 100 validation ROC-AUC: 0.7878404699576165 (not the selected checkpoint).
- No test evaluation was performed.

The new study wrapper rejected the restored AUC because its tolerance was 0.000001. The traceback confirms this check caused the stop, after results.json had already been saved. This is not evidence that the atom-budget method failed: neither atom-budget arm ran. Small numerical changes in inference and near-tied prediction rankings could explain the discrepancy, but that explanation has not yet been verified with repeated predictions.

Recommended recovery: repeat validation on the existing checkpoint, inspect prediction/ranking differences, then replace the overly brittle verification with a justified tolerance plus recorded diagnostics. Preserve and reuse the completed reference trajectory and finish the other 14 runs under the original protocol, rather than selecting a different seed or retraining favorable outcomes. No retry was launched during this results check.

Downloaded JSON evidence: outputs/assay/atom_budget_study_v1/training/. Checkpoints remain on the server. There is no completed three-arm summary yet.

## Recovery and resumed training

The reference checkpoint was replayed three times on GPU and once on CPU. Validation AUCs: 0.8182505921, 0.8182490339, 0.8182505921, and 0.8182435802. One GPU repeat changed one molecule's probability by 0.0001953 (mean absolute change 8.20e-8), changing one positive/negative pair ordering. CPU versus the first GPU replay changed five pair orderings, with maximum probability change 0.0026253. These observations support allowing small numerical variation; they do not identify the exact cause of the historical epoch-23 discrepancy because those original predictions were not saved.

The first recovery verification stopped before training because a newly added maximum-probability tolerance of 1e-5 was also overly strict. It is preserved under recovery/. The corrected recovery imports the four saved replay records, verifies checkpoint/configuration identity, and uses explicit engineering gates: AUC difference <=1e-4 for every replay; same-device maximum absolute probability difference <=0.001 and mean absolute difference <=1e-5. CPU probability differences are recorded separately. These tolerances are engineering checks, not a claim of scientific equivalence or a selection criterion. No training settings or model weights were changed.

181 local tests passed, along with the local smoke and a complete import check using the actual saved checkpoint and replay artifacts. Recovery checks full 100-epoch history, best epoch, configuration, raw/train/validation hashes and unchanged core training/model source hashes. The existing seed8401 reference is copied with hashes into the new run; no retraining or seed replacement. Its missing retention diagnostics are generated before continuing.

Fresh release: REU_Atom_Budget_Recovery_20260927_v3.
Manifest: 57ba8912df9d902aa768b736fd45f4dc59dbdecb118776d9f648895a6fd1f3e0.
New output: /home/lsantiago4/REU_Summer_2026/outputs/assay/atom_budget_study_v1/resumed_training/results.
Remaining budget: 14 trajectories x 100 epochs, sequential GPU 1, independent validation, no test evaluation. Estimated 3–5 hours, without automatic termination at that estimate. Existing sugar_run session persists after laptop shutdown.

```bash
tail -f ~/REU_Summer_2026/outputs/assay/atom_budget_study_v1/resumed_training/console.log
```

The original failed run and both verification records remain preserved. Future result checks should use resumed_training/. No agent agreement or SSH skill changes.
