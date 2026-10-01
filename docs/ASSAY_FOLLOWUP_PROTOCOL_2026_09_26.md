# Assay follow-up: frozen protocol

Prepared before any new test evaluation. User authorized all four recommendations and waived the per-command approval question for this task only. AGENTS and skill files remain unchanged.

1. Retain the ordered scorer with pooled contrastive loss 0.1 as the reference. Five views; edge ratio 0.5; training sample ratio 0.7; diversity and IRM off. Reuse all seeds 8301â€“8305 and their completed 100-epoch trajectories. Do not retrain, tune or choose seeds in this batch.
2. Replay the pooled, off and per-view arms on OOD validation. Confirm every checkpoint's expected AUC within 1e-4 before loading test for predictions. Record pooled, macro-assay and within-assay pair-weighted AUC. Single-class assays have undefined within-assay AUC and are excluded, not assigned zero. Evaluate both saved selection checkpoints using independent inference.
3. Inspect frozen independent-selected models: 100 validation molecules per seed/arm at edge ratio 0.5, stratified by original atom count. For pooled reference only, measure inference geometry at edge ratios 0.3, 0.5, 0.7 on those same molecules. Also use 40 stratified validation molecules, two deterministic draws, and all nine edge/sample combinations (edge 0.3/0.5/0.7; sample 0.5/0.7/0.9). Only the sampler is put into training mode; encoders and dropout stay in evaluation mode. These probes measure sampling sensitivity, not training performance. Top-k counts refer to full fragment-edge counts and are capped by eligible counts; ratios do not multiply. Record original/retained atom and bond counts, disconnected components, eligible/selected connection counts, and per-view Jaccard overlap. Prespecified descriptive atom bins: <=25, 26â€“50, >=51. No sampled-subset AUC or ratio-based model selection.
4. Freeze a validation-only decision before test: keep pooled as the primary reference; evaluate all three predeclared arms and all five seeds with unmodified ratios. Primary selection/inference = independent/independent; secondary = shared/independent. Thirty checkpoint evaluations total, zero training epochs. Do not nominate a new winner or start follow-up experiments based on test results. Record every result, mean, sample SD, and paired seed differences. Historical assay test inspection makes this exploratory rather than a pristine held-out assessment.

Existing validation means (primary): pooled 0.810994; per-view 0.810061; off 0.807509. Per-view changes both pair membership and reduction/scale. These differences do not establish a clear improvement.

The frozen runtime plan will contain checkpoint/config/source/data hashes. A mismatch, failed parity check or exception stops the pipeline. Fresh output directories, one job in sugar_run, no auto retries, no packages/accounts/services installed on the server. Estimated real evaluation/diagnostic time 15â€“45 minutes; elapsed time is not a termination condition. After this finite batch, interpret findings before deciding whether a new retention-controlled training study is warranted.

## Exact server commands executed

### smoke

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/REU_Assay_Followup_20260926_v1/scripts/reu_remote_job.py --record-dir /home/lsantiago4/REU_Summer_2026/outputs/smoke/assay_followup_v1 --gpu 1 -- assay-followup --smoke --pooled-dir /home/lsantiago4/REU_Summer_2026/outputs/smoke/contrastive_study_v1/results/off --off-dir /home/lsantiago4/REU_Summer_2026/outputs/smoke/contrastive_study_v1/results/off --per-view-dir /home/lsantiago4/REU_Summer_2026/outputs/smoke/contrastive_study_v1/results/per_view --processed-dir /home/lsantiago4/REU_Summer_2026/outputs/smoke/contrastive_study_v1/results/off/processed
```

### real

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/REU_Assay_Followup_20260926_v1/scripts/reu_remote_job.py --record-dir /home/lsantiago4/REU_Summer_2026/outputs/assay/assay_followup_v1 --gpu 1 -- assay-followup --pooled-dir /home/lsantiago4/REU_Summer_2026/outputs/assay/scorer_study_v1/results/ordered --off-dir /home/lsantiago4/REU_Summer_2026/outputs/assay/contrastive_study_v1/results/off --per-view-dir /home/lsantiago4/REU_Summer_2026/outputs/assay/contrastive_study_v1/results/per_view --processed-dir /home/lsantiago4/REU_Summer_2026/data/processed/baseline_study_v1/assay --raw-path /home/lsantiago4/REU_Summer_2026/data/raw/drugood/lbap_core_ec50_assay.json
```

Source release manifest: `a7f78947dff91387025c095afc3f0c169fbce875d17f0c9599d35bb96a6263d0`. Local checks: 171 tests passed and synthetic full-workflow smoke passed.
