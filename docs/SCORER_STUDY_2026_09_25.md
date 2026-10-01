# Scorer study preparation — September 25, 2026

## Completed locally

- Stable score sorting now resolves exact top-k ties by original edge order. Near ties can still respond to floating-point differences; this is not a claim of bitwise GPU/CPU identity or permutation-invariant tie resolution.
- Optional `scoring_mode=symmetric` averages forward/reverse endpoint logits before sigmoid. Endpoint atom/size feature blocks are swapped correctly. Ordered scoring remains the default for configurations without a mode. Parameter names/shapes are unchanged; historical checkpoints load strictly. Historical *exact inference* still requires the historical source release because tie handling changed globally.
- Tests cover the reproduced batch-composition fixture, symmetric scores/selections under renumbering, reconstructed predictions, classification gradients, state-dict compatibility, and invalid edge feature layouts.
- The sequential `main.py experiment scorer-study` workflow compares ordered and symmetric scoring at fixed settings and emits paired validation results. No test-evaluation flag is exposed.
- Selection runs now save `runtime_spec.json`: config, resolved per-view modes/ratios/seeds, optimizer groups, clip norm, policies, Python/package versions and numerical flags. This improves the current workflow; it is not yet a universal logging refactor of every historical entry point.
- Source packaging omits local AGENTS/skills, the SSH client, and Codex-named files. Credentials are never bundled. Historical verified releases are preserved.
- Full suite: **166 tests passed**. Two-arm local synthetic smoke: **two runs, three epochs each**, completed. Synthetic AUC is not model-quality evidence.

## Saved-model diagnostics

All 15 frozen checkpoints replayed locally on validation only with stable ties. Maximum pooled-AUC differences from the saved GPU metrics: learned 0.00001870, random 0.00000078, ERM numerical roundoff. These are diagnostic replays, not replacement benchmark results.

| Five-seed mean | Pooled AUC | Macro within-assay AUC | Pair-weighted within-assay AUC |
|---|---:|---:|---:|
| Learned | 0.8072 | 0.5502 | 0.5244 |
| Random | 0.7955 | 0.5437 | 0.5046 |
| ERM | 0.7727 | 0.5378 | 0.5128 |

Each split has 24 usable two-class assays and 22 excluded single-class assays. Within-assay pairs are only 1.068% of pooled positive-negative pairs. This descriptive decomposition does not establish leakage or explain the test gap causally. View/retention/numbering probes also completed on the same 100 connection-count-stratified validation molecules for all five learned checkpoints. These probes are not five independent datasets.

Artifacts: `outputs/assay/scorer_preparation_20260925/`: `group_summary.json`, `view_summary.json`, `validation_groups_stable_ties/`, `views_seed8101/` through `views_seed8105/`, test/smoke logs. Original checkpoints remain under `outputs/assay/audit_20260924/source_records/`.

## Server footprint

Access uses the local SSH helper, the user's credentials and trusted host key. No account, authorized key, Codex installation, service or persistent agent was added. A bounded inventory found no Codex-named entries at home top level or within project/release source trees, excluding data, outputs, virtual environments, caches and Git internals. This is not a whole-server forensic scan. Ordinary SSH/login records remain. REU source snapshots, launch logs, `.local/state/reu/pipeline.lock` and `.approved-releases` digest pins are intentional research bookkeeping, not independent access mechanisms. Older snapshots can contain project AGENTS/skills documentation; the new release excludes them without rewriting old verified manifests.

## Proposed sequential server batch — awaiting exact-command approval

Verified release: `/home/lsantiago4/reu_releases/REU_Scorer_Study_20260925_v1`.
Manifest SHA256: `e45843c27fbddff7e03ba488aa58381bc60b4285c8856ebd36908b56b78891fb` (already pinned; no hash argument needed).

Use existing `sugar_run` pane `%4`, observed idle. GPU 1 had 45,370 MiB free and 0% utilization at preparation; recheck before launching. No matching project training process was found. Launch smoke first and start the real study only if smoke succeeds. No automatic retry, parallel jobs, dependency installation or test-set evaluation.

### 1. GPU smoke

Two synthetic runs, three epochs each. Planning estimate 1–3 minutes, not a time limit.

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/REU_Scorer_Study_20260925_v1/scripts/reu_remote_job.py --record-dir /home/lsantiago4/REU_Summer_2026/outputs/smoke/scorer_study_20260925 --gpu 1 -- scorer-study --smoke
```

### 2. Real validation-only scorer study

Ten sequential runs: ordered and symmetric arms, matched seeds **8301–8305**, **100 epochs each** (1,000 epochs total). Both arms use stable ties. Primary independent/independent, secondary shared/independent; both validation policies and checkpoints saved. Five views, hidden dimension 128, three layers, batch 64, dropout 0.5, learning rate 0.0003, weight decay 0, edge ratio 0.5, candidate sampling 0.7, contrastive 0.1, diversity/IRM 0, no early stopping.

Prior five-seed learned pipeline took 95.9 minutes. Planning estimate **4–6 hours** for the two-arm study allows for extra symmetric scoring and shared-server variability; symmetric full-data timing is not yet measured. The approved job will continue if it exceeds this estimate.

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/REU_Scorer_Study_20260925_v1/scripts/reu_remote_job.py --record-dir /home/lsantiago4/REU_Summer_2026/outputs/assay/scorer_study_v1 --gpu 1 -- scorer-study --processed-dir /home/lsantiago4/REU_Summer_2026/data/processed/baseline_study_v1/assay --raw-path /home/lsantiago4/REU_Summer_2026/data/raw/drugood/lbap_core_ec50_assay.json
```

Results: `outputs/assay/scorer_study_v1/results/comparison.json`, with run records under `results/ordered/` and `results/symmetric/`. Interpret all five paired seeds, not the best seed. A positive validation change is not yet evidence of improved test AUC. Any eventual test evaluation requires a separately frozen decision and approval.
