# Assay cache and pilot commands — completed 2026-09-24

These commands replaced the earlier `/home/lsantiago4/reu_runs/...` proposals and were approved and completed sequentially in the existing `sugar_run` session. Do not rerun them against the same record directories. The verified source release is `/home/lsantiago4/reu_releases/REU_Baseline_Study_Source_20260923_v2`, manifest SHA256 `9a9612bd8e2dac29e95e0c3d53321c643aac0139acc5785974c91c90f9fc6515`.

## 1. Prepare a fresh Assay cache

No training; CPU; three native DrugOOD splits; estimate 5–20 minutes. The raw JSON is read from the existing server checkout. The new cache is under `data/processed/`, and launcher logs are under `outputs/`.

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python \
  /home/lsantiago4/reu_releases/REU_Baseline_Study_Source_20260923_v2/scripts/reu_remote_job.py \
  --manifest-sha256 9a9612bd8e2dac29e95e0c3d53321c643aac0139acc5785974c91c90f9fc6515 \
  --record-dir /home/lsantiago4/REU_Summer_2026/outputs/assay/baseline_study_v1/cache_preparation \
  --gpu cpu \
  -- select --dataset assay --prepare-cache \
  --raw-path /home/lsantiago4/REU_Summer_2026/data/raw/drugood/lbap_core_ec50_assay.json \
  --processed-dir /home/lsantiago4/REU_Summer_2026/data/processed/baseline_study_v1/assay
```

## 2. Repeat the real-data resource pilot

Only after step 1 succeeds and the cache hash/counts are verified. One fragment-model seed (8101), two epochs, OOD validation only, no test evaluation; GPU 1, subject to a fresh occupancy check; provisional estimate 2–10 minutes. If the estimate is exceeded, let the approved job complete and report the overrun. No automatic retry.

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python \
  /home/lsantiago4/reu_releases/REU_Baseline_Study_Source_20260923_v2/scripts/reu_remote_job.py \
  --manifest-sha256 9a9612bd8e2dac29e95e0c3d53321c643aac0139acc5785974c91c90f9fc6515 \
  --record-dir /home/lsantiago4/REU_Summer_2026/outputs/assay/baseline_study_v1/resource_pilot \
  --gpu 1 \
  -- select --dataset assay --model fragment --seeds 8101 --epochs 2 \
  --raw-path /home/lsantiago4/REU_Summer_2026/data/raw/drugood/lbap_core_ec50_assay.json \
  --processed-dir /home/lsantiago4/REU_Summer_2026/data/processed/baseline_study_v1/assay
```

The cache step succeeded with 4,540 train, 2,572 OOD validation, and 2,490 OOD test molecules. The pilot succeeded in about 38 seconds; epoch-2 OOD-validation AUC was 0.7513 shared and 0.7633 independent, with no test evaluation. The subsequent ERM tuning and matched final-arm commands must use these same cache/source paths and fresh `outputs/assay/baseline_study_v1/` subdirectories. They require separate exact-command approval. Older root-level `outputs_*` folders were migrated separately; see `docs/PATH_MIGRATION.md`.
