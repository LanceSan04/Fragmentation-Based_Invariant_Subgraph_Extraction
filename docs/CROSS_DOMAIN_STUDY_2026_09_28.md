# Frozen cross-domain study — 2026-09-28

Status: user approved the pilot; it launched on 2026-09-28 at 06:35:52 UTC in `sugar_run`, GPU 1, PID 458788. Preprocessing started. No new scientific test results yet.

Launch correction: v1 was rejected by the guarded launcher's workflow allowlist before any pipeline or output directory was created. Added `cross-domain` to that allowlist and a regression test (12 launcher tests passed), then staged and pinned a fresh `REU_Cross_Domain_20260928_v2` release. Its manifest SHA256 is `b90b37d5548f9499b2111dc0b8990d7659352f8dc5d5544f1984f38df527fac9`. The actual command is the pilot command below with only the source-release suffix changed from v1 to v2; dataset settings, output path, GPU and compute budget are unchanged. The running source snapshot is immutable; this local status update does not modify it.

## Question

Does the current fragment model transfer beyond EC50 Assay, and how does it compare with random fragment selection and a full-molecule model? This is a transfer study using fixed Assay hyperparameters. It is not dataset-specific tuning or an attempt to reproduce GOOD's published backbone/features.

## Conditions

| Model | Input to each of five classifiers | Auxiliary objectives | Checkpoint / evaluation |
|---|---|---|---|
| Learned | Learned fragment-edge selection with learned message weights | Pooled contrastive 0.1 | Independent / independent |
| Random | Random fragment-edge selection, selected message weights equal to 1 | Pooled contrastive 0.1 | Independent / independent |
| ERM | Full molecular atom graph | Prediction loss only | Full-graph ensemble / full-graph ensemble |

Random replaces both learned ranking and learned gating. It does not isolate ranking alone and is not the recent atom-budget random-ranking control that retained learned gates. ERM shares the classifier architecture, number of classifiers, optimizer settings, data and training budget; it is a matched control, not a separately tuned best-possible ERM. Learned versus ERM also changes contrastive training, so that contrast does not isolate fragmentation alone.

All datasets use the frozen config in `configs/cross_domain_reference.yaml`: 5 group-graph views, hidden dimension 128, 3 GNN layers, batch size 64, learning rate 0.0003, dropout 0.5, weight decay 0, edge ratio 0.5, training fragment sample ratio 0.7, diversity 0 and IRM 0. The old Size CLI overrides are not applied. We preserve existing selection/sampling semantics; the ratios are fractions of fragment connections, not guaranteed fractions of atoms.

Full studies use seeds 8701–8705 and 100 epochs per trajectory, with no early stopping, weight averaging or AUC smoothing. Select the highest OOD-validation ROC-AUC; exact ties retain the earlier epoch. Each model is freshly trained on its dataset's training split; weights are not transferred from Assay. Report all five seeds, mean, sample standard deviation and paired learned-minus-control differences. Do not promote one favorable seed or claim significance from five seeds alone.

## Datasets and staging

| Dataset/split | Training molecules | OOD validation | OOD test | Full trajectories |
|---|---:|---:|---:|---:|
| EC50 Scaffold | 2,570 | 2,532 | 2,533 | 15 |
| EC50 Size | 4,684 | 2,313 | 2,398 | 15 |
| GOOD-HIV Scaffold, covariate | 24,682 | 4,113 | 4,108 | 15 |
| GOOD-HIV Size, covariate | 26,169 | 2,773 | 3,961 | 15 |

1. Prepare fresh caches and measure one full epoch for each model on each dataset: 12 one-epoch trajectories, seed 8700, validation only. The pilot seed is excluded from scientific summaries. Preprocessing includes test records but no test predictions or AUC are computed.
2. After inspecting pilot errors, memory and timings, run EC50 Scaffold and Size: 30 trajectories. All training and validation replays complete before any test evaluation begins.
3. Run GOOD-HIV Scaffold: 15 trajectories. Review data/protocol correctness before the additional 15 GOOD-HIV Size trajectories. GOOD runs need measured runtime estimates before their launch proposals.

The fixed settings are intentionally a transfer test. Weak results do not prove the architecture can never work; later tuning must give the baselines comparable tuning effort. Earlier EC50 test results have been inspected, so those datasets are not pristine holdouts. Keep GOOD test outcomes out of later selection decisions as well.

## GOOD import and validation

Official source: https://github.com/divelab/GOOD and https://good.readthedocs.io/en/latest/_modules/GOOD/data/good_datasets/good_hiv.html . The adapter is original project code; it reads the MIT-licensed dataset archive without importing the GOOD package or copying its training implementation.

Downloaded archive: `data/raw/good/GOODHIV.zip` (37,564,781 bytes). Official download URL: https://drive.google.com/file/d/1CoOqYCuLObnG5M0D8a2P2NyL61WjbCzo/view . SHA256 of the downloaded archive: `767eebf62bda13a96da9507ec80391adfd2bf3f1b33aaff1f8ff3064aaaed4ee`. This digest pins the bytes inspected locally; it is not a publisher-signed checksum.

Re-download locally if needed, using the optional local downloader (`gdown==5.2.0`):

```powershell
.venv/Scripts/python.exe -m gdown 1CoOqYCuLObnG5M0D8a2P2NyL61WjbCzo --output data/raw/good/GOODHIV.zip
```

No new training-server dependency is needed. `src/data/good_hiv.py` validates the archive hash before deserializing, checks binary labels and disjoint source indices, and checks SMILES atom ordering, elements and bond connectivity against each stored graph. It maps official `val` to `ood_val`, `test` to `ood_test`, and retains original source indices, positions, domain/environment IDs and SMILES. It never regenerates splits or silently drops failed molecules. Whole-dataset fragmentation is still checked by the remote preparation step.

Our 46-dimensional atom and 14-dimensional bond features are recomputed from those SMILES. GOOD's stored 9/3 categorical features are not fed unchanged to our model. Comparisons among our three arms use identical features; published GOOD numbers may additionally differ in encoder, features and tuning budget.

Both domain archives were checked across all five official splits. No repeated source indices or identical canonical SMILES crossed those splits. This does not establish absence of close chemical analogues. OOD-test positives are only 81/4,108 (Scaffold) and 63/3,961 (Size); accuracy can look high from negative-class predictions alone. Primary scoring remains ROC-AUC.

## Local verification

- 48 focused unit/regression tests passed, covering split mapping, label/topology mismatches, source overlap, archive tampering, frozen settings, random-control semantics, paired summaries, release safeguards, and the rule that a failed validation replay prevents test loading.
- `outputs/smoke/cross_domain_v2/`: 12 synthetic trajectories, two epochs each, across all four dataset settings and three arms; checkpoint restoration and synthetic test evaluation passed.
- `outputs/smoke/cross_domain_real_v1/`: 12 one-epoch local CPU checks using eight real training and eight real validation molecules per dataset (four per class, up to 80 atoms), full five-view/128-hidden/3-layer model; no test evaluation. Exact selected records and source hashes are retained. These deliberately small balanced samples are smoke checks, not representative performance or server resource measurements.
- `outputs/good_hiv/adapter_validation_v1/report.json`: official split counts, hashes and canonical-overlap checks for both GOOD domains.
- The earlier synthetic `cross_domain_v1` attempt exposed a NumPy boolean JSON-serialization error before test evaluation; it was fixed and replaced by the successful v2 run. Preserve the failed record as diagnostic history.

## Outputs and provenance

All commands route through `main.py experiment cross-domain`. New launcher records live under `outputs/cross_domain/frozen_transfer_v1/`, reusable caches under `data/processed/frozen_transfer_v1/<dataset>/`, and GOOD raw data under `data/raw/good/`. No root-level `outputs_*` directories are created.

Each run stores full effective config, model runtime specification, per-epoch training components/validation AUC, selected epoch and checkpoint. Batch-level records store source/config hashes, raw/cache hashes, software versions, the frozen plan, per-run timing and peak allocated CUDA memory, validation replay differences, frozen checkpoint hashes, and `comparison.json`. Peak allocated CUDA memory is not total device occupancy or CPU RAM. Cache reuse requires matching hashes and preprocessing settings. A failed check stops the batch and preserves artifacts; there are no automatic retries.

## Exact proposed remote pilot

Use the existing idle `sugar_run` pane, one pipeline, physical GPU 1 (recheck availability immediately before launch). Verified source release: `/home/lsantiago4/reu_releases/REU_Cross_Domain_20260928_v1`. Source ZIP excludes credentials, local SSH helper, agent instructions, and generated results. The GOOD archive is transferred separately into the raw-data tree.

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/REU_Cross_Domain_20260928_v1/scripts/reu_remote_job.py --record-dir /home/lsantiago4/REU_Summer_2026/outputs/cross_domain/frozen_transfer_v1/resource_pilot --gpu 1 -- cross-domain --stage pilot --prepare-cache --datasets ec50_scaffold ec50_size good_hiv_scaffold good_hiv_size --raw-root /home/lsantiago4/REU_Summer_2026/data/raw --cache-root /home/lsantiago4/REU_Summer_2026/data/processed/frozen_transfer_v1
```

Scope: fresh strict preprocessing for four datasets, then 12 one-epoch trajectories sequentially; seed 8700, five views, no test evaluation. Initial planning estimate: 20–60 minutes, with substantial uncertainty from first-time GOOD fragmentation. Exceeding the estimate will not terminate an approved job or authorize extra runs. Stop on an error rather than retrying. Do not automatically advance to full studies after this pilot without their exact-command approval.

After successful pilot review, the EC50 full-study command is:

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/REU_Cross_Domain_20260928_v1/scripts/reu_remote_job.py --record-dir /home/lsantiago4/REU_Summer_2026/outputs/cross_domain/frozen_transfer_v1/ec50_study --gpu 1 -- cross-domain --stage study --datasets ec50_scaffold ec50_size --raw-root /home/lsantiago4/REU_Summer_2026/data/raw --cache-root /home/lsantiago4/REU_Summer_2026/data/processed/frozen_transfer_v1 --evaluate-test
```

This second command is a proposal, not a launched job: 30 trajectories x 100 epochs, followed by frozen-checkpoint OOD test evaluation. GOOD Scaffold/Size study commands use the same workflow with their dataset names and distinct record directories. Their full-run budget and time estimate will be presented after the resource pilot.
