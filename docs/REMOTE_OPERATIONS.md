# Remote operations: verified releases and one approved pipeline

These commands are templates, not standing authorization to upload code or run a remote job. Inspect current jobs, GPU occupancy and lab policy; present the concrete command, dataset, device, run/epoch count, estimated duration and fresh output location before a new pipeline. Approval of a specific sequential batch covers its individual seeds. Elapsed time does not terminate an approved run.

## Local setup

Only if missing, install optional client packages into the local venv:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-remote.txt
```

Continue using scripts/reu_ssh.py for the authorized hello/read-only commands. It requires a trusted known_hosts match and reads credentials from ignored .env/environment; do not expose credentials. The helper itself is not a sandbox. A dropped connection leaves job state unknown; inspect before relaunching.

## Prepare an exact source release

```powershell
.venv/Scripts/python.exe scripts/reu_release.py build --output outputs/transfers/reu_release_YYYYMMDD_unique.zip
```

The ZIP includes whitelisted source/config/tests/skill/license files and release-manifest.json. It excludes the actual .env, data, venv and outputs. Review source for hardcoded secrets before sharing; the whitelist is not a secret scanner. Record the printed manifest SHA256 in the experiment proposal. Existing ZIPs are never overwritten. Git HEAD is contextual; file hashes capture uncommitted changes too.

After authorization, use WinSCP to upload/extract this ZIP to a **fresh** directory such as /home/lsantiago4/reu_releases/RELEASE_ID. Preserve paths. Do not upload over an active checkout. No general upload/delete tool is added here deliberately: transfer scope remains explicit and reviewable. Never copy the local .env to the server.

Verify on the server using the existing server interpreter (replace paths and hash with the concrete approved values):

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/RELEASE_ID/scripts/reu_release.py verify --sha256 EXPECTED_MANIFEST_SHA256 --pin
```

Verification rejects missing, changed and extra source/config files within the included scope. `--pin` saves the verified manifest hash once under `~/reu_releases/.approved-releases/RELEASE_ID.sha256`, outside the source release. Future launches read that pin automatically, recheck every release file, and record the hash in `launch.json`; the long hash need not be repeated in each command. A different pin cannot silently replace it. Older releases without a pin still accept the explicit `--manifest-sha256` argument. Data, environment and generated artifacts are outside this manifest. Keep releases unchanged during jobs; verification is point-in-time, not filesystem write protection.

## Approved launch

After the concrete pipeline is approved, attach to your existing session with `tmux attach -t =sugar_run` and use an idle pane. Do not create another session. The example below is a template for one real Assay seed, not an instruction to start it now:

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/RELEASE_ID/scripts/reu_remote_job.py --record-dir /home/lsantiago4/REU_Summer_2026/outputs/assay/STUDY_ID/APPROVED_UNIQUE_RUN --gpu APPROVED_GPU_INDEX -- select --dataset assay --seeds APPROVED_SEED --epochs APPROVED_EPOCHS --processed-dir /home/lsantiago4/REU_Summer_2026/data/processed/STUDY_ID/assay
```

No test flag is supplied by default. Use --evaluate-test only if in the approved protocol. The launcher adds --device and --output-dir; do not repeat them. Select/tune/sweep/diagnostics may have different cache argument names; use the exact workflow help before approval. New release folders do not contain datasets, so give explicit approved cache paths. Searches that preprocess also need their documented raw data available; do not blindly use this select example for them.

--gpu cpu hides GPUs. A physical index sets CUDA_VISIBLE_DEVICES; within the program cuda:0 denotes the selected GPU. Under a scheduler, use --gpu allocated to preserve an existing single-device CUDA_VISIBLE_DEVICES assignment. This is device selection, not a reservation or quota. Do not override scheduler assignments.

The per-account lock ~/.local/state/reu/pipeline.lock excludes participating launches across all releases and sessions. Do not delete this lock file or use another lock path to bypass it. OS locks release when holders exit; a leftover filename is normal. The child inherits the lock so it stays held if the parent exits while that child survives. Avoid manual background descendants: the pipeline is expected to keep children foreground and wait for them. Direct unguarded jobs do not obey this lock; manual inventory remains necessary. Existing Python processes in the current project are additionally checked, but this is not a full account scheduler.

A fresh record directory stores source-manifest.json, launch.json, console.log and results/. launch.json records exact command, PID, GPU, timestamps and final exit code. It stores only selected metadata, not environment variables or SSH credentials. A launcher interruption can leave status=running/starting; verify PID and tmux before deciding the job failed. A running status file is not proof of liveness. Tail console.log for training progress. The launcher does not retry or kill jobs for exceeding estimates.

Keep new run records under the existing project `outputs/<dataset>/<study_id>/<run_name>/` tree. Keep reusable processed splits under `data/processed/<study_id>/<dataset>/`. The verified source release remains under the separate `~/reu_releases/` directory so runs cannot overwrite code they depend on. The two earlier `~/reu_runs/` records were moved with file-hash verification on 2026-09-24; `~/reu_runs/` was removed. The recovery map is `outputs/migrations/reu_runs_20260924.json`.

## Validation status

Local unit tests cover release verification/tampering, excluded .env and generated data, path containment, and lock acquire/conflict behavior with a mocked POSIX lock. The first source release was uploaded and verified on 2026-09-23; its synthetic CPU smoke completed in `sugar_run`. A subsequent approved two-epoch real-data pilot exited before training because the old cache failed the preprocessing-hash check. The second source release (`REU_Baseline_Study_Source_20260923_v2`, manifest SHA256 `9a9612bd8e2dac29e95e0c3d53321c643aac0139acc5785974c91c90f9fc6515`) prepared a fresh Assay cache in `data/processed/baseline_study_v1/assay` with 4,540 train, 2,572 OOD validation, and 2,490 OOD test molecules. Its approved seed-8101 two-epoch GPU-1 pilot completed in about 38 seconds. The epoch-2 OOD-validation AUC was 0.7513 with shared inference and 0.7633 with independent inference; no test evaluation was requested. The run is under `outputs/assay/baseline_study_v1/resource_pilot/`. Linux lock contention, tmux persistence across disconnect, and real-data GPU peak use remain unmeasured.

## Fixed session preference

All remote pipelines use the existing `sugar_run` session. The guarded launcher rejects missing/wrong tmux sessions before starting a job and records the verified session name. Read-only SSH checks do not need tmux. If the session is missing, stop and report it rather than create a replacement. Do not interrupt an occupied pane. If lab policy requires a scheduler, resolve that explicitly before running; this preference does not override allocations.
