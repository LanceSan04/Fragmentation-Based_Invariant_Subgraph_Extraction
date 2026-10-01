---
name: reu-remote-ssh
description: Connect to the REU shared research server with password SSH from local environment variables, inspect it safely, and run explicitly approved remote research pipelines under the project's resource agreement.
---

# REU remote SSH

Read the repository root `AGENTS.md` and `docs/REMOTE_WORKING_AGREEMENT.md` first. This skill applies only to work on this REU server. The remote account is unprivileged and the server is shared.

## Connection

- Use `scripts/reu_ssh.py` with this repository's `.venv` Python. It reads `SSH_HOST`, `SSH_USER`, and `SSH_PASSWORD` from the inherited environment or the ignored repository `.env`. `paramiko` and `python-dotenv` are local environment dependencies; install requirements-remote.txt into the local project virtual environment if absent, never globally. The training server does not need these SSH client dependencies.
- The helper requires a matching trusted key in the user's `~/.ssh/known_hosts`. Never use `AutoAddPolicy`, `StrictHostKeyChecking=no`, `sshpass` on a command line, or an unverified `ssh-keyscan` result. If no trusted key exists, obtain the fingerprint from a separate trusted source and verify it before enrolling the key. Stop on a mismatch.
- For the authorized connection validation: `.venv/Scripts/python.exe scripts/reu_ssh.py hello` on Windows. It prints only a fixed greeting plus the remote command's exit status. No GPU allocation or project files are involved.
- For a later remote command, use `scripts/reu_ssh.py run -- <program> <arg>...`. The helper quotes each argument for the remote POSIX shell and does not expose the password in process arguments. Keep commands within the logged-in user's home for project file access. It does not bypass the approval requirements below.

## Before remote work

1. Confirm the requested outcome and whether the proposed action is a pipeline. A pipeline includes preprocessing, smoke training, full training, compute-heavy evaluation, search, sweep, migration or a background job. Before **each** pipeline, show the user the exact command and device, number of runs/epochs, estimated duration, output path and the continuation plan if the estimate is exceeded; obtain approval. Include dataset/config when needed to understand the run. State that Codex will not stop an approved long-running job because of elapsed time. A dry-run or an earlier pipeline approval does not authorize the next pipeline.
2. For the first use, confirm server policy, scheduler, home/project paths, host-key trust and current jobs. A bounded read-only inventory after the authorized hello-world connection check is already approved. Query system resource totals and GPU state only as needed. Do not inspect other users' files. If a scheduler exists, use its allocation rules.
3. Check that no compute job from this project/user is already running. One job maximum across all agents/sessions; child workers are part of that job. Never automatically launch on both GPUs. Choose a GPU consistent with server policy and current occupancy, and use `CUDA_VISIBLE_DEVICES` or the scheduler allocation. Record the chosen device.
4. Agree a per-task total budget. Baseline proposals in the agreement are **provisional**. Measure real CPU, RAM, VRAM and epoch runtime with an approved representative pilot before claiming they are safe limits. A synthetic smoke test does not measure full-training needs. Avoid altering scientific settings to fit the budget without user agreement.
5. Use a fresh path under the remote user's home. No sudo, privilege escalation, system package or driver changes, access outside the user's home for project files, or deletion/overwrite of existing research artifacts. Resolve symlinks before file operations. Do not install dependencies or upload code without task authorization.
6. For overnight work, use the existing user-owned `sugar_run` tmux session, subject to any mandatory scheduler policy so the job can continue without a live SSH connection. Record the job/session identifier and output path. By default, check progress when the user asks; set up a quiet scheduled monitor only if requested. If the duration estimate is exceeded, let the approved job continue and report the overrun at the next check. If occupancy, memory or disk growth becomes concerning, alert the user and start no new work; do not terminate the running job. The user will stop long-running jobs when desired. Preserve logs/checkpoints and ask before a larger retry, extra trials or another pipeline. No automatic retry or parallelism. Treat an SSH disconnect as an unknown job state; reconnect and inspect before any relaunch.

## Terms for launch planning

- **Connection check:** fixed `hello world`, no research compute.
- **Synthetic smoke:** a small end-to-end pipeline with artificial molecules and reduced settings, often 2–3 epochs. It validates code paths, not model quality or real-data memory.
- **Resource pilot:** representative real-data run to estimate peak memory and runtime. It requires pipeline approval.
- **Full training:** one approved real-data model trajectory with specified config, seed, epochs, validation and checkpoint policy.
- **Search or sweep:** multiple full-training trajectories. Count all repeats and seeds; sequential execution is still potentially expensive.

## Release and guarded launch

Read docs/REMOTE_OPERATIONS.md. Build a source-only ZIP with scripts/reu_release.py; it deliberately excludes .env, credentials, datasets and results. Record the printed manifest SHA256. Upload/extract only after authorized transfer, into a fresh directory under the user home; verify the expected hash before proposing an experiment. Code/config hashes include dirty edits and are more precise than Git HEAD alone. Never overwrite an active release.

Use scripts/reu_release.py verify --sha256 HASH --pin once after uploading a new release. The pin lives outside the release under `~/reu_releases/.approved-releases/`, so it protects the exact approved source while avoiding a long hash in every later launch command. Use scripts/reu_remote_job.py inside the existing `sugar_run` tmux session for approved main.py experiments. Provide a fresh --record-dir and one approved --gpu (cpu, physical index, or allocated to preserve a single scheduler-provided CUDA_VISIBLE_DEVICES). The launcher reads the pin and verifies the release before starting. Older unpinned releases retain an optional explicit --manifest-sha256 argument. Workflow arguments follow --; the launcher supplies --device and --output-dir. One account-wide advisory lock covers all participating releases. Inspect manual jobs too; the lock cannot stop launches that bypass it. Child workers belong to the approved job. Source verification, PID, exact command, device, log, results and terminal exit status are retained in the record directory. Do not infer approval from the mere availability of this tool.

Approval can cover one explicit sequential batch; do not re-ask per seed or routine step within its scope. Do not silently expand it. Migrations/non-main commands still require their own scoped approval and no competing project job; the guarded experiment launcher does not claim to support arbitrary shell commands. On disconnect, inspect tmux, launch.json, logs and process state before deciding whether a new pipeline is needed.

Remote pipelines must use the existing `sugar_run` session. Attach using `tmux attach -t =sugar_run`; confirm the target pane is idle before entering commands. Never send commands into a busy training pane or use Ctrl+C to make room. Do not create a different session or automatically recreate a missing sugar_run session. Report a missing session; read-only checks may run outside it. The launcher checks TMUX_PANE against tmux session_name, rather than trusting a shell variable alone.
