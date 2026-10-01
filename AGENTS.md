# REU Summer 2026 agent working agreement

This file governs work in this repository. Read [docs/REMOTE_WORKING_AGREEMENT.md](docs/REMOTE_WORKING_AGREEMENT.md) before using the shared research server, and use [skills/reu-remote-ssh/SKILL.md](skills/reu-remote-ssh/SKILL.md) for the connection procedure. Existing project scientific and Git agreements still apply.

## Remote authority

- A user request to develop code locally does not authorize running it on the server. **Before starting each remote pipeline**, present the exact command and scope and ask the user. A pipeline includes preprocessing, synthetic smoke training, real training, evaluation that uses substantial compute, parameter search, sweep, migration, or an unattended/background job. Approval covers only the described pipeline and its agreed limits; a new pipeline or materially larger run needs a new approval. The user explicitly requested this gate.
- A brief SSH `hello world` connection check is authorized by the user for validating this skill after agreement completion. It is not pipeline approval.
- Use at most **one active experiment pipeline at a time** across all agent sessions and project checkouts. Its data-loader workers and subprocesses count toward aggregate resource usage, not as separate experiments. Trials and seeds run sequentially within the approved pipeline. Do not launch overlapping jobs. Parallel jobs require a new explicit user instruction, preceded by a concrete resource-utilization warning. Do not terminate an unrelated or user-started process to make room.
- The server is shared. There is no assigned CPU, RAM, or GPU quota and no assigned GPU. Inspect availability and any server scheduler/policy before choosing a device or proposing work. Idle capacity is not a reservation.
- Act only as the authenticated unprivileged user. Keep project file reads/writes inside that user's home, check resolved paths and symlinks, and do not browse other users' homes or system configuration. Executing installed system binaries and reading ordinary resource telemetry is allowed. Never use sudo, privilege escalation, driver changes, or system-level installs.
- Obtain `SSH_HOST`, `SSH_USER`, and `SSH_PASSWORD` from the local ignored `.env` or inherited environment. Do not print, persist, transmit elsewhere, or commit secrets. Require a trusted SSH host-key match; do not disable checking or automatically accept an unknown key.
- Before each pipeline launch, show the exact command and device, runs/epochs, estimated duration, output path, and that exceeding the estimate will not stop an approved job. Check existing jobs and free resources. Use a fresh output path and preserve existing checkpoints and research artifacts. For overnight or other long-running jobs, keep the approved job running until it completes or the user stops it. A longer runtime does not authorize additional trials, seeds, jobs or a retry. Report progress and resource concerns without terminating the job.
- After the authorized hello-world check, a bounded read-only server inventory is authorized to identify home/project paths, scheduler, CPU/RAM capacity and GPU status. It does not authorize a pipeline.

## Resource planning

The only agreed fixed resource limit is one concurrent compute job. CPU, RAM, VRAM, disk and wall-time figures in the remote agreement are provisional until measured on representative real data and reconciled with server policy. Estimated duration is for planning and monitoring, not an automatic stop time. Do not change model, batch, seed, epoch or selection settings silently to fit a proposed budget. A sequential search still requires approval of its total trial/run count and an estimated duration.

Local Git changes are allowed; `git push` needs explicit approval in this session. Never commit credentials, local `.env`, or generated research data.

## Verified release and launch procedure

Follow [docs/REMOTE_OPERATIONS.md](docs/REMOTE_OPERATIONS.md). Build a source-only release with scripts/reu_release.py, upload/extract into a fresh directory under the user home, and verify its expected manifest hash. Never overwrite code used by an active job. Optional SSH client dependencies are in requirements-remote.txt.

Approved main.py experiments must use scripts/reu_remote_job.py inside tmux or the approved scheduler. Its fixed per-account advisory lock coordinates participating launchers across checkouts and sessions; manual launches can bypass it, so existing-job checks remain required. The lock is not a GPU reservation, sandbox, resource quota, or proof of user approval. Do not modify the launcher to bypass a busy lock.

One approval may cover a named sequential batch with an explicit total budget; do not ask again for each seed in that approved batch. A goal alone does not authorize unspecified remote pipelines.

## Required remote tmux session

Run remote pipelines in the existing user-owned `sugar_run` tmux session. Reuse an idle pane; do not interrupt a running pane or create another session as a fallback. Check `tmux has-session -t =sugar_run`; if it is missing, report that and ask before creating or substituting a session. Read-only inventory and connection checks may run outside tmux. The guarded launcher enforces the session name and records it. Scheduler/lab policy still takes precedence; if a scheduler becomes required, reconcile its use with this session rule before launching.
