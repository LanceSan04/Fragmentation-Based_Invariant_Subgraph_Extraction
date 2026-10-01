# Remote working agreement

Updated: 2026-09-23. Root AGENTS.md and the SSH skill implement the settled rules. The user authorized a bounded SSH hello-world validation and subsequent read-only inventory; both completed successfully.

## Agreed boundaries

- Run at most one compute job at a time across this agent's remote activity, including across sessions. Check for existing jobs before launching another. Do not stop a user-started job to make room.
- Parallel compute jobs require an explicit user instruction and a resource-utilization warning before proceeding. An instruction to run a sweep alone does not authorize parallel execution.
- A job includes its subprocesses and data-loader workers; account for their aggregate resources. Sequential trials are permitted only within the authorized task's total budget.
- Remote project file access stays within the authenticated user's home directory. No sudo or privilege escalation. Installed system executables/libraries and ordinary read-only resource telemetry are allowed; project file access remains inside the user home.
- Credentials come from the local ignored environment configuration; never include their values in documentation, command arguments, logs, or Git.
- The server is shared and this user has no assigned CPU, RAM, or GPU allowance. The user reports that prior preprocessing, training, and searches did not appear to overload it when only one job ran at a time. This is useful operating history, not proof of a hard resource ceiling.
- Codex must ask before starting every remote pipeline, including preprocessing, smoke training, full training, and searches. Within the specific approved task and its agreed limits, Codex may proceed without asking again for each routine step. A new pipeline or expanded budget needs another approval.
- Overnight and other long-running jobs continue until the approved pipeline completes or the user stops them. An estimated duration is not an automatic termination threshold. If a job exceeds the estimate, Codex reports that at the next progress check but does not stop it or start extra jobs. The user retains control of stopping long runs.

## Terms

- Connection test: a bounded SSH hello-world command; no training or GPU allocation.
- Synthetic smoke test: a small end-to-end correctness check using synthetic molecules, reduced models and two or three epochs. It can exercise preprocessing, training, evaluation, and checkpoint restoration. It produces no scientific benchmark evidence and does not establish real-data memory requirements.
- Resource pilot: a separately scoped measurement using representative real data and the intended model configuration. It estimates peak RAM/VRAM and epoch runtime. Do not call a full real-data epoch cheap merely because it is a pilot.
- Full training: a research run on the real training split using its approved configuration, seed, epoch budget and validation/checkpoint policy. It does not necessarily include preprocessing, test-set evaluation, or a search.
- Search/sweep: multiple training runs with an explicit total run count, executed sequentially unless the parallel exception is invoked.

## Repository evidence

- `configs/experiment.yaml`: five views, hidden dimension 128, three GNN layers, batch size 64, four data-loader workers, and 100 epochs without early stopping.
- `configs/parameter_winner_test_base.yaml`: up to 100 epochs with early stopping, minimum 30 epochs and patience 20.
- `configs/workflows.yaml`: default parameter search has 24 trials with two repeats (48 training runs); selection uses three seeds and 100 epochs.
- `src/experiments/contrastive_sweep.py`: default four coefficients times ten seeds (40 training runs), launched sequentially.
- `src/experiments/plans.py`: synthetic fixture uses four molecules per split, batch size four, hidden dimension eight and one GNN layer.
- `src/experiments/selection.py`: smoke mode uses one seed and three epochs.
- `src/data/dataloader.py`: processed graph splits load into CPU memory; file sizes alone do not establish peak RAM.
- `docs/PATH_MIGRATION.md`: local paths changed recently; remote layout and code version must be verified before selecting commands.

## Validated connection and read-only inventory

On 2026-09-23, the skill's password SSH helper matched the existing trusted `known_hosts` entry, returned `hello world` and exit status 0. A bounded read-only inventory then found:

- Home: `/home/lsantiago4`; expected `REU_Summer_2026` directory exists.
- 96 logical CPUs; 503 GiB total RAM, approximately 449 GiB available at inspection time.
- Two NVIDIA L40S GPUs with 46,068 MiB memory each. At inspection, GPU 0 showed 5,797 MiB used and 1% utilization; GPU 1 showed 665 MiB used and 17% utilization. Occupancy can change.
- `/home` filesystem: 3.5 TiB total, 2.8 TiB used, 521 GiB available (85% used). Swap: 8 GiB total, approximately 7.9 GiB used. Treat the high swap use as a reason to check live memory pressure before a large job.
- `tmux` available; `sbatch` not found. This does not prove there is no lab policy or alternate scheduler.

This inspection did not establish actual peak usage of the project's real-data jobs. It started no pipeline.

## Proposals awaiting agreement

- Four logical CPUs, 16 GiB RAM and 12 GiB VRAM were initial discussion values. Given successful prior runs, no assigned allocation and the measured server size, these are **not agreed quotas**. Capture representative actual usage and any server policy before proposing numeric ceilings.
- Use no more than one GPU automatically; neither GPU is assigned to this user. Check occupancy and scheduler policy first.
- Plan a short synthetic smoke check separately from full training. Five minutes is an initial duration estimate, not a hard kill time; report an overrun at the next check.
- Full training and searches need an explicit total run/epoch count and estimated duration based on representative measurements. Do not silently change scientific settings to meet resource targets. The approved run count is the scope boundary; elapsed time alone does not stop it.
- Initial 10 GiB incremental output budget needs reconsideration for preprocessing caches and retained sweep checkpoints.

## Open questions

- Representative real-data runtimes and peak resources; applicable scheduler or lab allocation policy. Home/project paths above were observed on September 23 and should be rechecked after relocation.
- The mandatory prelaunch proposal includes runs/epochs, estimated duration, output path and continuation plan if the estimate is exceeded, plus exact command and device. A verified local `known_hosts` entry exists. A bounded read-only server inventory after hello world is authorized.
- Default progress monitoring is user-initiated; a scheduled monitor can be added if explicitly requested for a particular job. An approved unattended job may continue in the user's existing `sugar_run` tmux session after SSH disconnects.

The instructions are behavioral rules, not server-enforced isolation or resource quotas. The launch lock enforces exclusion only among cooperating launchers. See REMOTE_OPERATIONS.md for the verified-release procedure.

User preference: all remote pipelines run inside the existing sugar_run tmux session. No alternate/new session without explicit instruction. Connection/read-only inspection is exempt; a busy pane must not be interrupted. The guarded launcher verifies this session.
