# GOOD Size training and frozen selection/gating diagnostic

Approved and launched September 29, 2026 at 14:15:07 UTC (10:15 AM Eastern). Both exact commands below were submitted as one shell chain joined with `&&` in existing sugar_run pane %4, GPU 1. GOOD Size child PID 1344848; launcher confirmed running and cache validation started. The second command is queued in the shell and runs only if the first exits successfully. Both releases verified again before launch, both output paths were fresh, no other user training process was running, and GPU 1 was idle. No automatic retries. Estimates do not impose termination limits.

## 1. GOOD-HIV Size

Unchanged benchmark release/config: three arms (learned, random, ERM), seeds 8701-8705, 100 epochs each, 15 sequential trajectories. Independent validation/test inference for fragments; full-graph ensemble for ERM. Test only after successful checkpoint validation. Estimated 9-15 hours based on completed Scaffold and Size pilot. Current preflight: no project compute jobs, GPU 1 idle, sugar_run pane %4 idle. Recheck before launch.

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/REU_Cross_Domain_20260928_v2/scripts/reu_remote_job.py --record-dir /home/lsantiago4/REU_Summer_2026/outputs/cross_domain/frozen_transfer_v1/good_hiv_size_study --gpu 1 -- cross-domain --stage study --datasets good_hiv_size --raw-root /home/lsantiago4/REU_Summer_2026/data/raw --cache-root /home/lsantiago4/REU_Summer_2026/data/processed/frozen_transfer_v1 --evaluate-test
```

## 2. Frozen GOOD-HIV Scaffold validation diagnostic

Five existing learned checkpoints; zero training epochs; OOD validation only. Compare learned ranking/learned gates, learned ranking/unit gates, random ranking/learned gates, random ranking/unit gates. Three fixed random draws are reported separately, not treated as extra seeds or ensemble members. Total eight evaluation modes per checkpoint. Estimated 10-30 minutes; not a measured runtime.

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/REU_Selection_Gating_20260929_v1/scripts/reu_remote_job.py --record-dir /home/lsantiago4/REU_Summer_2026/outputs/good_hiv/scaffold_selection_gating_v1 --gpu 1 -- selection-gating-probe --reference-dir /home/lsantiago4/REU_Summer_2026/outputs/cross_domain/frozen_transfer_v1/good_hiv_scaffold_study/results
```

The fresh diagnostic release is staged, verified and pinned: `90e8b50c9001a3aa00b0ead516532e6b0dfb2c9d103733a5765967480039de86`. The existing benchmark release is unchanged. No credentials, agent files, or SSH helper were uploaded. No agent instructions changed.

Safeguards: frozen checkpoint and validation-cache hashes; current-policy AUC replay tolerance 1e-4; original reconstructed forward comparison; exact per-molecule fragment-edge count match for random ranking; unit-gate intervention clones the selected graph without altering atoms, bonds or features. Test data is never loaded by this probe. Source/checkpoint/config and per-molecule prediction records are retained.

Limitations: changing ranking can change retained atom count and connectivity despite matching fragment-edge counts. Frozen-model intervention changes the inference distribution and is not a retrained factorial. Unit gates remove the present gradient path to learned ranking, so a learned-ranking/unit-gate training arm would need a separately defined learning mechanism. This diagnostic should inform a later training proposal, not be described as that training experiment.

Validation completed locally: 17 focused tests passed (gate intervention, random selection, guarded launcher); synthetic end-to-end replay completed at `outputs/smoke/selection_gating_probe_v1`. Synthetic AUCs of 0.5 are plumbing checks, not scientific findings.
