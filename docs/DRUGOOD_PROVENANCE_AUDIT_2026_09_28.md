# DrugOOD provenance audit and GOOD-HIV launch preparation

## Completed locally

Ran `scripts/audit_drugood_provenance.py --output-dir outputs/data_audit/drugood_provenance_v1` using the project virtual environment. This reads all five splits of the three EC50 JSONs without changing them. It downloads and parses literal upstream curator configs without executing upstream Python. Reference: DrugOOD commit `eeb00b8da7646e1947ca7aec93041052a48bd45e`, specified by SuGAr.

- All scientific config fields match upstream after excluding physical source/target paths.
- ChEMBL 29 is recorded locally and specified upstream.
- Across all three files: zero invalid SMILES, zero canonical duplicates within splits, zero canonical molecule overlaps between splits.
- Recomputed classification thresholds agree with recorded threshold 5; zero exported classification-label mismatches against regression values.
- Train/OOD-validation/OOD-test domain IDs are disjoint. IID domain overlap is expected.
- Current metric code uses class-1 probability for ROC-AUC (`src/training/sugar_trainer.py` and `model_selection.py`).

| Dataset | Train | OOD validation | OOD test | All five splits |
|---|---:|---:|---:|---:|
| Assay | 4540 | 2572 | 2490 | 12661 |
| Scaffold | 2570 | 2532 | 2533 | 12661 |
| Size | 4684 | 2313 | 2398 | 12661 |

Size atom ranges: train 27-333, validation 23-26, test 6-22. Validation and test deliberately ask for different degrees of size extrapolation. These ranges do not establish that size alone causes the performance gap.

Evidence: `outputs/data_audit/drugood_provenance_v1/report.json`, pinned upstream config copies and hashes. RDKit version is recorded. Upstream threshold implementation is saved under `upstream/drugood/curators/curator.py` for inspection.

## Remaining limitations

Matching configs does not prove the original curator commit, random seed, runtime, or exact split membership matches SuGAr. Original ChEMBL measurements/units and exact SuGAr JSON bytes were not available for comparison. Exported regression-to-class labels are consistent, but this cannot verify upstream unit conversion. Near-duplicate chemical similarity and full raw-to-cache graph/label alignment are separate remaining checks. Do not regenerate or overwrite current splits. No basis from this pass to discard existing results or declare leakage absent in all forms.

## GOOD-HIV status

Update: user approved starting the prepared study. Launched September 29 at 02:45:37 UTC (September 28 evening Eastern) in existing `sugar_run` pane %4. GPU 1 replaced provisional GPU 0 because GPU 1 was idle. Source manifest verification passed, cache files were present, fresh output path was checked, and no competing project training process was running. Launcher reports running, child PID 1162508. Output: `outputs/cross_domain/frozen_transfer_v1/good_hiv_scaffold_study/`; initial log confirms cache validation started. The launch command below was used with `--gpu 1`. Scope remains 15 sequential runs, 100 epochs each, with final test evaluation; no automatic Size study or retry.

Connection attempts failed with hostname-resolution error `[Errno 11001] getaddrinfo failed`. No server status was verified and no training was launched. The previous successful release/cache/pilot remain the intended inputs; their current state must be rechecked after connection recovery.

Prepared scope: GOOD-HIV Scaffold covariate split, learned/random/ERM, seeds 8701-8705, 100 epochs each, 15 sequential runs, best validation checkpoint, test evaluation only after successful validation replay. Rough planning duration 15 hours from pilot timings. It continues beyond that estimate until complete or user-stopped; no automatic retry. GOOD Size follows a correctness review of Scaffold, not an automatic second launch.

Proposed exact command (physical GPU 0 is provisional until live resource inspection; existing sugar_run session and idle pane required):

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/REU_Cross_Domain_20260928_v2/scripts/reu_remote_job.py --record-dir /home/lsantiago4/REU_Summer_2026/outputs/cross_domain/frozen_transfer_v1/good_hiv_scaffold_study --gpu 0 -- cross-domain --stage study --datasets good_hiv_scaffold --raw-root /home/lsantiago4/REU_Summer_2026/data/raw --cache-root /home/lsantiago4/REU_Summer_2026/data/processed/frozen_transfer_v1 --evaluate-test
```

Requires exact-command approval per the existing project agreement, then successful connection, release-pin verification, fresh output-path check, no competing project job, and GPU availability. Agent files unchanged.

## Selection versus message weighting

`FragmentCIGA.score_fragment_edges` produces scores. `select_fragment_edges` uses detached scores for a Boolean top-k mask: connections are included or excluded. `forward` separately passes continuous scores into atom reconstruction; `AtomClassifier.forward` multiplies GNN messages by the resulting bond weights at each layer. Internal fragment bonds also receive derived weights. This is not just a weight on the final vote.

Current learned condition learns both ranking and gates. Current random selection condition replaces ranking with random values and gates with unit weights. A comparison between them cannot attribute performance to just one mechanism.

Suggested later experiment, not implemented/launched here: first hold saved graphs/weights fixed on validation and compare learned message gates with unit gates as an inference sensitivity diagnostic. Then train a random-ranking/learned-gating arm against the existing jointly learned and random/unit arms under matched budgets. Important: hard top-k is nondifferentiable. Simply training learned ranking with gates removed would remove the current prediction-loss gradient path to the scorer; that is not a valid clean fourth arm without an explicit alternative learning mechanism or frozen pretrained scorer. Frozen-policy diagnostics are not substitutes for retraining.
