# Cleanup and research handoff — September 23, 2026

No commit, push or staging was performed. Review the entire accumulated working tree before committing; this includes prior correctness/experiment changes, not just today's cleanup.

## Removed from active workspace

Archived499 files outside the repository: final_testing_outputs(200), logs(6), temp(2), outputs/smoke(213), outputs/transfers(8), outputs/archive(53), docs/reference(17). These are not all exact duplicates; some are unique historical artifacts retired from the active workspace. No evidence was permanently deleted or falsely classified as redundant.

Archive root: `C:/Users/Lance/Documents/REU_Research_Archive/2026-09-23_freeze/retired_workspace`.

`cleanup_moves.json` one level above maps every original folder to its archive destination. Recovery: copy the required archived item back to its recorded source after checking the source does not already exist. Old server-update ZIPs are archived and should not be mistaken for a full current deployment.

Two removed scratch Python files: temp/test_fragmentation.py (covered by maintained pipeline/reconstruction tests) and temp/testing_smiles.py (hardcoded one-molecule probe). Experiment wrappers and utility scripts remain because tests, main.py or existing commands use them. We did not equate lack of an import with unused CLI code.

Current outputs/assay and outputs/size remain intact. Referenced root caches/output paths and imported historical evidence remain. No model math, data, seed budget or scientific settings changed during cleanup. Tracked historical artifacts appear as deletions; ignored generated artifacts do not appear in Git status. Raw datasets already tracked in Git remain tracked; removing them from Git requires a deliberate data-distribution decision.

## Frozen evidence

`C:/Users/Lance/Documents/REU_Research_Archive/2026-09-23_freeze/evidence` contains991copied files including canonical current checkpoints. Each copy was compared by SHA256 with its source. Manifest: evidence_manifest.json. Original code state: source_before_cleanup.zip, working_tree.patch, git_head.txt and git_status.txt. This local copy is not an off-device backup. Keep the manifest and archive with research records.

## Shareable material

- [Professor report](results/2026-09-23/PROFESSOR_REPORT.md)
- [Detailed audit](results/2026-09-23/RESEARCH_AUDIT.md)
- [All100 recent training runs](results/2026-09-23/PER_SEED_RESULTS.md)
- [Normal-inference molecule gallery](results/2026-09-23/visualizations/evaluation/index.html)
- [Sampler-only demonstration](results/2026-09-23/visualizations/sampling/index.html)

Galleries are portable HTML/SVG/PNG and traces; no server is needed. Open locally in a browser. GitHub normally shows HTML source rather than running the gallery: download/extract the share ZIP and open index.html. No trained checkpoints or raw datasets are in the professor bundle.

## Proposed commit

Title: `Consolidate molecular OOD experiments and freeze reproducible research results`

Description:

- Correct ERM graph handling, checkpoint/search selection and evaluation safeguards.
- Add configurable shared/independent inference and fixed-trajectory paired comparisons.
- Consolidate workflows behind main.py experiment; retain compatible legacy entry points.
- Add contrastive sweeps, view diagnostics and actual-forward molecule visualizations.
- Preserve curated100-run Assay/Size evidence, protocol limitations and professor-facing reports.
- Retire historical generated artifacts and scratch probes from the active repository with an external recovery manifest.
- Validate with the full local test suite and source compilation; preserve model behavior during cleanup.

Review with git status and git diff; specifically review historical deletions and new docs/results assets. Do not use git add -f on ignored outputs. The user will commit after validation. No recommendation to rewrite history or force-push.

## Post-cleanup verification

- Full suite:135tests passed.
- Python source compilation completed without errors.
- git diff --check passed; staging index unchanged/empty.
- 120local gallery links resolve.
-991frozen evidence files verified against source SHA256;499archived files verified by path/size inventory.
- No expensive training, dependency changes, Git commit or push.

Follow-up: remaining referenced root output folders were migrated on September23. See [PATH_MIGRATION.md](PATH_MIGRATION.md); earlier kept-in-place statements describe the pre-migration freeze.
