# Frozen connectivity sensitivity study

This is the next bounded diagnostic after the assay follow-up. No training, model promotion or test evaluation. Existing ordered pooled-contrastive checkpoints, seeds 8301–8305, independent-selected and independent inference. Same data, weights and original selected atom IDs across all intervention modes.

## Hypothesis and comparisons

Do omitted original bonds between already-retained atoms contribute to poor validation discrimination?

- Current: unchanged forward pass. Must reproduce saved validation AUC within 1e-4.
- Induced: restore every original bond whose endpoint atoms were both retained. No new atoms; no nonexistent bonds. Preserve the existing score/gate associated with each bond; intra-fragment gates remain unchanged. This comparison adds bonds, so is not bond-budget matched.
- Random0/1/2: three fixed uniform draws of bonds from the induced candidate set, each with exactly the current bond count per molecule and exactly the current atom set. Keep isolated atoms in pooling. Preserve each chosen bond's associated gate. These controls change edge identity and gate distribution as well as connectivity; they are NOT the random-fragment training baseline. Each draw votes across the same five views; do not combine the draws into a larger ensemble.

Every mode uses full OOD validation; none loads test. Record within-assay as well as overall AUC, per-molecule predictions, checkpoint/config/data hashes. No ratio search or training. Stop after all five seeds; no result-dependent extra trials. The approved continuation is treated as authorization to execute these bounded additional diagnostics without changing agent files.

## Visual and geometric evidence

Render 25 actual saved selections (five molecules, five views, seed 8301), showing original, selected and induced graphs on identical atom coordinates. Example indices 2438, 732, 705, 1288, 1337 are selected using geometry, not labels or predictions. Graph schematics do not show bond order/stereochemistry; colors cycle after 20 connected components.

Decompose disconnection on all five seeds and the same 100 sampled molecules. Separate 51–100 atoms from >100 atoms: the previous >50 bin contained ten extreme molecules (165–335 atoms) and four moderate molecules (51–60 atoms). Treat repeated seeds/views as repeated measurements, not independent molecules.

## Execution

Local unit and end-to-end synthetic tests, then a server synthetic replay and a full five-seed validation probe on GPU 1 inside the existing sugar_run pane. Zero training epochs, 25 mode/seed evaluations, expected 5–15 minutes. Continue past an estimate if needed. Preserve results under outputs/assay/connectivity_study_v1/evaluation; visuals under outputs/assay/connectivity_study_v1/visualizations. One guarded pipeline at a time. Do not modify AGENTS.md or the SSH skill.

## Decision rule

Use these results to decide whether connectivity warrants a NEW, separately bounded training comparison. An inference-only intervention can mismatch how a classifier was trained. Improvement would motivate retraining confirmation; failure would not prove connectivity irrelevant. Do not switch the reference or touch test based on this diagnostic.

## Exact commands

### smoke

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/REU_Connectivity_Probe_20260926_v1/scripts/reu_remote_job.py --record-dir /home/lsantiago4/REU_Summer_2026/outputs/smoke/connectivity_probe_v1 --gpu 1 -- connectivity-probe --smoke --reference-dir /home/lsantiago4/REU_Summer_2026/outputs/smoke/contrastive_study_v1/results/per_view --processed-dir /home/lsantiago4/REU_Summer_2026/outputs/smoke/contrastive_study_v1/results/per_view/processed
```

### real

```bash
/home/lsantiago4/REU_Summer_2026/.venv-reu-server/bin/python /home/lsantiago4/reu_releases/REU_Connectivity_Probe_20260926_v1/scripts/reu_remote_job.py --record-dir /home/lsantiago4/REU_Summer_2026/outputs/assay/connectivity_study_v1/evaluation --gpu 1 -- connectivity-probe --reference-dir /home/lsantiago4/REU_Summer_2026/outputs/assay/scorer_study_v1/results/ordered --processed-dir /home/lsantiago4/REU_Summer_2026/data/processed/baseline_study_v1/assay --raw-path /home/lsantiago4/REU_Summer_2026/data/raw/drugood/lbap_core_ec50_assay.json
```

Source manifest: `744bc792add4ad8a3d646a081a3b00137d7c9fb8bddedf2104cfbab4984e15f9`. 174 local tests and local/server synthetic replay checks passed.
