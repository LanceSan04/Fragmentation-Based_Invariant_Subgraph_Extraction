# Connectivity study — 2026-09-26

## Main conclusion

Restoring omitted bonds between already-retained atoms does not improve average validation AUC in this frozen-model probe. Keep the current pipeline as the reference. The visual disconnection is real, but its appearance alone is not evidence that fixing connectivity will improve prediction.

## What was executed

- Generated 25 graph schematics: five validation molecules × five views, using real saved atom/bond IDs. The gallery shows original, selected, and same-atoms/all-original-bonds graphs on identical coordinates.
- Decomposed connectivity on the prior 100-molecule, five-seed sample. Separated moderate (51–100 atoms) and extreme (>100 atoms) sizes.
- Ran all five saved pooled-contrastive checkpoints on all 2,572 OOD validation molecules under five modes. No training and no test split access.
- Preserved the selected atoms exactly in every mode. The three random-bond controls also preserve the original selected bond count exactly for every molecule and view. They are NOT the earlier random-fragment training baseline.
- All 174 local unit tests and local/server synthetic smoke checks passed. All reference checkpoint and data hashes match the frozen assay follow-up. Current-mode validation parity passed within 1e-4. Agent and skill files were not changed.

## Validation AUC

Each row uses the same five views and the same saved independent-selected checkpoint. The random column is the mean of three separately evaluated random draws, not an enlarged voting ensemble.

| Seed | Current | Restore all bonds between retained atoms | Random bonds, exact current budget (3-draw mean) |
|---|---:|---:|---:|
| 8301 | 0.801913 | 0.803952 | 0.803617 |
| 8302 | 0.804879 | 0.804242 | 0.804091 |
| 8303 | 0.817795 | 0.815982 | 0.816852 |
| 8304 | 0.818590 | 0.817807 | 0.818169 |
| 8305 | 0.811792 | 0.811557 | 0.812172 |
| Mean | 0.810994 | 0.810708 | 0.810980 |

Restoring bonds changes mean AUC by -0.000286 and improves 1/5 seeds. The random-control mean differs from current by -0.000013. Differences this small do not justify a model switch.

| Metric, mean across seeds | Current | Restore bonds |
|---|---:|---:|
| macro_assay_auc | 0.536207 | 0.532941 |
| within_pair_weighted_auc | 0.517420 | 0.515918 |

At least one view gained bonds for 1604–1918 of 2,572 molecules, depending on seed. This was not an intervention that changed only a handful of examples.

## What the pictures and counts show

- Molecule 732, view 5: 49 atoms originally; 31 selected atoms split into 5 components. Restoring 3 original bonds leaves the same 31 atoms in 2 components.
- Molecule 1337, view 2: 326 original atoms; 249 selected atoms split into 35 components. Restoring 28 bonds leaves the same atoms in 9 components.
- Thus some disconnection comes from missing bonds among retained atoms, and some remains because intermediate atoms were removed. The current edge-selection design intentionally retains selected attachment bonds; this observation alone is not a coding bug.
- Graph schematics omit bond order and stereochemistry. Component colors repeat after 20 components. Consult the original SMILES/chemical depiction for chemistry.

| Original atoms | Distinct sampled molecules | Current components, mean | Same atoms with all bonds, mean |
|---|---:|---:|---:|
| 1–25 | 31 | 1.54 | 1.35 |
| 26–50 | 55 | 1.86 | 1.50 |
| 51–100 | 4 | 3.16 | 2.20 |
| 101–335 (sample maximum) | 10 | 12.39 | 8.09 |

The previous >50-atom bin mixed four 51–60-atom molecules with ten 165–335-atom molecules. That distinction matters when describing the 9.75-component average. Repeated views/seeds are not additional independent molecule samples.

## Descriptive size cohorts on full validation

These secondary analyses were calculated after the probe and are not selection criteria. The large cohorts contain very few negative labels; their AUCs are fragile.

| Original atoms | Molecules | Positives / negatives | Current AUC | Restored-bond AUC |
|---|---:|---:|---:|---:|
| 1–25 | 796 | 652 / 144 | 0.7334 | 0.7315 |
| 26–50 | 1425 | 1298 / 127 | 0.8349 | 0.8360 |
| 51–100 | 111 | 106 / 5 | 0.7396 | 0.6777 |
| 101–maximum | 240 | 236 / 4 | 0.3908 | 0.3916 |

## Interpretation limits

- Frozen classifiers were trained on the original reconstruction. This does not answer whether training a different reconstruction from scratch would help.
- Restoring bonds preserves atoms but increases the bond budget. Random controls preserve both budgets, but change bond identity and associated gate values together; they do not isolate connectivity alone.
- This does not prove the learned fragment scorer is useless. The retained atom sets were selected by that scorer in every mode.
- We did not choose a winning random draw, change checkpoints or revisit test results.

## Next direction

Prioritize what atoms/fragments are retained, rather than immediately training a bond-restoration architecture. A bounded training design is in CONNECTIVITY_NEXT_TRAINING_DESIGN_2026_09_26.md. It is not implemented or launched. Keep pooled contrastive, IRM/diversity off, fixed validation selection, and no further test inspection while developing it.

## Where to look

- Visual gallery: `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\assay\connectivity_study_v1\visualizations\index.html`
- Per-molecule connectivity data: `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\assay\connectivity_study_v1\visualizations\connectivity_decomposition.json`
- All model metrics/predictions and plan: `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\assay\connectivity_study_v1\evaluation\results`
- Server execution record: `C:\Users\Lance\Documents\GitHub\REU_Summer_2026\outputs\assay\connectivity_study_v1\evaluation\launch.json`
