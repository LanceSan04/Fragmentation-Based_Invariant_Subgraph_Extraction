# Next bounded training design — retained-atom budget

Status: proposed design only; not implemented and not launched. The connectivity probe did not justify switching reconstruction or running a bond-restoration training sweep.

## Question

Does choosing fragment edges to approach a retained-atom budget improve OOD validation compared with choosing a fixed fraction of fragment edges? Does learned ranking beat random ranking under that same budget rule when both retain learned message gates?

## Three arms

1. Current reference: learned fragment-edge ranking, edge_ratio=0.5, existing reconstruction and learned gates.
2. Atom-target selection: learned ranking, target 65% of original atoms; existing reconstruction and learned gates.
3. Random-ranking atom-target control: random edge priorities, the same 65% atom-target rule, but the same learned scorer/gating mechanism and auxiliary objectives as arm 2. This isolates ranking more closely than the historical random-fragment baseline, which also removed learned gates. Keep that older baseline named separately.

All arms keep the original fragment sampler at 0.7 for training and disable it for evaluation. No changes to fragmentation, atom features, backbone, contrastive coefficient (pooled 0.1), diversity (0), IRM (0), optimizer, number of views or learning-rate settings.

## Atom-target rule to implement and test

For each molecule/view, rank eligible fragment-edge instances, as in current top-k selection. Calculate the number of distinct atoms covered by endpoint fragments for every NONEMPTY prefix of that ranking. Pick the shortest prefix whose atom count is closest to ceil(0.65 * original_atom_count), resolving equal-distance ties in favor of the lower atom count and then fewer edges. Reconstruct those fragment edges with the existing reconstruction.

Whole fragments are indivisible: this matches a TARGET, not necessarily the exact realized count between arms. Do not claim exact atom-budget matching. Save target/actual atoms, residual error, selected/eligible edges, bonds, components and method for each molecule in diagnostics. Molecules with no fragment edges retain the existing full-graph fallback and must be reported separately. Random priority must have reproducible per-view training draws and molecule-stable evaluation draws; it must not reuse test labels or pick a favorable random draw.

This prefix rule avoids a hidden change to induced reconstruction: it does not automatically add every edge between retained atoms. No explicit connectivity constraint is introduced, because the frozen connectivity intervention showed no consistent benefit.

## Fixed scientific budget

Three arms × five new paired seeds (8401–8405) × 100 epochs = 15 sequential full training runs. Independent validation checkpoint selection and independent final validation inference only. No early stopping; no test loading. All other settings copied from the frozen reference config. Do not silently substitute fewer views, epochs, batch size or another device to fit runtime.

Estimate approximately 4–8 server hours after implementation, with uncertainty from the new selector. Benchmark a bounded pilot before committing to the full batch; a slow selector should be optimized or the scope explicitly revised, not silently run as an unbounded search. Continue an approved run past its estimate unless stopped by the user. One GPU/pipeline in sugar_run.

## Required checks before launch

- Budget counts agree with atom IDs emitted by reconstruction for batches, mixed fragmentation methods, ties, cycles, isolated fragments and no-edge molecules.
- A prefix's covered atom count is monotonic; tie handling and fallback are deterministic.
- Defaults reproduce existing model outputs and checkpoint loading.
- Random ranking changes only hard rank priority; learned gates remain differentiable and prediction loss still reaches the scorer.
- Evaluation is batch-order invariant and sampler behavior is unchanged.
- Training and inference use the same selector; no unsupported shared-ensemble path silently falls back to edge ratio.
- Smoke tests and a resource pilot pass before full training. Save complete configuration/source/data provenance.

## Decision before any future test

Report all seeds and paired deltas. Use mean independent-validation AUC as the primary metric, with within-assay AUC, retention errors and connectivity as secondary diagnostics. A provisional practically meaningful improvement is >=0.005 mean AUC with positive paired differences on at least four of five seeds; this is a screening rule, not a significance test. If realized retention distributions differ materially between learned and random arms (e.g. mean retained-atom fraction differs by >0.02), flag the comparison as not adequately budget-matched and do not attribute performance solely to ranking.

Do not promote a model using test results. Even after this validation screen, previously inspected assay test data remain exploratory. Cross-dataset confirmation is still needed.
