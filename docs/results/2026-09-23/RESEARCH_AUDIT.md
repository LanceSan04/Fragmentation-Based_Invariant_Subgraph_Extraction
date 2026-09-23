# Research audit and freeze — September 23, 2026

## Scope and evidence status

This records the September repair, controlled experiments, workflow consolidation, visualization and cleanup sequence. Earlier report/poster findings remain historical and are not silently combined with current runs. Values below come from downloaded result artifacts reviewed in this conversation; current freeze includes source snapshot, a dirty-tree patch, Git HEAD/status and independent copies of canonical Assay/Size evidence. This cleanup is a repository inventory/reference review and test pass, not a new exhaustive forensic review of every algorithm or all Git history.

Technical Explanation: a seed controls pseudorandom choices, not dataset membership. A checkpoint is a saved set of learned weights. Each batch computes loss, backpropagates and updates weights; an epoch traverses the training loader. Validation ranks checkpoints after epochs; test does not select a checkpoint within the implemented comparison runner. Repeated researcher inspection of test results still makes the overall development process exploratory.

Dumb It Down: we repaired and organized the machinery, then ran controlled comparisons. We now preserve both favorable and unfavorable results before deciding what to change next.

## Model and inference

Five views start from shared initialized weights, then learn through sampled fragment graphs. Each has a fragment scorer GNN and an atom classifier GNN. Training samples fragment-edge instances, scores candidates, applies hard top-k, reconstructs an atom graph with continuous message weights, and predicts a molecule-level class. Gradients reach the scorer through continuous weights; the discrete top-k choice itself is detached. Current GroupGraph experiments use the same fragmentation method across all views, not five different fragmentation algorithms.

Shared inference aggregates scorer outputs, constructs a shared selected graph and averages classifier probabilities. Independent inference selects each view's own graph and averages its probabilities. Both retain top-k. Evaluation disables random sampling. The selection budget is based on the full fragment-edge count, capped by eligibility (100 original, sample70, select50 in a documented example), not necessarily the sampled count multiplied by edge_ratio.

Loss combines summed per-view prediction loss and weighted supervised contrastive loss; diversity and IRM are optional. Contrastive loss uses labels and embeddings, not ROC-AUC directly. Current main recipe has λcontrast=.1, diversity=IRM=0. ERM is a separately trained full-molecule control: scorer/fragment selection are bypassed, auxiliary losses zero. It is not evaluation of a pretrained fragment checkpoint and was added after the original REU work.

## Work chronology

- Source/history review identified several plausible regression mechanisms and protocol mismatches; it did not prove a single historical causal explanation.
- Correctness work repaired ERM graph direction, complete moving-average window eligibility, search test suppression, and inappropriate fragment-explanation paths for ERM. Checkpoint/default configuration and incompatible modes received guards.
- User adopted best individual validation AUC (window1). Controlled selection runs use fixed100-epoch trajectories and no early stopping to compare two selection rules over the same weights. Ordinary training presets still support early stopping.
- Compared multiview prediction-only and auxiliary objectives. Overnight factorial loss search explored contrastive, diversity and IRM; selected a contrastive-only reference. IRM/environment diagnostics exposed limitations of environment construction; no general IRM benefit established.
- Added delayed/ramped diversity. Apparent favorable schedule results could select checkpoints before activation; pre-activation trajectories were not identical. Kept diversity off rather than claim a causal improvement.
- Added numerical view comparison. Own-graph voting performed better in initial saved-checkpoint validation comparisons; integrated an explicit inference policy rather than replacing defaults silently.
- Tuned ordinary Assay hyperparameters with fixed loss. Frozen Assay recipe: lr=.0003, wd=0, dropout=.5, edge_ratio=.5, sample_ratio=.7; five views, hidden128, three GNN layers, batch64. Completed additional seeds and frozen-recipe Scaffold/Size benchmarks.
- Corrected fragment scorer encoder directionality: canonical unique edges remain the provenance basis, while the encoder sees reverse edges with endpoint features swapped. Historical checkpoints trained before this fix are not interchangeable with current-code evidence.
- Size parameter search used training/validation only; selected lr=.001, wd=.0001, dropout0, edge_ratio=.7, sample_ratio=.7. Best two-repeat search objective was .760579; repeated occurrences of a setting were not additional independent seeds.
- Added fixed-trajectory dual policy runner, then paired fragment/ERM and contrastive controls. Final evaluation compares both saved checkpoints under both policies.
- Consolidated experiment implementation under src/experiments and main.py experiment; compatibility wrappers retained. 135 tests passed after consolidation and visualization work. Historical CPU parity checks established identical core behavior for consolidation.
- Added actual forward-trace visualization and 60-checkpoint numerical diagnostics. Added explicit contrastive sweep command; no architectural change in that sweep.

## Current frozen experimental ledger

The shareable results.json and PER_SEED_RESULTS.md contain 100 recent runs, all seeds 7001–7010 and100 epochs:
- Assay: contrastive .1,0,.01 and ERM (40 runs).
- Size: contrastive .1,0,.01,.03,.3 and ERM (60 runs).
Each fragment run yields four selection/inference combinations; those four results are not four independently trained models.

Assay test means (S/S,S/I,I/S,I/I):
- λ.1: 76.58,76.83,76.29,76.61.
- λ0: 74.86,75.44,76.07,76.53.
- λ.01: 75.31,75.93,75.37,76.66.
- ERM:75.61±3.35. Reference S/I:76.83±1.55, above ERM in 7/10 paired seeds.

Size test means (same order):
- λ.1:62.65,63.95,62.67,64.11.
- λ0:62.83,63.79,62.53,63.25.
- λ.01:62.53,63.62,62.29,63.57.
- λ.03:61.75,62.92,62.21,63.56.
- λ.3:62.99,64.03,63.75,64.50.
- ERM:59.82±1.64. Reference I/I beats ERM in 10/10 seeds, +4.30points.
Size I/I validation means for λ0,.01,.03,.1,.3:77.34,77.65,77.33,77.42,77.20. The highest test and validation settings differ. Do not adopt .3 merely because its test result won.

Earlier frozen-Assay-recipe five-seed benchmarks: Scaffold shared fragments68.38, independent68.55, ERM67.86; Size shared63.11, independent63.44, ERM60.56. These use different seeds/protocol from recent comparisons and are historical context, not pooled into the current means.

## Views and visual evidence

Sixty Assay reports each evaluate2,572 validation molecules, across3coefficients×10seeds×2selectionpolicies. Independent vote beats best individual own-graph view50/60times. Shared-selected λ.1 mean Jaccard=.60013, own probability correlation=.78466, disagreement=.13403. λ0: .57952,.76085,.14315. λ.01: .59676,.78987,.12627. Jaccard excludes empty unions; correlations exclude undefined cases. Drop-one-vote improves sharedensemblevalidation137/300checks; does not justify post-hoc pruning as a confirmed improvement.

Conclusion: views differ and voting is often useful. Neither greater difference nor attractive molecular pictures guarantees better generalization. Assay diagnostics do not automatically describe Size views.

Portable galleries in docs/results/2026-09-23/visualizations contain evaluation and sampling modes for checkpoint seed7001 best_shared, validationindices0–4. Each traces a single actual forward per view and molecule. Sampling-demo freezes BatchNorm/dropout and enables sampler only. It is not a logged real training batch. Selected structures are highlights on the complete molecule. There is no separate sharedensemble graph, fragment-node plot or score heatmap yet. Firstfive molecules are large and strongly positive in predictions, so representative example selection remains future work. Visuals are qualitative, not causal evidence.

## Limitations and unbiased next decisions

- ERM shares fragment-tuned settings rather than its own search. Do not claim superiority to a fully optimized baseline.
- Learned selection versus matched random selection has not been isolated. Graph perturbation, graph size and message gating can explain some benefit.
- Shared initialization and zero dropout raised possible ERM duplication; inspected Size checkpoint classifiers differ across all 10 seeds. Useful prediction diversity remains unmeasured.
- Tests have repeatedly informed discussion. Use validation to choose settings and describe current comparisons as exploratory; reserve an untouched confirmation when available.
- No current-run causal attribution to historical bugfixes without old/new controlled reproduction. GPU nondeterminism can produce seed-matched trajectory differences.
- No exact full-training resume guarantee. Several runners fail on existing outputs intentionally. Diagnostic launcher supports skipping completed reports, not resumed training.
- OGB and other endpoints have not been completed in this work. Official SuGAr numbers are contextual, not a reproduced apples-to-apples baseline. Do not claim SOTA.
- Prior explanations sometimes overstated the universality of independent inference. Current evidence is stronger on Size than Assay.

Next bounded work: baseline-specific validation tuning, learned-vs-random selection control, validation domain/size error breakdown, and a frozen Scaffold protocol. Architecture experiments should follow a named hypothesis, fixed seed/budget, a baseline and stopping criteria. Do not optimize for cherry-picked test seeds.

## Freeze, cleanup and commit handoff

External freeze: C:/Users/Lance/Documents/REU_Research_Archive/2026-09-23_freeze. Evidence copy contains991 files from outputs/assay and outputs/size including checkpoints, verified SHA256 against originals. Source ZIP, Git HEAD/status and binary diff capture uncommitted state; Git HEAD alone does not identify experimental code. Evidence manifest records bytes and hashes. This is a local snapshot, not an off-device backup or a claim that all historical source environments were reconstructed.

Cleanup archived499 files across final_testing_outputs, logs, temp, outputs/smoke, outputs/transfers, outputs/archive and docs/reference. No permanent deletion. cleanup_moves.json records original/destination paths. temp/test_fragmentation.py is superseded by maintained tests; temp/testing_smiles.py is a one-molecule scratch probe. Kept wrappers used by tests/documented server commands, active cache paths, historical experiment presets, licenses and raw data. No dataset or training semantics were changed.

Canonical current results remain at outputs/assay and outputs/size (ignored by Git). Imported historical evidence remains under outputs/imported. Tracked historical results/logs/scratch files appear as deletions for the user to review. Already-tracked data remains tracked; .gitignore does not retroactively remove it. Share curated docs/results rather than committing all outputs/checkpoints. No Git commit or push performed by the agent; user validates and commits.

Validation after cleanup: full test suite, source syntax compilation, diff whitespace checks, manifest/portable-gallery link checks. See CLEANUP.md for exact outcome and commit description. No expensive training run launched.
