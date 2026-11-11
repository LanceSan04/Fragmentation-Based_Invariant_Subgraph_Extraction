# Same-checkpoint SuGAr inference comparison

Purpose: compare shared versus independent inference without retraining or
changing checkpoint selection. Shared is the paper-oriented reference;
independent is explicitly an additional experimental policy.

Use the five frozen tuned reference runs, initializations2-6, with the exact
same ten per-view validation-selected checkpoints for both policies. Recipe:
contrast32, penalty.01, sampling90%, released contrastive indexing,40epochs.
This recipe was tuned for shared inference; neither policy gets further tuning.

- Shared: mean raw scorer outputs, common hard top-k, uniform probability vote.
- Independent: each model's own hard top-k graph, uniform probability vote.
- All ten views; no greedy selection, new seeds or checkpoint reselection.
- Recompute both policies on validation, then test, sequentially.
- Hash-pin weights, training specifications, previous validation/test reports
  and replication summary. Verify shared AUC against existing records (1e-4).
- Save every prediction and per-seed score, means/sampleSD, and paired differences.
- Any seed-level t interval is approximate with only five seeds. No automatic
  policy promotion, additional studies, tuning, retries or test-based selection.

Interpretation: isolates final-inference behavior conditional on existing weights.
Does not compare independent-versus-shared checkpoint selection, fully retuned
systems, or explain the cause of any gap. Test scores for the shared policy have
already been observed, so this is an exploratory comparison, not a new untouched
test benchmark. A favorable result would motivate validation-first prospective
confirmation rather than retroactively changing the paper-reference primary.

Budget: zero training runs, five saved models, both splits and policies;
estimate2-5minutes on one availableGPU. This is not a timeout. Exact remote
command approval is required before evaluation. No Ki/GOOD continuation.
