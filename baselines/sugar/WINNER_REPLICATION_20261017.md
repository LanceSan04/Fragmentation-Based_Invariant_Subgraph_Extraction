# Frozen SuGAr recipe replication

Selected by the 18-trial validation-only search: contrast32, upstream diversity
penalty .01, perturbation sampling90%. The exact search completion record is
SHA256-pinned in `configs/sugar_winner_replication_20261017.json`.

Scope: five fresh initializations 2-6, 40 epochs, ten jointly trained views per
initialization, original DrugOOD39/10 features, released training/indexing.
These are five independent initialization repeats, not fifty independent seeds.
Initialization1 was used for hyperparameter selection and is excluded from this
replication summary. The numerical seed IDs differ from the paper's 1-5.

All training and validation runs finish before test evaluation begins.
Each view restores its best validation checkpoint. Primary inference remains
all_mean: all ten views, mean raw scores, shared top-k, equal probability vote.
No greedy subset search, independent inference, corrected training, further
hyperparameter tuning, early stopping, retries or automatic continuation.
The evaluator logs all_max validation as a secondary diagnostic, but test is
restricted to all_mean. Every test result is reported; no best seed selection.

Each run's generated protocol pins its validation record, checkpoint and data
hashes, with all ten views fixed. The evaluator replays validation before test.
Failures stop the batch and preserve completed artifacts; no partial successes
are silently substituted. Aggregate standard deviation is sample SD (ddof=1).

Estimated runtime 2-3 hours on one GPU, sequential, based on measured search
runtime. This is not a timeout; an overrun neither stops nor expands the job.
The command requires --execute plus the project's explicit remote approval.

Synthetic two-seed smoke checks cover the full train/validation/frozen-test
sequence; synthetic scores are not scientific evidence. Existing initial-recipe
test results have been observed, so this is not a claim that the dataset test
split has never previously been inspected. This remains a documented SuGAr
adaptation with released training issues and unresolved author-file equivalence,
not a certified exact replication of the published benchmark.
