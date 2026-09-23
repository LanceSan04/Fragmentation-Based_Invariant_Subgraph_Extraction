# Fixed-budget checkpoint selection comparison

## What this tests

Each model trains once for 100 epochs. Early stopping is disabled explicitly in
comparison mode (the YAML patience value is ignored). Two policies observe the
same weights and validation curve without changing optimization:

- `best_raw`: checkpoint with highest individual-epoch validation AUC, including
  epochs 1–4.
- `best_ma_endpoint`: endpoint checkpoint of the highest complete five-epoch
  validation-AUC window. This is the previously used policy.

Ties keep the earlier winner. All views are saved together. No weight averaging,
greedy view selection, test reporting, or resume is allowed in this comparison.
Missing validation or unavailable AUC fails clearly.

The best-raw checkpoint necessarily has at least as high an observed validation
AUC as the endpoint checkpoint on the same trajectory. This is not an unbiased
proof of better generalization. The experiment measures the endpoint effect and
its seed variation; lock a defensible policy before final test evaluation rather
than repeatedly adjusting it to maximize test results.

## Server steps

Upload the updated project files through WinSCP, including the new files under
`src/training`, `configs`, `scripts`, and `tests`. Do not overwrite the server's
virtual environment or existing results.

From the project root:

```bash
source .venv-reu-server/bin/activate
python -m unittest discover -s tests -q
python scripts/smoke_test.py --device cuda --compare-selection
bash scripts/run_selection_comparison.sh
```

Expected local test count: 108. CUDA must be verified on the server. The runner
starts six fresh trajectories: ERM and fragment selection, seeds 11/22/33. There
are two saved policy checkpoints per trajectory, not twelve training runs.
Budget: 600 total epochs. Use the server's persistent session/job facility if
disconnecting SSH. The runner stops on errors or existing destination folders.

No preprocessing rerun is needed: the new configs have the same preprocessing
hashes as their corresponding controls. Preserve any server-specific raw-path
edits consistently if applicable.

## Outputs to return

Download `outputs_policy_comparison/drugood_lbap_core_ec50_assay/group_graph/`.
Each of the six run directories contains ordinary config, manifest, loss CSV,
and final metrics, plus:

```text
selection_comparison/
    comparison.json
    best_raw.pt
    best_ma_endpoint.pt
```

The JSON distinguishes `selection_score` from `validation_auc_at_selection`.
For `best_ma_endpoint`, the first is the five-epoch average; the second is the
endpoint model's actual epoch AUC. It records selected epochs and completed
budget. The policy checkpoint files omit optimizer state and are intended for
evaluation, not exact training resume.

The ordinary final metrics/checkpoint still use the MA-endpoint restoration.
Use `selection_comparison/comparison.json` to compare both choices. Local smoke
tests reload each policy checkpoint and check its validation AUC against the
recorded value. No raw DrugOOD training was run locally for this setup.

## Earlier suggestions: implementation status

Implemented in the partner branch and retained: fragment-encoder reverse edges,
stable log probabilities/soft voting, full-count top-k budget, shared fast/slow
reconstruction weights and broader seeding.

Implemented in the subsequent correctness pass: ERM bidirectional atom input,
complete-window warm-up, smoothed-score logging, top-K runner-up retention,
BatchNorm weight-averaging guard, ERM rationale guard, validation-only search,
matched control configs and executable smoke checks.

Not yet implemented as architectural changes: alternative message gating,
diversity redesign, feature-width/encoder changes, a new learned atom encoder
before pooling, full weight averaging with BatchNorm recalibration, exact resume,
or a broad new tuning campaign. These require separate experiments or engineering
work. The previous three-seed results do not justify changing all of them now.

IRM's mean-to-sum change is already in the partner branch. The current control
runs disable IRM, diversity and contrastive penalties intentionally; that does
not mean those mechanisms have been deleted. Test them individually later.
