# Audit and Changes — September 2026

A review of the full codebase against the initial-stage results, prompted by two symptoms:
OOD test AUC below the SuGAr baseline, and tuned hyperparameters scoring *worse* than
arbitrary ones (~0.69 vs ~0.74 on assay).

Both symptoms turned out to have causes unrelated to model capacity. The headline finding
is that the baseline we were measured against is itself disputed.

---

## Part 1 — The comparison was anchored on a contested number

The report treats SuGAr's Table 2 as ground truth. A second peer-reviewed paper (AdvMood,
`papers/AdvMood.pdf`, Table I) measures the *same* baselines on the *same* DrugOOD splits
and lands ~8 points lower:

| EC50-Assay, OOD test ROC-AUC | ERM | CIGA | best reported |
|---|---|---|---|
| SuGAr, Table 2 | 75.57 ± 1.23 | 77.52 ± 0.97 | 78.62 ± 0.84 (SuGAr ENS) |
| AdvMood, Table I | **67.87 ± 2.22** | **69.37 ± 1.81** | 78.54 ± 0.41 (AdvMood) |

Two papers placing plain ERM 7.7 points apart on identical splits means at least one is
using a different protocol. Against AdvMood's numbers, our results are **at or above ERM on
all three EC50 splits** and level with CIGA on two of them:

| split | ours | AdvMood ERM | AdvMood CIGA | AdvMood best |
|---|---|---|---|---|
| EC-Assay | 68.43 | 67.87 | 69.37 | 78.54 |
| EC-Scaffold | 67.38 | 66.03 | 67.53 | 68.57 |
| EC-Size | 63.74 | 62.49 | 65.54 | 66.83 |

So the project is not "below baseline" in any unambiguous sense. It is below one paper's
baseline and at another's. **Nothing further should be tuned until our own ERM number says
which regime we are in** — that is what the new `erm_baseline` flag (Part 5) is for.

---

## Part 2 — Correctness bugs found and fixed

### 2.1 Training and evaluation built different-sized subgraphs
`src/models/fragment_masker.py` — `segment_topk_mask`

`k` was computed over *eligible* edges only. With `fragment_sample_ratio: 0.7` and
`edge_ratio: 0.3`, training selected `0.7 × 0.3 = 0.21·E` edges while evaluation selected
`0.30·E`. The classifier's BatchNorm running statistics were therefore fitted to graphs
**~43% smaller** than the ones it was scored on — a train/eval distribution shift created
entirely by the sampler.

**Fix:** size `k` from the graph's full edge count, then clip to the number of eligible
edges. The sampler still restricts *which* edges may be chosen; it no longer changes *how
many*. Pinned by `test_eligible_mask_does_not_shrink_selection`.

This is the change most likely to move the number on its own.

### 2.2 The fragment graph passed messages in one direction only
`src/fragmentation/group_graph.py` (also `brics.py`, `recap.py`) → consumed in
`src/models/fragment_masker.py`

`build_unique_fragment_edges` emits one canonical column per inter-fragment bond, ordered
`src <= dst`, and nothing symmetrized it. The masker's GINE therefore only ever propagated
from low to high fragment id, so a fragment's representation was built from an arbitrary
subset of its neighbours — determined by fragment numbering, which carries no chemistry.

**Fix:** symmetrize inside `FragmentMasker.forward` for the encoder only. The stored
`fragment_edge_index` is left untouched — top-k selection, reconstruction provenance and the
diversity loss all index it — and scoring still happens on the canonical columns. The
reversed half gets src/dst-swapped edge features via the new `reverse_fragment_edge_attr`
(layout `[src atom 46 | dst atom 46 | bond 14 | src size | dst size]`), so the reverse
direction is chemically correct rather than a copy. Pinned by three tests.

### 2.3 Analysis figures were rendered under gating the model never used
`src/models/atom_reconstructor.py`

Two reconstructors disagreed on internal-bond weight: the fast path gated internal bonds by
their fragment's soft importance, the slow path left them at 1.0. Training and evaluation
use the fast path; `SuGAREnsemble.extract_subgraphs` and `FragmentCIGA.get_invariant_subgraph`
— everything feeding the interpretability figures — use the slow one. **The report's Figure 3
rationale comparison therefore visualized a subgraph the model was never run on.** A comment
in the file asserted the two paths gave "identical GNN output"; they did not.

**Fix:** extracted the gating rule into one shared `fragment_soft_importance` helper used by
both paths, extended `build_reconstructed_bond_weight` to gate internal bonds, and corrected
the false docstring. `test_edge_weights_match` now asserts the two paths agree and passes.

Figure 3 should be regenerated before it is used again.

### 2.4 Gradient was being killed on confidently-wrong examples
`src/models/fragment_ciga.py`, `src/models/sugar_ensemble.py`

Both computed `log(softmax(x).clamp(min=1e-8))`. Once a prediction is confidently wrong the
clamp saturates, the gradient goes to zero, and the example stops contributing — exactly the
examples that carry the most learning signal. The ensemble additionally averaged
probabilities before taking the log, so one confident view could drag the average to the
clamp floor.

**Fix:** `F.log_softmax` for the single-view path; `logsumexp(...) - log(n)` for the
ensemble, which is the log of the mean probability computed stably in log space.

Two call sites (`train_sugar.py`, `src/training/model_selection.py`) then applied `softmax`
to values that were *already* log-probabilities. Numerically a near-no-op for ranking, but it
read as a bug and would silently break if the return type ever changed — switched to `.exp()`.

### 2.5 Auxiliary loss weights were silently divided by `num_views`
`src/training/sugar_trainer.py`

Prediction loss was **summed** over views; the IRM term was **averaged**. At `num_views: 5`,
`lambda_irm: 2.0` was really 0.4 relative to prediction. Every recorded λ in the search and
in the report means something different from what it says.

**Fix:** IRM is now summed, matching prediction and matching SuGAr, which backpropagates the
sum of its base models' losses. The reduction convention is documented in the code and in
`configs/experiment.yaml`.

⚠️ **This changes what previously recorded λ values meant.** Old IRM settings are not
comparable to new ones; the search should be re-run rather than reinterpreted.

---

## Part 3 — Why tuned parameters lost to arbitrary ones

This was the second symptom, and it has a concrete cause — two, actually.

### 3.1 The tuned runs were undertrained; the "arbitrary" run was not
`scripts/grid_search.py` — `run_eval`

`best_params.json` records the overrides used during the search (`epochs: 40`), and eval
applied those *on top of* the spec's `epochs: 100`. So the spec was silently overridden and
every headline result trained for 40 epochs, not the 100 the report claims.

Verified in the archived outputs:

| file | epochs | ood_test AUC |
|---|---|---|
| `eval_results_3.json` | 40 | 0.6796 ± 0.059 |
| `eval_results_6.json` | **100** | **0.7185 ± 0.016** |
| `eval_results_assay.json` | 40 | 0.6843 ± 0.018 |
| `eval_results_scaffold.json` | 40 | 0.6738 ± 0.009 |
| `eval_results_size.json` | 40 | 0.6374 ± 0.007 |

Same config, 40 vs 100 epochs: **+0.039 AUC and a third of the variance.** At
`batch_size: 256` on 4540 assay molecules, 40 epochs is only ~710 optimizer steps.

That alone plausibly explains the gap you noticed: a hand-set config that happened to train
longer would beat a "tuned" one capped at 40 epochs.

**Fix:** the spec's overrides now win, and any conflict with the recorded search-time value
is printed at startup rather than applied silently.

### 3.2 The search optimized a statistic that does not transfer
The objective is `best_ma5_val_auc` — a max over epochs of a moving average, then an argmax
over trials, on 2–3 seeds. Two problems compound:

- **The winner is inside the noise.** The top-10 assay trials span 0.0097 val AUC while
  within-config seed error is ~0.0053. The search is ranking noise.
- **`ood_val` cannot proxy `ood_test` on this benchmark.** Mean fragments per molecule is
  train 16.4 / ood_val 21.1 / **ood_test 12.3** — validation and test sit on *opposite sides*
  of train in molecular size. Your own per-epoch logs show corr(val, test) = +0.25
  (`outputs/test2.csv`) and +0.12 (`outputs/test3.csv`).

Picking the argmax over a weakly-correlated, noise-dominated validation signal is a
winner's-curse machine: it reliably selects configs whose val score was lucky, and luck does
not transfer. **This is the structural reason tuning underperformed**, and no amount of extra
trials fixes it.

**Fixes applied:**
- `selection_ma_window: 1 → 5`, so in-run checkpoint selection is smoothed and matches the
  search objective. (Your existing `selection_ma_window` / `selection_weight_avg` machinery
  was already built for this but was left switched off.)
- Eval now also reports the validation split, so the honest val→test gap is recorded for the
  final runs instead of living only in the search leaderboard.
- `eval_base_seed: 1000` — search seeds are `base_seed + trial × repeats + r`, so eval seeds
  1–5 were literally re-runs of the earliest search trials. They now do not overlap.
- `search_repeats: 2 → 3`.

**Protocol going forward:** report the **mean of the last 10 epochs across ≥5 seeds**, never
the argmax-val epoch. Two same-config seeds in your logs differ by 0.028 test AUC, so
**only gaps larger than ~0.03 AUC are real.**

---

## Part 4 — Configuration corrections

`configs/experiment.yaml`:

| setting | was | now | why |
|---|---|---|---|
| `selection_ma_window` | 1 | 5 | §3.2 |
| `fragment_sample_ratio` | 0.7 | 0.9 | outside the file's own documented 0.8–1.0 range; drives §2.1 |
| `batch_size` | 128 | 64 | 4540 training molecules; SuGAr uses 32 |
| `weight_decay` | 0.01 | 0.0001 | was 20× above its own documented ceiling; SuGAr uses 1e-4 |
| `erm_baseline` | — | `false` | new, see Part 5 |

A comment block claimed *"Prediction-only baseline is active below: contrastive, diversity,
and IRM weights are all zero"* while the file actually set `lambda_diversity: 0.5` and
`lambda_irm: 2.0`. **This is the highest-risk defect in the repo** — it mislabels runs, and
anyone trusting it would report an ablation that never ran. Corrected, along with stale
guidance on epochs, batch size and weight decay.

`configs/grid_search.yaml`: `params_path` pointed at `best_params_size.json`, which does not
exist — `eval` mode raised `FileNotFoundError` unless `--params` was passed. Cleared.

---

## Part 5 — New: the ERM anchor

`erm_baseline: true` (or `--erm-baseline`) makes every view skip fragment scoring, top-k
selection and the edge gate, and classify the full atom graph: a plain GINE baseline under
this repo's own featurizer, splits, metric and training loop.

This is the measurement Part 1 needs, and it is immune to the cross-paper protocol dispute:

- **Our ERM ≈ 68** → we are in AdvMood's regime; our results are at parity with ERM and level
  with CIGA, and the 100-epoch fix alone (→ ~71.9) puts us clearly above CIGA.
- **Our ERM ≈ 75** → we are in SuGAr's regime and the fragment machinery is genuinely costing
  ~7 points, which the ablation ladder then localizes.

Run it with `num_views: 1` and all three lambdas at 0.0.

Also fixed while wiring this: `train_sugar.py` and `main.py` **never seeded anything**, so
direct runs were not reproducible at all — only `grid_search.py` seeded. Added
`seed_everything` (random / numpy / torch / CUDA) at the top of `run_training`. And the final
metrics report now records `epochs`, `batch_size`, `edge_ratio`, `hidden_dim`, `seed` and the
rest of the training config, which is why it was hard to tell what produced the archived
numbers in the first place.

---

## Part 6 — Issues found but NOT fixed

These need your decision or a training run; no code was changed.

### Report corrections required
- **"evaluated five times over 100 epochs"** — all three headline results used `epochs: 40`.
- **"100 Optuna trials"** — assay completed 54, scaffold 58, size 100 (`leaderboard_*.csv`).
- **"Each trial was repeated twice"** — assay used 3 repeats.
- The search space **differed across datasets**: the assay leaderboard has no
  `learning_rate` column; the earlier study searched `learning_rate` but not
  `gnn_layers` / `hidden_dim` / `batch_size`.
- **Table II ablations are single 13–20 epoch runs** (`logs/*.csv`), well inside seed noise.
  "IRM only 74.86" and "Diversity only 73.28" are not supported by the data behind them.
- **Accuracy must not be reported.** ood_test accuracy 0.9064 is exactly the majority-class
  rate (2257/2490) — the model never predicts the negative class; only its ranking is
  informative.
- **Figure 3 was rendered under the wrong gating** (§2.3) and needs regenerating.
- The comparison table should cite both published baselines and state the discrepancy.

### Structural questions the ladder must price
- **Fragment pooling discards intra-fragment structure.** `src/models/fragment_pooler.py`
  mean-pools *raw* atom and bond features per fragment with no atom GNN first; CIGA and SuGAr
  run an atom-level GNN over the full molecular graph. `src/models/optional_atom_encoder.py`
  exists as an empty placeholder for exactly this ablation.
- **Both of SuGAr's diversity mechanisms are effectively off.** `lambda_diversity` was 0.0 in
  the winning scaffold and size configs and 0.5 for assay, and the random sampler was omitted
  by mentor direction. That is close to the paper's own SU-None ablation — its weakest
  variant — so the multi-view ensemble is currently views of near-identical models.
- **The edge gate is a sigmoid; SuGAr's is a raw logit.** Because `set_masks` applies the same
  mask at every layer, ours attenuates each edge by `w^L` — at init `w ≈ 0.5, L = 3`, ~8×
  suppression on every message. SuGAr's unsquashed gate can amplify as well as attenuate.
  Cheap to ablate.
- **`num_envs` is searched even when `lambda_irm = 0.0`**, where IRM never engages — roughly
  20% of trials vary an inert parameter and dilute TPE.

### Hygiene not worth changing yet
- The checked-in `experiment.yaml` reproduces **none** of the three archived eval parameter
  sets.
- Empty modules: `src/evaluation/metrics.py`, `src/evaluation/ood_splits.py`,
  `src/models/weight_average.py`.
- Stale 133-dim cache at `data/processed/bbbp/group_graph/*.pt` (`preprocess_version` still
  `0.1.0`); `src/data/preprocess.py` still documents the old 133/149 widths. Harmless today —
  the width check rejects it — but a *same-width* featurizer change would not be caught.
- There is **no LR scheduler anywhere in the repo**; worth adding cosine or step decay.

---

## Part 7 — Files changed

| file | change |
|---|---|
| `src/models/fragment_masker.py` | top-k sizing (§2.1); graph symmetrization + `reverse_fragment_edge_attr` (§2.2) |
| `src/models/atom_reconstructor.py` | shared `fragment_soft_importance`; internal-bond gating in the slow path; corrected docstring (§2.3) |
| `src/models/fragment_ciga.py` | `log_softmax` (§2.4); `erm_baseline` + `_forward_erm` (Part 5) |
| `src/models/sugar_ensemble.py` | `logsumexp` ensemble (§2.4); ERM view routing |
| `src/training/sugar_trainer.py` | IRM sum (§2.5); ERM guards |
| `src/training/model_selection.py` | `.exp()` on log-probs (§2.4) |
| `scripts/grid_search.py` | override precedence (§3.1); eval seeds; val split reporting; CUDA seeding |
| `train_sugar.py` | `seed_everything`; `.exp()`; ERM wiring + CLI; richer metrics metadata; **deleted 153 lines of unreachable dead code** after `main()`'s early return |
| `main.py` | `--erm-baseline` flag and key forwarding |
| `configs/experiment.yaml` | Part 4 |
| `configs/grid_search.yaml` | Part 4 |
| `tests/test_reconstruction_invariants.py` | **new** — 13 tests |

Note: `src/training/sugar_trainer.py`, `scripts/analyze_subgraphs.py` and
`scripts/export_subgraph_edges.py` also contain earlier uncommitted work of yours
(`selection_ma_window` / `selection_weight_avg` / `log_test_curve`, and the subgraph-analysis
features). A raw `git diff` mixes those with the changes listed here.

### Test status
`python -m unittest discover -s tests` → **91 of 92 pass**, plus 13 new tests all passing.

The one failure, `test_trainer_writes_epoch_loss_component_log_and_uses_auc_selection`, is
**pre-existing and environment-only** — it asserts the `val_auc` column is non-empty, and
scikit-learn is not installed in the interpreter used here. Verified by stashing all changes
and re-running: it fails identically on the original code.

The new tests pin the invariants that broke silently:
- top-k selects the same subgraph size with and without an eligible mask, stays within
  eligible edges, and clips `k` when candidates are scarce;
- the fast and slow reconstructors produce identical `edge_weight`;
- the reverse edge-feature swap is an involution and moves the right blocks;
- `ATOM_FDIM == 46`, `BOND_FDIM == 14`, `FRAGMENT_FEATURE_DIM == 62`,
  `FRAGMENT_EDGE_FEATURE_DIM == 108` — previously **nothing** pinned these, because every
  other test builds synthetic tensors at hardcoded dimensions, so a featurizer regression
  would have passed the whole suite.

---

## Part 8 — What to run next

**Do not tune anything until step 1 is done.** Every rung below: 5 seeds, 100+ epochs, scored
on the last-10-epoch mean. At <15 min/run the whole ladder is roughly 7 hours.

1. **ERM anchor** — `erm_baseline: true`, `num_views: 1`, all lambdas 0.0. Decides which
   published table the report should be anchored on (Part 1). *This is the one run that
   changes the story.*
2. **Re-run the current best config at 100 epochs, 5 seeds.** Expect ≈ 0.719 rather than
   0.684 from the epoch fix alone, before any of the correctness fixes are counted.
3. **Ablation ladder**, each rung priced against the ERM anchor:

   | rung | config | prices |
   |---|---|---|
   | L1 | `edge_ratio 1.0`, `sample_ratio 1.0`, 1 view, no aux | the soft gate alone |
   | L2 | `edge_ratio 0.3`, `sample_ratio 1.0`, 1 view, no aux | hard fragment selection |
   | L2b | L2 + atom GNN before fragment pooling | the raw-pooling design choice |
   | L3 | L2 + `sample_ratio 0.9` | the sampler |
   | L4 | L3 + `num_views 5` + `lambda_diversity` | the ensemble and diversity |
   | L5 | L4 + `lambda_irm` / `lambda_contrast` | the auxiliary objectives |

4. **Only then** re-tune — at most 3 knobs, ≥3 seeds, accepting a config only if it beats the
   incumbent by more than ~0.03 AUC. Include the knobs that define the method and were
   **never searched**: `fragment_sample_ratio`, `num_views`, `contrast_temperature` (frozen at
   0.1 while SuGAr uses 1.0), `aggregation_mode`.
5. **Regenerate Figure 3** (§2.3) and correct the report (Part 6).
