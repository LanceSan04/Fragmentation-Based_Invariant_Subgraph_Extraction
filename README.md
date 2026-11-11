# Fragment-based invariant subgraph learning for molecular OOD prediction

This project combines **SuGAr** (an ensemble of CIGA-style subgraph learners) with
**Group Graph fragmentation**. Each molecule is split into chemically meaningful
fragments; a learned scorer selects fragment connections; the selected fragments are
rebuilt into an atom-level subgraph; and a GNN classifies the molecule from that
subgraph. Several such "views" are trained and their predictions are averaged.

Datasets: DrugOOD LBAP-core EC50 (assay, scaffold and size splits) and GOOD-HIV.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt     # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # Linux/macOS
```

Raw data goes in `data/raw/` (`data/raw/drugood/lbap_core_ec50_*.json`,
`data/raw/good/GOODHIV.zip`). Processed graph caches are written to `data/processed/`
and results to `outputs/`; both are git-ignored.

## Running

Everything runs through `main.py`.

```bash
python main.py smoke                                   # 1-minute synthetic end-to-end check
python main.py preprocess --config configs/experiment.yaml
python main.py train      --config configs/experiment.yaml
python main.py eval       path/to/best_model.pt --config configs/experiment.yaml
```

| Command | Purpose |
|---|---|
| `preprocess`, `train`, `resume`, `eval`, `full` | Build caches, train an ensemble, evaluate a checkpoint |
| `search`, `search-eval` | Optuna hyperparameter search and repeated evaluation (`configs/grid_search.yaml`) |
| `experiment select` | Multi-seed comparison of the learned model, random-selection control and ERM on one dataset |
| `experiment cross-domain` | Same three arms across EC50 scaffold/size and GOOD-HIV, test evaluated once after validation |
| `experiment tune` | Bounded hyperparameter search for the fragment model |
| `experiment compare-inference`, `compare-views` | Shared vs independent inference; per-view overlap and agreement |
| `experiment selection-gating-probe` | Frozen-checkpoint test of learned vs random ranking and gating |
| `experiment visualize-views`, `diagnose` | Molecule galleries of selected subgraphs; gradient/domain diagnostics |
| `diagnostics fragments\|losses\|subgraphs`, `inspect` | Fragmentation statistics, loss curves, subgraph analysis, cache inspection |
| `tool <name>` | Fragment figures, fragment structure audit, view overlap, within-assay AUC, DrugOOD provenance |

Add `--help` to any command for its options, and `--smoke` to `select`, `cross-domain`, `tune` or `diagnose`
for a fast synthetic run.

### Model variants

Set in the config (or via the `experiment` workflows):

- **Learned** (default): learned fragment-connection ranking and learned message gates.
- **Random control**: `selection_mode: random` selects the same number of connections at random.
- **ERM**: `erm_baseline: true` classifies the full molecule, no fragmentation.
- Options: `selection_budget: atoms` (target share of atoms), `scoring_mode: symmetric`,
  `inference_policy: shared | independent`, `contrast_scope: pooled | per_view`.

## Layout

```
main.py                 command-line entry point
configs/                experiment.yaml (documented defaults), reference_assay.yaml (frozen recipe),
                        cross_domain_reference.yaml, search/smoke/diagnostic bases, ERM/fragment controls
src/data/               DrugOOD/GOOD loading, featurization, preprocessing and loaders
src/fragmentation/      Group Graph, BRICS and RECAP fragmentation
src/models/             fragment scorer, selection, atom reconstruction, classifier, ensemble
src/losses/             supervised contrastive, diversity, IRM
src/training/           trainer, checkpoint selection, training/evaluation pipeline
src/evaluation/         rationale metrics, within-assay (grouped) AUC
src/experiments/        multi-seed / cross-dataset workflows behind `main.py experiment`
src/analysis/           diagnostics, visualizations and data audits behind `main.py tool`
tests/                  unit tests: python -m unittest discover -s tests
tools/remote/           SSH, release and job-launch helpers for the shared GPU server
```

## Notes on evaluation

DrugOOD EC50 labels are a single potency threshold applied across many assays, so pooled
ROC-AUC is dominated by differences between assays. Report within-assay AUC
(`main.py tool assay-groups`) alongside pooled AUC, and compare the learned model against the
random-selection control at the same selection budget.

## Selection-learning and granularity checks

The historical model still uses `selection_estimator: hard` by default.
`selection_estimator: straight_through` is experimental: hard membership in
the training forward pass, a budget-constrained sigmoid surrogate backward,
and ordinary hard reconstruction at evaluation. `selection_temperature`
defaults to 0.1. Active atoms alone contribute to both classifier BatchNorm
locations. Training retains full graph tensors, so memory and runtime need
a representative pilot. This is not proof of improved OOD generalization.

`view_fragmentations: {atom: 5}` selects the singleton control: one fragment
per atom, original bonds as connections, no internal fragment bonds. Caches
must be separate from Group Graph. Same connection ratio does not guarantee
the same atom retention. As in the historical reconstruction, edgeless
molecules retain all atoms, while isolated atoms in molecules that have
connections are omitted unless selected through a connection (which is
impossible for a truly isolated atom). This policy is recorded rather than
silently changed in the control.

Local synthetic checks, using fresh output paths:

```bash
python main.py experiment select --smoke --device cpu --fragmentation atom --selection-estimator straight_through --compare-checkpoint-rules --report-within-assay --output-dir outputs/smoke/selection_check
python main.py experiment granularity --smoke --device cpu --datasets ec50_scaffold --evaluate-test --output-dir outputs/smoke/granularity_check
python main.py experiment motif-check --epochs 30 --output-dir outputs/smoke/motif_check
```

The `select` checkpoint comparison is opt-in. It saves best raw validation
AUC, final epoch, and best complete five-epoch trailing-average AUC (endpoint
weights, no weight averaging). Ties retain the earliest epoch. Within-assay
AUC is reported only for groups with at least five examples of each class;
macro AUC, pair-weighted AUC, and coverage are logged. It never selects a
checkpoint. Synthetic examples lack assay metadata, so this metric is null.

`granularity` compares fragment/atom x learned/random with paired seeds,
independent validation/inference, all three checkpoint rules, and a primary
validation difference of learned-minus-random advantages across representations.
Defaults: EC50 Scaffold and Size, five seeds, 100 epochs, contrastive 0.1,
diversity/IRM off. Optional GOOD-HIV Scaffold expands the budget. Prepare
fresh caches with `--stage prepare`; train with the same paths after checking
the plan with `--dry-run`. All checkpoints are frozen and validation replayed
before optional `--evaluate-test` loads any test split. Reports include the
per-seed interaction; they do not automatically declare success. Agree on a
success threshold and compute budget before real training. Failure wording:
"A learned-selection advantage was not established under this protocol."

The motif check uses carbonyl presence on a small set of real molecular
structures with synthetic labels and training label noise. It reports loss,
scorer gradients, within-molecule score ranges, held-out motif-fragment
retention, and atom retention for hard/ST/random arms. Its two positive
held-out examples are an execution fixture, not a statistically powered
learning demonstration. Expand the fixed fixture and seeds before treating
motif recovery as a gate to real-data studies.

For a larger fixed engineering check, use `experiment motif-check --expanded
--epochs 50 --seeds 23 24 25 26 27 --output-dir outputs/diagnostics/NEW_NAME`.
This builds valid constructed molecular structures, balances motif presence,
matches positive/negative atom counts, and removes canonical train/held-out
duplicates. Held-out prefix families differ, but similar chemistry remains.
The report includes exact uniform-random motif-retention probability to expose
ceiling effects. Final epochs are evaluated; seeds/epochs are not selected
against held-out performance. These labels are synthetic, not measured activity.

Granularity runs save wall time and CUDA peak allocated/reserved memory.
Linux RSS is a process-lifetime high-water mark for the parent process;
it does not include aggregate data-loader worker memory.

## Licenses

Group Graph and RelMole components: see `GROUP_GRAPH_LICENSE.md` and `RELMOLE_LICENSE.md`.
# Experimental scorer pathways

`main.py experiment scorer-paths` runs a fixed Scaffold comparison: the current
hard selector with learned message weights, straight-through selection with
learned or unit weights, and random ranking with learned or unit weights.
The random arms use the same seeded ranking implementation. Unit weights in the
straight-through arm preserve node/connection selection masks; they do not
restore dropped bonds. Hard learned ranking with unit weights is rejected because
it has no prediction-loss learning path for the scorer. Defaults elsewhere remain
unchanged. `--smoke` checks all arms locally; `--stage evaluate --source-results ...`
restores frozen checkpoints without retraining. Full runs require a verified cache,
report three checkpoint rules and validation retention, and never select by test.
# Prepared mentor study queue

`python main.py experiment study-queue --dry-run --output-dir outputs/mentor_study`
prints the draft budget in `configs/mentor_study_queue.yaml` without training.
The native queue compares static fragment pooling with an optional
`fragment_feature_mode: internal_gnn` encoder (GINE inside each fragment, then
pooling), screens small/ramped diversity, and tunes contrastive strength using
validation only. Confirmation seeds are disjoint; test never selects settings.
The draft maximum is 198 training trajectories, including two short pilots.
Use `--stage-limit pilot` for resource measurement before approving the sweep.
`--resume` verifies completed jobs and resumes between jobs; it never retries
failed/interrupted trajectories automatically. Queue status, per-job logs,
configs, checkpoint rules, view diagnostics and final results live beneath the
specified output folder. It stops on resource errors; independent transfer
dataset branches can continue after a dataset-specific failure.

This is our current pipeline, **not** the official SuGAr reproduction. The
pinned reference and its outstanding compatibility work are documented in
`baselines/sugar/REPRODUCTION.md`; upstream jobs are not silently substituted.

`python main.py experiment ki-reference-controls --project-root . --output-dir outputs/cross_domain/ki_reference_preview --stage full`
prints the frozen Ki Scaffold/Size comparison without running it. It compares
learned fragment ranking, random ranking with learned gates, prediction-only
full-graph ERM, and the pinned SuGAr transfer reference. `--stage pilot` plans
eight three-epoch validation-only trajectories; `--stage full` plans forty
trajectories and validation-gated test evaluation. Both require `--execute` to
run. See `docs/results/2026-10-24/ki_reference_controls_plan.md` for recipes,
protocol differences, uncertainty, and the separate remote approval requirement.
Real training additionally requires `--reference-python /path/to/reference/venv/bin/python`;
the workflow verifies the reference imports before starting any model and uses
that interpreter for reference training and evaluation. It installs nothing.
