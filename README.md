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

## Licenses

Group Graph and RelMole components: see `GROUP_GRAPH_LICENSE.md` and `RELMOLE_LICENSE.md`.
