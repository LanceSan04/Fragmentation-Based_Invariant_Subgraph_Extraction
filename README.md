# Fragment-based invariant subgraph learning for molecular OOD prediction

This project studies whether selecting chemical fragments helps a model predict molecular activity under distribution shift. It uses a SuGAr-inspired ensemble of CIGA-style subgraph learners.

1. Preprocessing converts each molecule to an atom graph and builds one or more fragment graph representations.
2. Each model view scores connections between fragments and selects a subgraph.
3. The selected structure is reconstructed at the atom level and classified by a GNN.
4. The ensemble combines view predictions. Shared inference uses a common selected graph, while independent inference lets each view select its own graph.

## Setup and data

Use Python 3.9 with the pinned Torch, PyTorch Geometric, and RDKit versions in [requirements.txt](requirements.txt). The examples below assume the environment is active and the current directory is the repository root.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` or use `.\.venv\Scripts\python.exe` in place of `python`.

| Dataset and protocol | Input and version |
|---|---|
| DrugOOD EC50 and IC50, later tuning and confirmation | Use the downloaded ChEMBL30 LBAP-core JSONs under `data/raw/drugood_official_chembl30_20261028/`. The [tuning protocol](configs/official_original_tuning_20261030.json) records exact paths and SHA-256 hashes for assay, scaffold, and size. IC50 is a historical study. |
| DrugOOD Ki, downloaded-data transfer and readout studies | Use the ChEMBL30 LBAP-core Ki JSONs under the same versioned directory. The [Ki input helper](src/experiments/original_tuning.py) pins each split file hash. |
| DrugOOD EC50 and Ki, earlier studies | These used ChEMBL29-derived inputs under `data/raw/drugood/`. Some basenames match the downloaded files, but the contents and split membership differ. The standard `experiment.yaml` points at the earlier EC50 Assay path. |
| GOOD-HIV | Place the published `GOODHIV.zip` under `data/raw/good/`. The [adapter](src/data/good_hiv.py) pins the archive hash and reads the predefined covariate splits for the configured domain. |

DrugOOD preprocessing preserves the membership in the input split file. A training seed controls training randomness and the order of examples within the training split. The default DrugOOD run uses `ood_val` for selection and `ood_test` for final reporting. For a recorded experiment, verify the raw file hash against its protocol and keep the processed cache tied to that version.

## Run through `main.py`

```bash
python main.py smoke --device cpu
python main.py full --config configs/experiment.yaml --validate-loaders
python main.py eval path/to/best_model.pt --config path/to/effective_config.yaml --splits ood_test
```

The smoke command uses a tiny synthetic fixture and temporary files. For real data, use `full` to preprocess and train in one invocation, or `preprocess` and `train` separately to reuse caches. Use `resume` to continue from a saved checkpoint. For evaluation, supply the effective config and cache associated with the checkpoint.

| Task | Command to start with |
|---|---|
| Find workflows and their options | `python main.py experiment --help` |
| Search hyperparameters | `python main.py search --config configs/grid_search.yaml --dry-run` |
| Retrain and evaluate the search winner | `python main.py search-eval --config configs/grid_search.yaml --params path/to/best_params.json --dry-run` |
| Compare the learned model, random selection, and full-graph ERM | `python main.py experiment select --help` |
| Compare those models across datasets | `python main.py experiment cross-domain --help` |
| Inspect the recorded EC50 replication protocol | `python main.py experiment ec50-replication --help` |
| Inspect the original Ki Assay readout pilot | `python main.py experiment ki-readout --help` |
| Compare inference policies or inspect view overlap | `python main.py experiment compare-inference --help` and `python main.py experiment compare-views --help` |
| Render molecular subgraph galleries | `python main.py experiment visualize-views --help` |
| Check fragmentation, losses, or an existing cache | `python main.py diagnostics --help` and `python main.py inspect --help` |
| Find data audits and visualization tools | `python main.py tool --help` |

The standard model ranks connections by learned scores and uses those scores to gate messages. Prediction gradients reach the scorer through the gates in the default hard-selection path. The random-selection control replaces the ranking with a seeded random ranking. ERM uses the full atom graph. For comparisons, record the selection budget, actual retained atoms and bonds, and the checkpoint and inference rules. Within-assay ROC-AUC can supplement pooled ROC-AUC for DrugOOD when enough assays contain both classes.

## Configuration and results

See [initial benchmark results](results/README.md) for the EC50 tables and per-seed validation and test AUCs.

- [configs/experiment.yaml](configs/experiment.yaml) defines the standard model, data, and artifact settings. [configs/grid_search.yaml](configs/grid_search.yaml) defines the generic Optuna search.
- The remaining YAML and JSON files under `configs/` specify controls, fixtures, and recorded studies. Use the chosen workflow's help to find its config and prerequisites. The `tests/` directory contains verification fixtures and checks.
- Standard runs write to `outputs/<dataset>/<fragmentation_recipe>/<run_name>/` when `output_root` is set. Look for `run_manifest.json`, `config/effective_config.yaml`, `checkpoints/best_model.pt`, and `final_evaluation_metrics.json`, as enabled.
- When a study accepts `--output-dir`, give it a fresh path such as `outputs/<dataset>/<study>/<run-id>/`. Read its manifest or completion file to confirm which runs finished and whether test evaluation was enabled.
- Keep processed graph caches under `data/processed/`. Cache location and preprocessing settings are part of the recipe for a rerun.
- Set `deterministic_training: true` or use `--deterministic-training` on supported commands to request deterministic Torch operations. Operations without a deterministic implementation can still run and emit warnings. Record the device and software versions for repeated runs.
- The native classifier uses `classifier_readout: mean` by default. The `sum` and `mean_count` options test alternative ways to combine atom embeddings. The optional `fragment_feature_mode: molecular_gnn` computes atom embeddings before selection. With `classifier_feature_mode: encoder` and hard cropping, the classifier also receives those embeddings for the retained atoms. The standard static recipe uses pooled input features for fragments and raw atom features for classification.

## Code layout and verification

| Directory | Role |
|---|---|
| `src/data/`, `src/fragmentation/` | Dataset loading, atom and bond features, fragmentation, graph caches, and batching |
| `src/models/`, `src/losses/` | Fragment scoring, selection, reconstruction, classification, ensembling, and training objectives |
| `src/training/`, `src/evaluation/` | Training, checkpoint selection, prediction, and evaluation metrics |
| `src/experiments/`, `src/analysis/` | Study protocols, analysis, and visualization behind the CLI |

```bash
python -m unittest discover -s tests -b
```

Tests for local release and launch helpers skip when those helpers are absent. Unittest buffers output from passing tests and shows captured output when a test fails.

The separate SuGAr reference in the research checkout uses a pinned upstream snapshot and compatibility adapters. Its environment, DrugOOD feature conventions, and evaluator are documented in `baselines/sugar/REPRODUCTION.md`. The native atom-only control and that reference are different experimental conditions.

Group Graph and RelMole attribution and license texts are in [GROUP_GRAPH_LICENSE.md](GROUP_GRAPH_LICENSE.md) and [RELMOLE_LICENSE.md](RELMOLE_LICENSE.md).
