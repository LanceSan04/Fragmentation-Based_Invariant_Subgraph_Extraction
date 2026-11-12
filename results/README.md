# Initial benchmark results

## DrugOOD LBAP-core EC50

This page records the completed EC50 study on the downloaded ChEMBL30 Assay, Scaffold, and Size splits. [Per-run CSV](ec50_runs.csv)

- **Static** uses pooled atom and bond descriptors for fragment features and raw atom descriptors for classification. It includes 30 seeds per split (11401–11430).
- **Shared GNN** uses a molecular GNN to compute atom embeddings before selection. Those embeddings feed the fragment scorer and the classifier. It includes 10 seeds per split (11401–11410). Both models use independent inference.

| Split | Static test AUC, 30 seeds | Shared GNN test AUC, 10 seeds | Paired GNN − static | 95% interval for paired difference |
|---|---:|---:|---:|---:|
| Assay | 76.53 ± 1.76 | 76.99 ± 1.87 | +0.16 | [-1.30, +1.61] |
| Scaffold | 67.45 ± 1.11 | 66.24 ± 0.73 | -1.68 | [-2.47, -0.88] |
| Size | 66.47 ± 0.79 | 64.64 ± 1.11 | -1.61 | [-2.59, -0.64] |

AUC columns show mean ± sample standard deviation on a 0–100 scale. The paired columns are AUC percentage points, computed using seeds 11401–11410 in both models. The intervals describe seed variation on these fixed splits. The paired Assay interval crosses zero. On Scaffold and Size, the shared GNN scored lower on average under the fixed recipes.

### Run protocol

Both models used three Group Graph views and trained for 150 epochs per seed. Each checkpoint was selected by the highest pooled `ood_val` ROC-AUC on CPU and replayed before final `ood_test` evaluation. The CSV stores AUC on a 0–1 scale and includes every completed trajectory in this study. Recipe 1 was used for Assay and recipe 2 for Scaffold and Size. The recipes were selected in an earlier validation search and held fixed for the encoder comparison.

- [EC50 replication protocol](../configs/ec50_replication_20261031.json)
- [Tuning recipes and raw split file hashes](../configs/official_original_tuning_20261030.json)
- [Study runner](../src/experiments/ec50_replication.py)

The export was checked against the archived `static/completed.json`, `shared-gnn/completed.json`, and `paired_comparison.json` for study `ec50_replication_20261031_v2`. The archived source manifest SHA-256 is `37436f0945a73385b4faf9f822cfcdd4fd213ce03d82de29168a13c1884b0eeb`. Exact historical replay requires the matching release, the pinned input files, and the prior validation-search artifact. The historical runner checks config hashes byte-for-byte, including line endings.
