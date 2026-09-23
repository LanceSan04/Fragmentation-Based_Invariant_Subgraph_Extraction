# REU molecular OOD project

Use **main.py** as the front door. The current command map, directory layout, server migration, and verification record are in [docs/CODEBASE_GUIDE.md](docs/CODEBASE_GUIDE.md).

```bash
python main.py --help
python main.py experiment --help
python main.py experiment select --help
```

- `src/models`, `src/training`, `src/data`, `src/fragmentation`: model and data implementation.
- `src/experiments`: reusable experiment workflows and shared helpers.
- `configs`: training presets; `workflows.yaml` records search budgets/space and selection defaults.
- `scripts`: compatibility entry points and standalone utility tools. Old commands remain usable.
- `outputs`: new experiment artifacts; `data/processed`: reusable graph caches.
- `docs`: protocol explanations and historical experiment notes.

Historical results, scratch scripts and generated artifacts were archived outside the repository on September 23. See [cleanup and recovery](docs/CLEANUP.md). Current outputs remain local and ignored. Code cleanup does not imply better model performance.

## Research handoff

- [Professor report and portable molecule galleries](docs/results/2026-09-23/PROFESSOR_REPORT.md)
- [Full research audit](docs/results/2026-09-23/RESEARCH_AUDIT.md)
- [All 100 recent per-seed results](docs/results/2026-09-23/PER_SEED_RESULTS.md)
- [Cleanup, verification and proposed commit](docs/CLEANUP.md)

Output/cache migration is complete locally. See [server migration instructions](docs/PATH_MIGRATION.md) before using the updated configs on an older server layout.
