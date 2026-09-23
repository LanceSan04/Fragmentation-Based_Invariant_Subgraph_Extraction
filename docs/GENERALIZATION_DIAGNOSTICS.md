# Generalization diagnostics

Branch: `fix/generalization-audit-and-training-pipeline`. Existing changes were
preserved on this branch; they are not automatically committed or pushed.

## Dumb It Down

We are checking what reaches the predictor, whether training groups are useful,
and whether the extra losses push the model in conflicting directions. This is
diagnosis, not a new performance experiment. It does not evaluate test molecules.

## Server instructions

Upload `scripts/diagnose_generalization.py` and the current project code it imports
(including `scripts/run_overnight.py` and `configs/preliminary_contrastive_test.yaml`).
From the project root inside tmux:

```bash
.venv-reu-server/bin/python scripts/diagnose_generalization.py --smoke --device cpu
```

Then run the actual train/validation checks:

```bash
.venv-reu-server/bin/python scripts/diagnose_generalization.py --device cpu
```

CPU is the default to avoid competing for GPU memory with the active search.
If CPU is too slow and a GPU has capacity, use `--device cuda` instead.
Each command creates a new `outputs_diagnostics_<timestamp>/diagnostics.json`.
Download that file for review. Logs are visible in the terminal.

The real run reads the existing overnight processed cache. It scans all train and
validation metadata and performs one loader pass to count populated environments.
Gradient probes use at most three training batches, before and after five optimizer
steps on those same batches. It neither modifies the cache nor saves trained models.
`--batches` supports 1..10 and `--warmup-steps` supports 0..50.

## Technical Explanation

### Graph path: findings from source review

- `src/models/fragment_ciga.py`, `FragmentCIGA.forward`: sample fragment edges,
  score them, choose eligible hard top-k edges, reconstruct atoms, then classify.
- The top-k decision uses detached scores. It does not itself have a gradient.
  Selected soft weights are passed into reconstruction and message gates; this
  provides a differentiable route to the scorer. Hard selection still limits what
  information the model can use, and this check does not establish optimal selection.
- `src/models/atom_reconstructor.py`, `reconstruct_batch_fast`: retained fragment
  endpoints determine retained atoms. Internal bonds and selected boundary bonds
  are reconstructed; both directed copies share their attributes and weights.
  Internal bonds use fragment importance and boundary bonds use edge scores.
- `src/models/fragment_ciga.py`, `AtomClassifier.forward`: uses PyG message masks
  with sigmoid disabled because weights are already scores; masks clear in finally.
  The GIN self term remains ungated. A zero edge gate is therefore not a zero model.
- `src/models/gnn_encoders.py`, `SimpleGNN`: this predictor uses GINE with bond
  attributes, not a bond-blind GIN. Bond-awareness is already implemented.
- Training samples edges; evaluation uses the sampler's evaluation behavior.
  This intentional distribution difference should be considered when interpreting
  graph retention. Our gradient probes are training-mode probes, not a test of
  generalization or full train/eval equivalence.

The script records feature widths, retained atom/edge counts, eligible fragment
edges, prediction gradients to scores, and the effect of zeroing gates or bond
attributes on the same reconstructed graph in classifier evaluation mode.
It asserts aligned edge features/weights, valid indices, no cross-molecule edges,
finite features/losses, and one prediction per input molecule. Existing reconstruction
tests cover additional fast/reference reconstruction invariants.

### Domain construction

`SuGARTrainer.prepare_irm_environments` computes each loader's environment IDs
separately, before metadata stripping. Validation domains do not define training
buckets. Bucket numbers across splits are not guaranteed to denote corresponding
scientific environments.

The configured value strategy sorts unique domain IDs and splits the domain list
into buckets. It balances the number of domains, not necessarily molecule counts
or labels. Numeric ID order is not automatically a meaningful scientific grouping.
Missing metadata normally causes random fallback; this diagnostic fails instead.

The report includes each domain/bucket's label counts and molecule-size min, max,
and mean. Batch coverage reports how many batches contain enough environments for
IRM. This is one shuffled pass, not a guarantee for every epoch. The IRM implementation
returns zero when fewer than two environments are present.

### Loss influence

Diagnostic coefficients are explicitly contrastive 0.1, diversity 0.5 and IRM 1.
They activate every component for measurement; they are not recommendations or the
current validation winner. Prediction losses and IRM penalties sum over views.
Contrastive embeddings concatenate across views; diversity compares compatible
fragment scores. Diversity does not directly update atom-classifier parameters.

For the same forward pass, autograd computes each weighted loss's gradient without
accumulating parameter .grad. Scorer and predictor parameter blocks are measured
separately. Norm means gradient size; cosine means directional agreement. Negative
cosines indicate opposition on that batch. Zero-gradient comparisons are null,
not evidence of agreement or conflict. Values are before gradient clipping and
Adam updates; they do not equal the final parameter change.

The short warmup uses the combined diagnostic objective. Probe forwards themselves
update BatchNorm statistics and consume random draws; this is a diagnostic trajectory,
not an exact replica of a production training run or a clean causal comparison.
Use real results across multiple batches to decide whether a controlled follow-up
is worthwhile. Do not infer a cause of the test gap from synthetic gradient values.

## Next steps

1. Inspect real domain balance, IRM batch coverage, scorer gradients and loss norms.
2. Identify repeatable observations rather than a single negative cosine.
3. If needed, test one change at a time: grouping, loss strength/schedule or retained
   graph size. Keep validation selection and matched seeds fixed.
4. Continue the existing overnight run unchanged. No model repair was justified by
   this synthetic check alone.
