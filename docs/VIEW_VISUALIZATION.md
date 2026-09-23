# View visualization

Command: `python main.py experiment visualize-views CHECKPOINT --indices 0 1 2 3 4 --mode evaluation --device cuda --output-dir OUTPUT`

Use `--mode sampling-demo --seed 42` for sampler-only variation. Dropout and BatchNorm remain in evaluation mode; no gradients or checkpoint writes. Each view is called exactly once per selected molecule. Evaluation uses all eligible edges. Demonstration seeds are 42+view index and counters reset on load. Input order/indices are recorded and affect the sequence of random draws.

Default outputs: outputs/assay/visualize_views/TIMESTAMP. Open index.html. Each molecule has input.svg, per-view sampled-edge and selected-graph SVGs, selected PNGs, trace.json and its own index.html. Trace records scores, eligibility, exact selected mask, original atom/bond IDs and probabilities. Manifest records checkpoint hash, validation split, IDs, mode and seed. Drawings retain unselected molecule as context; highlights identify selected content. Both modes show each view's own graph, not the shared ensemble graph. No AUC is calculated on five illustrative molecules.

Implementation adds provenance IDs to fast reconstruction and returns them only through the optional view intermediates. Predictor inputs and training calculations are unchanged. Requires RDKit drawing support already used by the project. Uses the checkpoint's processed directory unless --processed-dir overrides it.

Current scope: own-view highlights, scores in JSON, probability per view and mean vote. Separate fragment-node diagram and score heatmap are future presentation enhancements. Selections are model behavior, not proof of causal chemistry.
