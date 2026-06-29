# SuGAR Ensemble - Quick Start Guide

## Quick Start (3 Steps)

### Step 1: Prepare Data (If Needed)
```bash
# Skip if you already have processed data
python src/data/preprocess.py --config configs/experiment.yaml
```

### Step 2: Train Model
```bash
python -m train_sugar \
  --processed_dir data/processed \
  --batch_size 32 \
  --gnn_layers 3 \
  --hidden_dim 128 \
  --edge_ratio 0.5 \
  --epochs 100 \
```

The script will:
1. Load training/validation/test splits
2. Infer feature dimensions automatically
3. Train with early stopping
4. Report validation performance each epoch
5. Evaluate on test set (if available)

## Common Commands

### Train with Config File
```bash
python train_sugar.py --config configs/experiment.yaml
```

### GPU Training
```bash
python train_sugar.py --processed_dir data/processed --device cuda
```

### CPU Training (Debugging)
```bash
python train_sugar.py --processed_dir data/processed --device cpu --batch_size 8
```

### Smaller Model (Memory-Constrained)
```bash
python train_sugar.py \
  --processed_dir data/processed \
  --gnn_layers 2 \
  --hidden_dim 64 \
  --batch_size 16
```

### More Aggressive Edge Selection
```bash
python train_sugar.py \
  --processed_dir data/processed \
  --edge_ratio 0.25 \
  --gnn_layers 4 \
  --hidden_dim 256
```

## Configuration Parameters

### Data Parameters
- `--processed_dir`: Path to processed data (default: `data/processed`)
- `--batch_size`: Batch size (default: 32)
- `--num_workers`: DataLoader workers (default: 0)

### Model Parameters
- `--num_classes`: Output classes (default: 2)
- `--gnn_layers`: GNN depth (default: 3) [2-5 recommended]
- `--hidden_dim`: GNN hidden size (default: 128) [64-256 recommended]
- `--edge_ratio`: Invariant subgraph ratio (default: 0.5) [0.1-0.9]
- `--dropout`: Dropout rate (default: 0.1)

### Training Parameters
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: L2 regularization (default: 0.0)
- `--epochs`: Max epochs (default: 100)
- `--patience`: Early stopping patience (default: 20)
- `--device`: Device to use (default: `cuda`)

## Pipeline Explanation

### What Does the Model Learn?

1. **Fragment Graph Construction**
   - Molecular graph decomposed into functional groups (fragments)
   - Fragment-level edges represent inter-fragment bonds
   - Each fragment node has pooled atomic features

2. **Invariant Subgraph Extraction**
   - GNN learns to identify "important" fragment edges
   - Top-50% (configurable) edges selected
   - Selected fragments form the invariant subgraph

3. **Classification**
   - Atoms from selected fragments reconstructed
   - Features pooled to graph level
   - Final prediction on selected subgraph

### Why This Works

- **Robustness**: Model learns which parts of molecules matter most
- **Interpretability**: Invariant edges show important functional groups
- **Multi-View Ready**: Can use different fragmentations for ensemble

## Expected Results

### BBBP Dataset (Example)
- **Baseline Accuracy**: ~65%
- **Expected with SuGAR**: ~68-72%
- **Training Time**: ~5-10 minutes (GPU), ~30 minutes (CPU)

### Convergence Pattern
```
Epoch 1/100 | Train Loss: 0.6931 | Val Loss: 0.6899 | Val Acc: 0.5200
Epoch 2/100 | Train Loss: 0.6820 | Val Loss: 0.6791 | Val Acc: 0.5800
Epoch 3/100 | Train Loss: 0.6234 | Val Loss: 0.5987 | Val Acc: 0.6400
...
Epoch 50/100 | Train Loss: 0.4123 | Val Loss: 0.4567 | Val Acc: 0.7234
...
Early stopping at epoch 75
```

## Troubleshooting

### Error: "No training data found"
```
Solution: Run preprocessing first
python src/data/preprocess.py --config configs/bbbp.yaml
```

### Error: "CUDA out of memory"
```
Solution: Reduce batch size and model size
python train_sugar.py --batch_size 8 --hidden_dim 64 --gnn_layers 2
```

### Error: "Feature dimension mismatch"
```
Solution: Check processed data files are valid
python test_sugar_pipeline.py --data_path data/processed/train.pt --num_samples 1
```

### Model not improving
```
Possible issues:
1. Too high learning rate → use 0.0005
2. Too small edge_ratio → increase to 0.7
3. Dataset imbalance → use class weights
4. Insufficient layers → increase gnn_layers to 4
```

## Next Steps

1. **Run test pipeline** to validate end-to-end
2. **Train model** with default parameters
3. **Adjust hyperparameters** based on validation performance
4. **Compare results** with baseline models
5. **(Future) Add multi-view learning** with BRICS fragmentation

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Molecular Input (SMILES)                           │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  Atom Graph Construction │
        │  (RDKit features)        │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  Fragment Decomposition  │
        │  (GroupGraph)            │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  Fragment Graph          │
        │  (Pooled features)       │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  GNN + Edge Scoring      │
        │  (SimpleCIGA)            │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  Top-K Edge Selection    │
        │  (Invariant Subgraph)    │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  Atom Reconstruction     │
        │  (Selected fragments)    │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  Classification          │
        │  (Pool & Predict)        │
        └────────────┬─────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  Final Prediction        │
        └──────────────────────────┘
```

## Documentation

- **Detailed Architecture**: See `SUGAR_ARCHITECTURE.md`
- **Implementation Details**: See `INTEGRATION_SUMMARY.md`
- **Source Code**: See comments in each file

## Support & Questions

For implementation questions, refer to:
1. `INTEGRATION_SUMMARY.md` - What was implemented and why
2. `SUGAR_ARCHITECTURE.md` - How the pipeline works
3. Source code comments - Implementation details

---

**Ready to train!** Run `python train_sugar.py --help` to see all options.
