"""SuGAR Ensemble Integration Architecture

This module documents the complete integration of the SuGAr pipeline into REU_Summer_2026.

PIPELINE ARCHITECTURE:
======================

The pipeline processes molecular graphs through the following stages:

1. MOLECULE LOADING & PREPROCESSING
   - Input: SMILES strings
   - Converts to atom-level graphs with RDKit features
   - Creates fragment-level representation via GroupGraphFragmenter
   - Pools atom features to fragment level
   - Output: MoleculeData with both atom and fragment provenance

2. FRAGMENT GRAPH CONSTRUCTION  
   - Converts fragment-level provenance to PyG graph format
   - Fragment nodes: pooled atom features
   - Fragment edges: inter-fragment connectivity
   - Edge attributes: learned edge features between fragments
   - Output: PyG Data with fragment-level representation

3. INVARIANT SUBGRAPH EXTRACTION (CIGA)
   - GNN encoder processes fragment graph
   - Edge attention mechanism scores fragment edges for importance
   - Top-k edge selection based on importance scores
   - Creates "invariant subgraph" containing important fragments
   - Output: Binary mask indicating selected fragment edges

4. ATOM-LEVEL RECONSTRUCTION
   - Selected fragment edges expanded back to atom level
   - Includes atoms from selected fragments
   - Includes internal bonds within fragments
   - Includes inter-fragment bonds in the selection
   - Output: Atom-level subgraph for classification

5. CLASSIFICATION & AGGREGATION
   - Each reconstructed subgraph pooled to graph-level features
   - Separate classifiers per view
   - Predictions aggregated across views
   - Output: Final predictions

KEY COMPONENTS:
===============

src/models/fragment_gnn.py:
  - SimpleGNN: Graph neural network for embedding nodes
  - EdgeScorer: Scores edges based on node importance
  - SimpleCIGA: Main model for invariant subgraph extraction
    * forward(): Returns predictions and edge masks
    * _select_top_k_edges(): Selects top-k fragment edges per graph

src/models/fragment_graph.py:
  - build_fragment_graph(): Creates PyG Data from fragment provenance
  - build_fragment_graph_batch(): Batches fragment graphs

src/models/fragment_reconstruction.py:
  - reconstruct_from_fragment_edges(): Expands fragment selection to atoms
  - batch_reconstruct_from_fragment_edges(): Batch processing

src/models/sugar_ensemble.py:
  - SuGAREnsemble: Main orchestrator model
    * forward(): End-to-end inference
    * get_invariant_subgraph(): Extracts subgraph for analysis

src/training/sugar_trainer.py:
  - SuGARTrainer: Training loop handler
    * train_epoch(): Single epoch training
    * evaluate(): Validation/test evaluation
    * train(): Full training with early stopping

train_sugar.py:
  - Main training script with argument parsing
  - DataLoader setup
  - Model initialization
  - Training orchestration

test_sugar_pipeline.py:
  - End-to-end pipeline test
  - Demonstrates all stages
  - Validates intermediate outputs

USAGE:
======

1. PREPARE DATA
   python src/data/preprocess.py --config configs/bbbp.yaml

   Creates data/processed/{train,valid,test}.pt

2. TEST PIPELINE (Optional - for debugging)
   python test_sugar_pipeline.py --data_path data/processed/train.pt --num_samples 2

   Shows:
   - Data loading and batching
   - Fragment graph construction
   - Model inference
   - Invariant subgraph extraction
   - Atom reconstruction

3. TRAIN MODEL
   python train_sugar.py \
     --processed_dir data/processed \
     --batch_size 32 \
     --gnn_layers 3 \
     --hidden_dim 128 \
     --edge_ratio 0.5 \
     --epochs 100 \
     --learning_rate 0.001 \
     --device cuda

   Optional: Use config file
   python train_sugar.py --config configs/bbbp.yaml

CONFIGURATION:
===============

Key hyperparameters in train_sugar.py:

  --num_classes: Number of output classes (default: 2)
  --gnn_layers: Number of GNN layers (default: 3)
  --hidden_dim: GNN hidden dimension (default: 128)
  --edge_ratio: Ratio of edges to keep in invariant subgraph (default: 0.5)
  --dropout: Dropout rate (default: 0.1)
  --learning_rate: Optimizer learning rate (default: 0.001)
  --weight_decay: L2 regularization (default: 0.0)
  --epochs: Maximum training epochs (default: 100)
  --patience: Early stopping patience (default: 20)

MULTI-VIEW EXTENSION (FUTURE):
===============================

Current implementation uses num_views=1 (single fragmentation method).

To extend to multiple views:

1. Update src/fragmentation/multi_view.py
   - Add more fragmentation methods (BRICS, RECAP, etc.)

2. Modify SuGAREnsemble
   - Increase num_views parameter
   - Each view uses different fragmentation

3. Training automatically handles multiple views
   - Processes each view independently
   - Aggregates predictions

DATA FLOW:
==========

MoleculeData (atom + fragment provenance)
    |
    v
build_fragment_graph_batch()
    |
    v
Fragment-level PyG Batch
    |
    v
SimpleCIGA (fragment GNN + edge scoring)
    |
    v
Edge mask (top-k fragment edges)
    |
    v
batch_reconstruct_from_fragment_edges()
    |
    v
Atom-level subgraphs (one per view)
    |
    v
Feature pooling + Classification
    |
    v
Aggregation
    |
    v
Final predictions

IMPORTS & DEPENDENCIES:
=======================

Internal:
- src.data.dataloader: DataLoader utilities
- src.data.pyg_batch: MoleculeData batching
- src.data.atom_graph: Atom feature extraction
- src.fragmentation.group_graph: Fragment creation
- src.models.fragment_gnn: Fragment-level GNN
- src.models.fragment_graph: Fragment graph construction
- src.models.fragment_reconstruction: Atom reconstruction
- src.models.sugar_ensemble: Main ensemble model

External:
- torch, torch.nn: PyTorch
- torch_geometric.data, torch_geometric.nn: Graph neural networks
- torch_geometric.loader: Data loaders
- torch_geometric.utils: Graph utilities
- rdkit.Chem: Molecular processing

TROUBLESHOOTING:
================

Issue: Feature dimension mismatch
  -> Ensure fragment features are correctly pooled from atoms
  -> Check fragment_feat_dim matches fragment_batch.x.size(1)

Issue: Edge mask shape incorrect
  -> Verify fragment_edge_index matches mask dimensions
  -> Check batch offsets in PyG Batch

Issue: Reconstruction fails
  -> Check atom_to_fragment mapping in MoleculeData
  -> Verify fragment_to_atom_index tensor format

Issue: Out of memory
  -> Reduce batch_size
  -> Reduce gnn_hidden_dim
  -> Reduce num_gnn_layers
"""


# If this file is imported, ensure all critical components are available
def verify_imports():
    """Verify all required modules can be imported."""
    try:
        from src.models.fragment_gnn import SimpleCIGA
        from src.models.fragment_graph import build_fragment_graph_batch
        from src.models.fragment_reconstruction import batch_reconstruct_from_fragment_edges
        from src.models.sugar_ensemble import SuGAREnsemble
        from src.training.sugar_trainer import SuGARTrainer
        print("✓ All SuGAR components successfully imported")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


if __name__ == "__main__":
    verify_imports()
