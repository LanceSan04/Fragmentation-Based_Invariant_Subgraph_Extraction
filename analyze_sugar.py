#!/usr/bin/env python
"""Utility script to analyze invariant subgraphs extracted by SuGAR ensemble."""

import torch
import argparse
from pathlib import Path
from tqdm import tqdm

from src.data.dataloader import load_processed_split
from src.data.pyg_batch import batch_molecules
from src.models.sugar_ensemble import SuGAREnsemble
from src.models.fragment_reconstruction import batch_reconstruct_from_fragment_edges


def analyze_invariant_subgraph(model, batch_data, molecule_idx=0):
    """Extract and analyze invariant subgraph for a specific molecule.
    
    Args:
        model: Trained SuGAREnsemble
        batch_data: Batched MoleculeData
        molecule_idx: Index of molecule to analyze
        
    Returns:
        dict: Analysis results
    """
    with torch.no_grad():
        subgraph = model.get_invariant_subgraph(batch_data, view_idx=0)
        subgraph = subgraph[molecule_idx]
        
        # Get full molecule for comparison
        full_data = batch_data.to_data_list()[molecule_idx]
        
        # Statistics
        analysis = {
            "molecule_idx": molecule_idx,
            "full_atoms": full_data.x.size(0),
            "subgraph_atoms": subgraph.x.size(0),
            "atom_retention_pct": 100.0 * subgraph.x.size(0) / full_data.x.size(0),
            "full_edges": full_data.edge_index.size(1) if full_data.edge_index.numel() > 0 else 0,
            "subgraph_edges": subgraph.edge_index.size(1) if subgraph.edge_index.numel() > 0 else 0,
        }
        
        if analysis["full_edges"] > 0:
            analysis["edge_retention_pct"] = 100.0 * analysis["subgraph_edges"] / analysis["full_edges"]
        
        return analysis, subgraph


def batch_analyze(model, data_loader, num_samples=100):
    """Analyze invariant subgraphs across a batch of molecules.
    
    Args:
        model: Trained SuGAREnsemble
        data_loader: DataLoader with molecules
        num_samples: Number of molecules to analyze
        
    Returns:
        list: Analysis for each molecule
    """
    model.eval()
    results = []
    
    count = 0
    for batch in data_loader:
        for mol_idx in range(len(batch.to_data_list())):
            if count >= num_samples:
                break
            
            analysis, _ = analyze_invariant_subgraph(model, batch, mol_idx)
            results.append(analysis)
            count += 1
        
        if count >= num_samples:
            break
    
    return results


def print_analysis_summary(results):
    """Print summary statistics from analysis results.
    
    Args:
        results: List of analysis dicts
    """
    if not results:
        print("No results to summarize")
        return
    
    print("\n" + "=" * 80)
    print("INVARIANT SUBGRAPH ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nAnalyzed {len(results)} molecules\n")
    
    # Calculate statistics
    atom_retention = [r["atom_retention_pct"] for r in results]
    edge_retention = [r["edge_retention_pct"] for r in results if "edge_retention_pct" in r]
    
    print("Atom Retention (%):")
    print(f"  Mean: {sum(atom_retention)/len(atom_retention):.2f}%")
    print(f"  Min:  {min(atom_retention):.2f}%")
    print(f"  Max:  {max(atom_retention):.2f}%")
    print(f"  Std:  {_std_dev(atom_retention):.2f}%")
    
    if edge_retention:
        print("\nEdge Retention (%):")
        print(f"  Mean: {sum(edge_retention)/len(edge_retention):.2f}%")
        print(f"  Min:  {min(edge_retention):.2f}%")
        print(f"  Max:  {max(edge_retention):.2f}%")
        print(f"  Std:  {_std_dev(edge_retention):.2f}%")
    
    print("\nDetailed Results (first 10 molecules):")
    print("-" * 80)
    for i, r in enumerate(results[:10]):
        print(f"\nMolecule {i}:")
        print(f"  Atoms: {r['full_atoms']} → {r['subgraph_atoms']} ({r['atom_retention_pct']:.1f}%)")
        print(f"  Edges: {r['full_edges']} → {r['subgraph_edges']} "
              f"({r.get('edge_retention_pct', 0):.1f}%)")


def _std_dev(values):
    """Calculate standard deviation."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def export_analysis(results, output_path):
    """Export analysis results to CSV.
    
    Args:
        results: List of analysis dicts
        output_path: Path to save CSV
    """
    import csv
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "molecule_idx",
                "full_atoms",
                "subgraph_atoms",
                "atom_retention_pct",
                "full_edges",
                "subgraph_edges",
                "edge_retention_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults exported to {output_path}")


def main(args):
    """Main analysis function."""
    # Load model
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    print("Loading model...")
    # Placeholder for model loading - user would need to specify model architecture
    # This is a template showing how to use the analysis functions
    
    # For now, just show the structure
    print(f"Would load model from: {args.model_path}")
    print(f"Would analyze {args.num_samples} samples from: {args.data_path}")
    print(f"Would export results to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze invariant subgraphs from SuGAR ensemble")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/sugar_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/test.pt",
        help="Path to processed data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of molecules to analyze",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="invariant_analysis.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    
    args = parser.parse_args()
    main(args)

    # Example of how to use the analysis functions:
    print("""
    
Example Usage:

1. Load trained model and test data:
   >>> model = torch.load('checkpoints/sugar_model.pth')
   >>> batch = load_test_batch('data/processed/test.pt')

2. Analyze single molecule:
   >>> analysis, subgraph = analyze_invariant_subgraph(model, batch, molecule_idx=0)
   >>> print(f"Atoms selected: {analysis['subgraph_atoms']}/{analysis['full_atoms']}")

3. Analyze batch:
   >>> results = batch_analyze(model, test_loader, num_samples=100)
   >>> print_analysis_summary(results)

4. Export results:
   >>> export_analysis(results, 'analysis.csv')
    """)
