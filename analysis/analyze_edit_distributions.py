#!/usr/bin/env python3
"""
Analyze Edit Distribution for Proofreading Cost Estimation.

Gets the edit history for proofread neurons across different connectome datasets
and calculates statistics on:
- Total number of edits per neuron
- Distribution of merge vs split operations
- Heavy-tail analysis (neurons with many edits)

This data is used to estimate computational costs for AI-based proofreading systems.

Usage:
    # Analyze mouse (MICrONS) dataset
    python analysis/analyze_edit_distributions.py --species mouse --sample-sizes 100 1000 5000

    # Analyze fly (FlyWire) dataset
    python analysis/analyze_edit_distributions.py --species fly --sample-sizes 100 1000

    # Get full dataset stats (no sampling)
    python analysis/analyze_edit_distributions.py --species mouse --full-dataset
"""

import sys
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import caveclient
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.connectome_visualizer import ConnectomeVisualizer


# ============================================================================
# Configuration
# ============================================================================

DATASET_CONFIG = {
    "mouse": {
        "datastack": "minnie65_public",
        "server": None,  # Uses default
        "proofread_table": "proofreading_status_and_strategy",
        "neuron_id_column": "valid_id",
        "expected_neuron_count": 200000,  # ~200k reported in MICrONS paper
    },
    "fly": {
        "datastack": "flywire_fafb_public",
        "server": None,
        "proofread_table": "proofread_neurons",
        "neuron_id_column": "pt_root_id",
        "expected_neuron_count": 10000,  # Approximate for FlyWire
    },
}


# ============================================================================
# Data Collection Functions
# ============================================================================

def get_proofread_neurons(species: str, auth_token: Optional[str] = None) -> List[int]:
    """
    Get list of all proofread neurons for a given species.

    Args:
        species: Dataset to query ("mouse" or "fly")
        auth_token: Optional authentication token for restricted datasets

    Returns:
        List of neuron IDs
    """
    config = DATASET_CONFIG[species]

    print(f"\n{'='*70}")
    print(f"Getting proofread neurons for {species.upper()}")
    print(f"{'='*70}")

    # Initialize CAVEclient
    if config["server"]:
        client = caveclient.CAVEclient(
            datastack_name=config["datastack"],
            server_address=config["server"],
            auth_token=auth_token
        )
    else:
        client = caveclient.CAVEclient(config["datastack"])

    # Query proofread neurons table
    print(f"Querying {config['proofread_table']} table...")
    table = client.materialize.query_table(config['proofread_table'])
    neuron_ids = list(table[config['neuron_id_column']])

    print(f"✓ Found {len(neuron_ids):,} proofread neurons")

    if len(neuron_ids) == 0:
        raise ValueError(f"No proofread neurons found for {species}!")

    return neuron_ids


def get_edit_history(neuron_id: int, species: str,
                     auth_token: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Get edit history for a single neuron.

    Args:
        neuron_id: Neuron ID to query
        species: Dataset species
        auth_token: Optional authentication token

    Returns:
        DataFrame with edit history, or None if error
    """
    try:
        visualizer = ConnectomeVisualizer(species=species, verbose=False)
        edit_history = visualizer.get_edit_history(neuron_id)
        return edit_history
    except Exception as e:
        print(f"Error getting edit history for neuron {neuron_id}: {e}")
        return None


def analyze_edit_history(edit_history) -> Dict[str, Any]:
    """
    Analyze edit history DataFrame to extract statistics.

    Args:
        edit_history: DataFrame from get_tabular_change_log() or dict

    Returns:
        Dictionary with edit statistics
    """
    # Handle case where edit_history is None, empty, or dict
    if edit_history is None:
        return {
            "total_edits": 0,
            "merge_edits": 0,
            "split_edits": 0,
            "merge_percentage": 0,
            "split_percentage": 0,
        }

    # Convert dict to DataFrame if needed (CAVEclient returns dict with neuron_id as key)
    if isinstance(edit_history, dict):
        if len(edit_history) == 0:
            return {
                "total_edits": 0,
                "merge_edits": 0,
                "split_edits": 0,
                "merge_percentage": 0,
                "split_percentage": 0,
            }
        # Extract the DataFrame from the dict (key is neuron_id, value is DataFrame)
        edit_history = list(edit_history.values())[0]

    if len(edit_history) == 0:
        return {
            "total_edits": 0,
            "merge_edits": 0,
            "split_edits": 0,
            "merge_percentage": 0,
            "split_percentage": 0,
        }

    # Count edit types
    # is_merge column indicates merge operations (True = merge, False = split)
    if 'is_merge' in edit_history.columns:
        merge_edits = (edit_history['is_merge'] == True).sum()
        split_edits = (edit_history['is_merge'] == False).sum()
    else:
        # Fallback: infer from before/after root IDs
        # Merge: multiple before -> one after
        # Split: one before -> multiple after
        merge_edits = 0
        split_edits = 0
        for idx, row in edit_history.iterrows():
            before_roots = row.get('before_root_ids', [])
            after_roots = row.get('after_root_ids', [])
            if len(before_roots) > len(after_roots):
                merge_edits += 1
            else:
                split_edits += 1

    total_edits = len(edit_history)

    return {
        "total_edits": int(total_edits),
        "merge_edits": int(merge_edits),
        "split_edits": int(split_edits),
        "merge_percentage": float(merge_edits / total_edits * 100) if total_edits > 0 else 0,
        "split_percentage": float(split_edits / total_edits * 100) if total_edits > 0 else 0,
    }


def collect_edit_statistics(neuron_ids: List[int], species: str,
                           auth_token: Optional[str] = None) -> pd.DataFrame:
    """
    Collect edit statistics for a list of neurons.

    Args:
        neuron_ids: List of neuron IDs to process
        species: Dataset species
        auth_token: Optional authentication token

    Returns:
        DataFrame with per-neuron edit statistics
    """
    results = []

    print(f"\nCollecting edit histories for {len(neuron_ids)} neurons...")
    for neuron_id in tqdm(neuron_ids, desc="Processing neurons"):
        edit_history = get_edit_history(neuron_id, species, auth_token)
        stats = analyze_edit_history(edit_history)
        stats["neuron_id"] = neuron_id
        results.append(stats)

    return pd.DataFrame(results)


# ============================================================================
# Analysis Functions
# ============================================================================

def calculate_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics from edit data.

    Args:
        df: DataFrame with per-neuron edit statistics

    Returns:
        Dictionary with summary statistics
    """
    total_neurons = len(df)

    # Total edits across all neurons
    total_edits = df['total_edits'].sum()
    total_merges = df['merge_edits'].sum()
    total_splits = df['split_edits'].sum()

    # Per-neuron statistics
    edits_per_neuron = df['total_edits']

    # Heavy-tail analysis (95th percentile)
    heavy_tail_threshold = edits_per_neuron.quantile(0.95)
    heavy_tail_neurons = df[df['total_edits'] > heavy_tail_threshold]

    # Distribution percentiles
    percentiles = {
        "p25": float(edits_per_neuron.quantile(0.25)),
        "p50": float(edits_per_neuron.quantile(0.50)),
        "p75": float(edits_per_neuron.quantile(0.75)),
        "p90": float(edits_per_neuron.quantile(0.90)),
        "p95": float(edits_per_neuron.quantile(0.95)),
        "p99": float(edits_per_neuron.quantile(0.99)),
    }

    return {
        "dataset_info": {
            "total_neurons": int(total_neurons),
            "neurons_with_edits": int((df['total_edits'] > 0).sum()),
            "neurons_without_edits": int((df['total_edits'] == 0).sum()),
        },
        "edit_totals": {
            "total_edits": int(total_edits),
            "total_merges": int(total_merges),
            "total_splits": int(total_splits),
            "merge_percentage": float(total_merges / total_edits * 100) if total_edits > 0 else 0,
            "split_percentage": float(total_splits / total_edits * 100) if total_edits > 0 else 0,
        },
        "per_neuron_stats": {
            "mean_edits": float(edits_per_neuron.mean()),
            "median_edits": float(edits_per_neuron.median()),
            "std_edits": float(edits_per_neuron.std()),
            "min_edits": int(edits_per_neuron.min()),
            "max_edits": int(edits_per_neuron.max()),
        },
        "distribution": {
            "percentiles": percentiles,
            "heavy_tail": {
                "threshold": float(heavy_tail_threshold),
                "neuron_count": len(heavy_tail_neurons),
                "percentage": float(len(heavy_tail_neurons) / total_neurons * 100),
                "edit_contribution": float(heavy_tail_neurons['total_edits'].sum() / total_edits * 100) if total_edits > 0 else 0,
            },
        },
    }


def extrapolate_to_full_dataset(sample_stats: Dict[str, Any],
                                full_dataset_size: int) -> Dict[str, Any]:
    """
    Extrapolate statistics from sample to full dataset.

    Args:
        sample_stats: Statistics from sample
        full_dataset_size: Total number of proofread neurons

    Returns:
        Dictionary with extrapolated estimates
    """
    sample_size = sample_stats["dataset_info"]["total_neurons"]
    scaling_factor = full_dataset_size / sample_size

    # Scale edit counts
    total_edits_estimate = int(sample_stats["edit_totals"]["total_edits"] * scaling_factor)
    merge_edits_estimate = int(sample_stats["edit_totals"]["total_merges"] * scaling_factor)
    split_edits_estimate = int(sample_stats["edit_totals"]["total_splits"] * scaling_factor)

    return {
        "sample_size": sample_size,
        "full_dataset_size": full_dataset_size,
        "scaling_factor": float(scaling_factor),
        "estimated_totals": {
            "total_edits": total_edits_estimate,
            "merge_edits": merge_edits_estimate,
            "split_edits": split_edits_estimate,
        },
        "per_neuron_estimates": sample_stats["per_neuron_stats"],
        "note": "Per-neuron statistics assumed to be representative across full dataset"
    }


def create_visualizations(df: pd.DataFrame, summary_stats: Dict[str, Any],
                         species: str, sample_size: int, output_dir: Path) -> None:
    """
    Create visualizations of edit distribution statistics.

    Args:
        df: DataFrame with edit statistics per neuron
        summary_stats: Dictionary with summary statistics
        species: Dataset species
        sample_size: Number of neurons in sample
        output_dir: Directory to save figures
    """
    # Create figures directory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Get statistics
    per_neuron = summary_stats["per_neuron_stats"]
    edit_totals = summary_stats["edit_totals"]
    distribution = summary_stats["distribution"]

    # Create a 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Edit Distribution Analysis - {species.upper()} (n={sample_size})',
                 fontsize=16, fontweight='bold')

    # ---- Plot 1: Histogram of edits per neuron ----
    ax1 = axes[0, 0]
    ax1.hist(df['total_edits'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(per_neuron['mean_edits'], color='red', linestyle='--', linewidth=2, label=f'Mean: {per_neuron["mean_edits"]:.0f}')
    ax1.axvline(per_neuron['median_edits'], color='orange', linestyle='--', linewidth=2, label=f'Median: {per_neuron["median_edits"]:.0f}')
    ax1.set_xlabel('Edits per Neuron', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Edits per Neuron', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # ---- Plot 2: Merge vs Split bar chart ----
    ax2 = axes[0, 1]
    merge_count = edit_totals['total_merges']
    split_count = edit_totals['total_splits']
    merge_pct = edit_totals['merge_percentage']
    split_pct = edit_totals['split_percentage']

    bars = ax2.bar(['Merge', 'Split'], [merge_count, split_count],
                   color=['#ff6b6b', '#4ecdc4'], alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Total Operations', fontsize=11, fontweight='bold')
    ax2.set_title('Merge vs Split Operations', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, [merge_pct, split_pct])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%\n({int(height):,})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ---- Plot 3: Box plot of edit distribution ----
    ax3 = axes[1, 0]
    bp = ax3.boxplot([df['total_edits']], vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_edgecolor('black')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)

    ax3.set_ylabel('Edits per Neuron', fontsize=11, fontweight='bold')
    ax3.set_title('Edit Distribution Statistics', fontsize=12, fontweight='bold')
    ax3.set_xticks([1])
    ax3.set_xticklabels(['All Neurons'])
    ax3.grid(axis='y', alpha=0.3)

    # Add percentile annotations
    percentiles = distribution['percentiles']
    ax3.text(1.3, percentiles['p25'], f"Q1: {percentiles['p25']:.0f}", va='center', fontsize=9)
    ax3.text(1.3, percentiles['p50'], f"Median: {percentiles['p50']:.0f}", va='center', fontsize=9, color='red', fontweight='bold')
    ax3.text(1.3, percentiles['p75'], f"Q3: {percentiles['p75']:.0f}", va='center', fontsize=9)
    ax3.text(1.3, percentiles['p95'], f"P95: {percentiles['p95']:.0f}", va='center', fontsize=9, color='darkred')

    # ---- Plot 4: Heavy-tail analysis scatter plot ----
    ax4 = axes[1, 1]
    sorted_edits = df['total_edits'].sort_values(ascending=False).reset_index(drop=True)
    heavy_tail_threshold = distribution['heavy_tail']['threshold']

    # Color neurons above heavy-tail threshold differently
    colors = ['red' if x >= heavy_tail_threshold else 'steelblue' for x in sorted_edits]
    ax4.scatter(range(len(sorted_edits)), sorted_edits, c=colors, alpha=0.6, s=30)
    ax4.axhline(heavy_tail_threshold, color='red', linestyle='--', linewidth=2,
               label=f'Heavy-tail (P95): {heavy_tail_threshold:.0f}')
    ax4.axhline(per_neuron['mean_edits'], color='orange', linestyle='--', linewidth=1,
               label=f'Mean: {per_neuron["mean_edits"]:.0f}')

    ax4.set_xlabel('Neuron Rank (sorted by edit count)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Edits per Neuron', fontsize=11, fontweight='bold')
    ax4.set_title(f'Heavy-Tail Pattern ({distribution["heavy_tail"]["neuron_count"]} neurons = {distribution["heavy_tail"]["percentage"]:.1f}%)',
                 fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure
    figure_path = figures_dir / f"edit_distribution_{species}_n{sample_size}.png"
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization to {figure_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze edit distributions for proofreading cost estimation"
    )
    parser.add_argument(
        "--species",
        choices=["mouse", "fly"],
        required=True,
        help="Dataset to analyze"
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=[100, 1000, 5000],
        help="Sample sizes to analyze (default: 100 1000 5000)"
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Analyze full dataset instead of sampling (WARNING: very slow)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/edit_distributions"),
        help="Output directory for reports (default: reports/edit_distributions)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Get all proofread neurons
    all_neurons = get_proofread_neurons(args.species)

    # Determine which sample sizes to run
    if args.full_dataset:
        sample_sizes = [len(all_neurons)]
        print(f"\n⚠️  WARNING: Analyzing full dataset ({len(all_neurons):,} neurons)")
        print("This may take several hours to complete!")
    else:
        sample_sizes = [s for s in args.sample_sizes if s <= len(all_neurons)]
        if len(sample_sizes) != len(args.sample_sizes):
            print(f"\n⚠️  Some sample sizes exceed dataset size, adjusted to: {sample_sizes}")

    # Process each sample size
    all_results = {}

    for sample_size in sample_sizes:
        print(f"\n{'='*70}")
        print(f"Analyzing sample size: {sample_size:,} neurons")
        print(f"{'='*70}")

        # Sample neurons
        if sample_size == len(all_neurons):
            sampled_neurons = all_neurons
            print(f"Using full dataset: {len(sampled_neurons):,} neurons")
        else:
            sampled_neurons = random.sample(all_neurons, sample_size)
            print(f"Sampled {len(sampled_neurons):,} neurons from {len(all_neurons):,}")

        # Collect edit statistics
        edit_df = collect_edit_statistics(sampled_neurons, args.species)

        # Calculate summary statistics
        summary_stats = calculate_summary_statistics(edit_df)

        # Extrapolate to full dataset
        extrapolated_stats = extrapolate_to_full_dataset(summary_stats, len(all_neurons))

        # Combine results
        results = {
            "species": args.species,
            "timestamp": datetime.now().isoformat(),
            "sample_size": sample_size,
            "full_dataset_size": len(all_neurons),
            "summary_statistics": summary_stats,
            "extrapolated_to_full_dataset": extrapolated_stats,
        }

        all_results[f"sample_{sample_size}"] = results

        # Save individual report
        output_file = args.output_dir / f"edit_distribution_{args.species}_n{sample_size}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved report to {output_file}")

        # Create visualizations
        create_visualizations(edit_df, summary_stats, args.species, sample_size, args.output_dir)

        # Print summary
        print(f"\n{'='*70}")
        print(f"SUMMARY (n={sample_size:,})")
        print(f"{'='*70}")
        print(f"Total edits: {summary_stats['edit_totals']['total_edits']:,}")
        print(f"  Merges: {summary_stats['edit_totals']['total_merges']:,} ({summary_stats['edit_totals']['merge_percentage']:.1f}%)")
        print(f"  Splits: {summary_stats['edit_totals']['total_splits']:,} ({summary_stats['edit_totals']['split_percentage']:.1f}%)")
        print(f"\nPer-neuron average: {summary_stats['per_neuron_stats']['mean_edits']:.2f} edits")
        print(f"Heavy-tail neurons: {summary_stats['distribution']['heavy_tail']['neuron_count']} ({summary_stats['distribution']['heavy_tail']['percentage']:.1f}%)")
        print(f"\nExtrapolated to full dataset ({len(all_neurons):,} neurons):")
        print(f"  Total edits: {extrapolated_stats['estimated_totals']['total_edits']:,}")
        print(f"  Merge edits: {extrapolated_stats['estimated_totals']['merge_edits']:,}")
        print(f"  Split edits: {extrapolated_stats['estimated_totals']['split_edits']:,}")

    # Save combined report
    combined_file = args.output_dir / f"edit_distribution_{args.species}_all_samples.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved combined report to {combined_file}")

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
