#!/usr/bin/env python3
"""
Compute Cost Analysis for ConnectomeBench Benchmark.

Analyzes three cost dimensions for the benchmark:
1. LLM API costs (token-based pricing for different models)
2. Proofreading complexity metrics (vertex counts, interface distances)
3. Full dataset cost projections with confidence intervals

Provides statistical analysis of edit distributions to identify heavy-tail
neurons and project costs from sample to full dataset.

Usage:
    # Analyze fly dataset (proof-of-concept)
    python analysis/analyze_compute_costs.py --species fly

    # Analyze human dataset
    python analysis/analyze_compute_costs.py --species human

    # Analyze all available datasets
    python analysis/analyze_compute_costs.py --species all

Output:
    - JSON reports: reports/compute_costs/compute_costs_analysis_{species}.json
    - Visualizations: reports/compute_costs/figures/
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis_utils import (
    MODEL_PRICING,
    estimate_token_count,
    calculate_api_costs,
    analyze_edit_distribution,
    analyze_edit_complexity,
    project_full_dataset_costs,
    clean_report_for_json,
)


# ============================================================================
# Configuration
# ============================================================================

# Full dataset sizes (estimated edited neurons from get_delta_roots)
FULL_DATASET_SIZES = {
    "fly": 10000,        # Approximate for Drosophila FlyEM dataset
    "human": 100000,     # Approximate for H01 human connectome
    "zebrafish": 50000,  # Approximate for Fish1 zebrafish dataset
}

# Images per sample (varies by task type)
# Binary task (merge identification): typically 3-5 views
# Multiple choice task (merge comparison): typically 5-6 views
IMAGES_PER_SAMPLE = {
    "binary": 5,
    "multiple_choice": 6,
}


# ============================================================================
# Data Loading
# ============================================================================

def load_training_data(species: str, data_dir: Path = None) -> Optional[pd.DataFrame]:
    """
    Load training data JSON for specified species.

    Args:
        species: Species name ("fly", "human", "zebrafish")
        data_dir: Directory containing data files (default: current directory)

    Returns:
        DataFrame with training data, or None if file not found
    """
    if data_dir is None:
        data_dir = Path.cwd()

    # Try multiple patterns for finding data files
    search_patterns = [
        # Exact pattern
        data_dir / f"training_data_{species}.json",
        # Pattern with timestamp
        list(data_dir.glob(f"test_data_{species}/training_data_*.json")),
        # Pattern without directory prefix
        data_dir / f"test_data_{species}" / f"training_data_*.json",
    ]

    data_file = None

    for pattern in search_patterns:
        if isinstance(pattern, Path):
            if pattern.exists():
                data_file = pattern
                break
        elif isinstance(pattern, list) and pattern:
            # glob returns a list; use the most recent file (highest timestamp)
            data_file = sorted(pattern)[-1]
            break

    if data_file is None:
        # Try glob pattern directly
        glob_results = list(data_dir.glob(f"**/training_data_*{species}*.json"))
        if glob_results:
            data_file = sorted(glob_results)[-1]

    if data_file is None:
        print(f"Warning: No training_data file found for {species}. Skipping {species} dataset.")
        return None

    print(f"Loading {species} training data from {data_file}...")

    try:
        with open(data_file, 'r') as f:
            data = json.load(f)

        # Handle nested structure: if data is a list with nested lists, flatten it
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            # Nested list structure: flatten
            flattened = []
            for sublist in data:
                if isinstance(sublist, list):
                    flattened.extend(sublist)
            data = flattened

        df = pd.DataFrame(data)
        print(f"  Loaded {len(df)} records, {df['neuron_id'].nunique()} unique neurons")
        return df

    except Exception as e:
        print(f"Error loading {data_file}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Analysis Pipeline
# ============================================================================

def run_analysis(df: pd.DataFrame, species: str) -> Dict[str, Any]:
    """
    Run complete compute cost analysis pipeline.

    Args:
        df: Training data DataFrame
        species: Species name for reporting

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*70}")
    print(f"Analyzing Compute Costs: {species.upper()}")
    print(f"{'='*70}")

    results = {
        "species": species,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    # A. Basic Dataset Info
    print("\n[1/5] Analyzing dataset info...")

    # Convert timestamp to datetime if it's in Unix format
    if 'timestamp' in df.columns:
        try:
            if df['timestamp'].dtype == 'int64' or df['timestamp'].dtype == 'float64':
                # Unix timestamp, convert to datetime
                date_min = pd.to_datetime(df['timestamp'].min(), unit='s').isoformat()
                date_max = pd.to_datetime(df['timestamp'].max(), unit='s').isoformat()
            else:
                date_min = pd.to_datetime(df['timestamp'].min()).isoformat()
                date_max = pd.to_datetime(df['timestamp'].max()).isoformat()
        except:
            date_min = None
            date_max = None
    else:
        date_min = None
        date_max = None

    results["dataset_info"] = {
        "num_samples": len(df),
        "num_unique_neurons": int(df['neuron_id'].nunique()),
        "date_range": [date_min, date_max],
    }
    print(f"  ✓ {results['dataset_info']['num_samples']} samples")
    print(f"  ✓ {results['dataset_info']['num_unique_neurons']} unique neurons")

    # B. Edit Distribution Analysis
    print("\n[2/5] Analyzing edit distribution...")
    edit_dist = analyze_edit_distribution(df)
    results["edit_distribution"] = edit_dist
    print(f"  ✓ Mean edits per neuron: {edit_dist['basic_stats']['mean']:.2f}")
    print(f"  ✓ Heavy-tail neurons (>95th percentile): {edit_dist['heavy_tail']['neuron_count']}")
    print(f"    - Threshold: {edit_dist['heavy_tail']['threshold']:.1f} edits")
    print(f"    - Percentage of neurons: {edit_dist['heavy_tail']['percentage']:.1f}%")

    # C. Complexity Analysis
    print("\n[3/5] Analyzing complexity metrics...")
    complexity = analyze_edit_complexity(df)
    results["complexity_metrics"] = complexity
    if 'vertex_counts' in complexity:
        print(f"  ✓ Mean vertex count (before): {complexity['vertex_counts']['before']['mean']:.0f}")
        print(f"  ✓ Mean interface distance: {complexity['interface_distances']['mean']:.1f}")
    if 'correlation_edits_vs_complexity' in complexity:
        print(f"  ✓ Edit count vs complexity correlation: {complexity['correlation_edits_vs_complexity']:.3f}")

    # D. API Cost Estimation (using average images per sample)
    print("\n[4/5] Calculating API costs for sample...")
    avg_images = (IMAGES_PER_SAMPLE["binary"] + IMAGES_PER_SAMPLE["multiple_choice"]) / 2
    api_costs = calculate_api_costs(
        num_samples=len(df),
        num_images_per_sample=int(avg_images),
        model_pricing=MODEL_PRICING,
    )
    results["api_costs_sample"] = {
        "per_model": api_costs["per_model"],
        "total_all_models": api_costs["total_all_models"],
        "avg_per_model": api_costs["avg_per_model"],
    }

    print(f"  ✓ Total API cost (all models): ${api_costs['total_all_models']:.2f}")
    print(f"  ✓ Average per model: ${api_costs['avg_per_model']:.2f}")
    print("\n  Cost breakdown by model:")
    for model, costs in sorted(api_costs["per_model"].items(), key=lambda x: x[1]["total_cost"]):
        print(f"    - {model}: ${costs['total_cost']:.2f} ({costs['total_tokens']:,} tokens)")

    # E. Full Dataset Projection
    print("\n[5/5] Projecting costs to full dataset...")
    full_size = FULL_DATASET_SIZES.get(species, len(df))
    projections = project_full_dataset_costs(
        api_costs,
        full_dataset_size=full_size,
        confidence_level=0.95,
    )
    results["projected_full_dataset"] = projections

    print(f"  ✓ Projecting from {len(df)} samples to ~{full_size:,} neurons")
    print("\n  Projected costs with 95% confidence intervals:")
    for model, proj in sorted(projections["models"].items(),
                             key=lambda x: x[1]["projected_cost"]):
        cost = proj["projected_cost"]
        ci_lower = proj["ci_lower"]
        ci_upper = proj["ci_upper"]
        print(f"    - {model}: ${cost:,.2f} (CI: ${ci_lower:,.2f} - ${ci_upper:,.2f})")

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")

    return results


# ============================================================================
# Visualization
# ============================================================================

def create_visualizations(df: pd.DataFrame, results: Dict[str, Any],
                         output_dir: Path) -> None:
    """
    Generate publication-quality visualizations.

    Args:
        df: Training data DataFrame
        results: Analysis results dictionary
        output_dir: Directory to save figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 5)
    plt.rcParams['font.size'] = 10

    species = results["species"]

    # 1. Edit Distribution Histogram
    print("\n  Creating edit distribution histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))

    edits_per_neuron = df.groupby('neuron_id').size()
    ax.hist(edits_per_neuron, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Edits per Neuron')
    ax.set_ylabel('Number of Neurons')
    ax.set_title(f'Edit Distribution: {species.upper()} Dataset')
    ax.axvline(edits_per_neuron.mean(), color='red', linestyle='--',
               label=f'Mean: {edits_per_neuron.mean():.1f}')
    ax.axvline(edits_per_neuron.median(), color='green', linestyle='--',
               label=f'Median: {edits_per_neuron.median():.1f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'edit_distribution_{species}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. API Cost Comparison
    print("  Creating API cost comparison chart...")
    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(results["api_costs_sample"]["per_model"].keys())
    costs = [results["api_costs_sample"]["per_model"][m]["total_cost"] for m in models]

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, costs, color=colors, edgecolor='black', alpha=0.8)

    ax.set_ylabel('Cost (USD)', fontsize=12)
    ax.set_title(f'API Cost Comparison (Sample): {species.upper()}', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f'api_costs_comparison_{species}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Projected Costs with Confidence Intervals
    print("  Creating projected costs visualization...")
    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(results["projected_full_dataset"]["models"].keys())
    projections = results["projected_full_dataset"]["models"]

    means = [projections[m]["projected_cost"] for m in models]
    ci_lower = [projections[m]["ci_lower"] for m in models]
    ci_upper = [projections[m]["ci_upper"] for m in models]

    errors = [
        [means[i] - ci_lower[i] for i in range(len(models))],
        [ci_upper[i] - means[i] for i in range(len(models))]
    ]

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, means, yerr=errors, capsize=10, color=colors,
                  edgecolor='black', alpha=0.8, error_kw={'elinewidth': 2})

    ax.set_ylabel('Projected Cost (USD)', fontsize=12)
    ax.set_title(f'Projected API Costs (Full Dataset) with 95% CI: {species.upper()}', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for i, (bar, cost) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:,.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f'projected_costs_{species}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Complexity Scatter (if available)
    if 'before_vertex_counts' in df.columns and 'correlation_edits_vs_complexity' in results["complexity_metrics"]:
        print("  Creating complexity analysis scatter plot...")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Group by neuron for scatter
        neuron_edits = df.groupby('neuron_id').size()
        neuron_vertices = df.groupby('neuron_id')['before_vertex_counts'].apply(
            lambda x: pd.to_numeric(x, errors='coerce').mean()
        )

        common_neurons = set(neuron_edits.index) & set(neuron_vertices.index)
        if len(common_neurons) > 1:  # Need at least 2 points for a trend
            edits_aligned = neuron_edits[list(common_neurons)]
            vertices_aligned = neuron_vertices[list(common_neurons)]

            ax.scatter(vertices_aligned, edits_aligned, alpha=0.5, s=30, color='steelblue', edgecolors='black')
            ax.set_xlabel('Average Neuron Size (Vertex Count)')
            ax.set_ylabel('Number of Edits')
            ax.set_title(f'Edit Complexity vs Neuron Size: {species.upper()}')

            # Add trend line only if we have enough non-constant data
            try:
                # Check if vertices have variance
                if len(set(vertices_aligned.values)) > 1 and not np.isnan(vertices_aligned).any():
                    z = np.polyfit(vertices_aligned, edits_aligned, 1)
                    p = np.poly1d(z)
                    ax.plot(vertices_aligned, p(vertices_aligned), "r--", alpha=0.8, label='Trend')
                    ax.legend()
            except Exception as e:
                # Skip trend line if fitting fails
                pass

            plt.tight_layout()
            plt.savefig(output_dir / f'complexity_analysis_{species}.png', dpi=150, bbox_inches='tight')
            plt.close()

    print(f"  ✓ Saved {len(list(output_dir.glob('*.png')))} figures to {output_dir}")


# ============================================================================
# Report Generation
# ============================================================================

def save_report(results: Dict[str, Any], output_file: Path) -> None:
    """
    Save analysis results to JSON report.

    Args:
        results: Analysis results dictionary
        output_file: Path to save report
    """
    # Clean numpy types for JSON serialization
    clean_results = clean_report_for_json(results)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(clean_results, f, indent=2)

    print(f"\n✓ Report saved to {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Compute cost analysis for ConnectomeBench benchmark"
    )
    parser.add_argument(
        '--species',
        choices=['fly', 'human', 'zebrafish', 'all'],
        default='fly',
        help='Species to analyze (default: fly)'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path.cwd(),
        help='Directory containing training data JSON files (default: current directory)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path.cwd() / 'reports' / 'compute_costs',
        help='Output directory for reports and figures'
    )

    args = parser.parse_args()

    # Determine which species to analyze
    species_list = ['fly', 'human', 'zebrafish'] if args.species == 'all' else [args.species]

    # Ensure output directories exist
    output_dir = args.output_dir
    figures_dir = output_dir / 'figures'

    for species in species_list:
        # Load data
        df = load_training_data(species, args.data_dir)
        if df is None or len(df) == 0:
            print(f"Skipping {species}: no data available")
            continue

        # Run analysis
        results = run_analysis(df, species)

        # Generate visualizations
        print(f"\nGenerating visualizations...")
        create_visualizations(df, results, figures_dir)

        # Save report
        report_file = output_dir / f'compute_costs_analysis_{species}.json'
        save_report(results, report_file)

    print(f"\n{'='*70}")
    print("All analyses complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
