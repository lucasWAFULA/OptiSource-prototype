"""
Cost analysis and decomposition for TSSP optimization results.

This module provides functions to analyze and visualize cost breakdowns
from the TSSP optimization model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


def analyze_costs(
    tssp_model,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Perform comprehensive cost analysis on TSSP model results.
    
    Parameters:
    -----------
    tssp_model : TSSPModel
        Solved TSSP model instance
    output_dir : Optional[Path]
        Directory to save visualizations
    
    Returns:
    --------
    Dictionary with analysis results
    """
    if tssp_model.solution is None:
        raise ValueError("Model must be solved before cost analysis")
    
    decomposition = tssp_model.get_cost_decomposition()
    
    # Verify cost calculation
    calculated_total = decomposition['stage1_cost'] + decomposition['stage2_expected_cost']
    optimal_value = tssp_model.solution.get('objective_value', 0.0)
    
    verification = {
        'optimal_objective_value': optimal_value,
        'calculated_total_cost': calculated_total,
        'difference': abs(optimal_value - calculated_total),
        'verified': abs(optimal_value - calculated_total) < 1e-6
    }
    
    # Create visualizations
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_cost_by_behavior(decomposition['by_behavior'], output_dir)
        plot_cost_by_source(decomposition['by_source'], output_dir)
        plot_cost_pie_chart(decomposition, output_dir)
    
    return {
        'decomposition': decomposition,
        'verification': verification
    }


def plot_cost_by_behavior(
    behavior_costs: Dict[str, float],
    output_dir: Path
):
    """
    Plot Stage 2 expected recourse cost by behavior class.
    
    Parameters:
    -----------
    behavior_costs : Dict[str, float]
        Cost breakdown by behavior class
    output_dir : Path
        Directory to save plot
    """
    if not behavior_costs:
        return
    
    behavior_classes = list(behavior_costs.keys())
    costs = list(behavior_costs.values())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=behavior_classes, y=costs, palette='viridis')
    plt.title('Stage 2 Expected Recourse Cost by Behavior Class')
    plt.xlabel('Behavior Class')
    plt.ylabel('Expected Recourse Cost Contribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_path = output_dir / 'cost_by_behavior.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_path}")


def plot_cost_by_source(
    source_costs: Dict[str, float],
    output_dir: Path,
    top_n: int = 10
):
    """
    Plot Stage 2 expected recourse cost by source (top N).
    
    Parameters:
    -----------
    source_costs : Dict[str, float]
        Cost breakdown by source
    output_dir : Path
        Directory to save plot
    top_n : int
        Number of top sources to display
    """
    if not source_costs:
        return
    
    # Sort by cost and get top N
    sorted_sources = sorted(source_costs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    sources = [s[0] for s in sorted_sources]
    costs = [s[1] for s in sorted_sources]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=sources, y=costs, palette='viridis')
    plt.title(f'Stage 2 Expected Recourse Cost by Source (Top {top_n})')
    plt.xlabel('Source ID')
    plt.ylabel('Expected Recourse Cost Contribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = output_dir / 'cost_by_source.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_path}")


def plot_cost_pie_chart(
    decomposition: Dict,
    output_dir: Path
):
    """
    Plot pie chart showing Stage 1 vs Stage 2 cost proportions.
    
    Parameters:
    -----------
    decomposition : Dict
        Cost decomposition dictionary
    output_dir : Path
        Directory to save plot
    """
    stage1 = decomposition['stage1_cost']
    stage2 = decomposition['stage2_expected_cost']
    
    if stage1 + stage2 == 0:
        return
    
    plt.figure(figsize=(8, 8))
    plt.pie(
        [stage1, stage2],
        labels=['Stage 1 (Strategic Tasking)', 'Stage 2 (Expected Recourse)'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#2ecc71', '#e74c3c']
    )
    plt.title('Cost Distribution: Stage 1 vs Stage 2')
    plt.tight_layout()
    
    output_path = output_dir / 'cost_pie_chart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_path}")


def generate_cost_report(
    decomposition: Dict,
    verification: Dict,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate a text report of cost analysis.
    
    Parameters:
    -----------
    decomposition : Dict
        Cost decomposition dictionary
    verification : Dict
        Verification results
    output_path : Optional[Path]
        Path to save report
    
    Returns:
    --------
    str
        Report text
    """
    report = []
    report.append("=" * 60)
    report.append("TSSP COST ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Verification
    report.append("COST VERIFICATION:")
    report.append(f"  Optimal Objective Value: {verification['optimal_objective_value']:.2f}")
    report.append(f"  Calculated Total Cost: {verification['calculated_total_cost']:.2f}")
    report.append(f"  Difference: {verification['difference']:.6f}")
    report.append(f"  Verified: {'✓' if verification['verified'] else '✗'}")
    report.append("")
    
    # Cost breakdown
    report.append("COST DECOMPOSITION:")
    report.append(f"  Stage 1 Cost (Strategic Tasking): {decomposition['stage1_cost']:.2f}")
    report.append(f"  Stage 2 Expected Recourse Cost: {decomposition['stage2_expected_cost']:.2f}")
    report.append(f"  Total Expected Cost: {decomposition['total_cost']:.2f}")
    report.append(f"  Stage 2 Proportion: {decomposition['stage2_proportion']:.2%}")
    report.append("")
    
    # By behavior
    report.append("STAGE 2 COST BY BEHAVIOR CLASS:")
    for behavior, cost in sorted(
        decomposition['by_behavior'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        proportion = cost / decomposition['stage2_expected_cost'] if decomposition['stage2_expected_cost'] > 0 else 0
        report.append(f"  {behavior:15s}: {cost:10.2f} ({proportion:5.1%})")
    report.append("")
    
    # Top sources
    report.append("TOP 10 SOURCES BY STAGE 2 COST:")
    sorted_sources = sorted(
        decomposition['by_source'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for source, cost in sorted_sources:
        report.append(f"  {source:15s}: {cost:10.2f}")
    report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Report saved to: {output_path}")
    
    return report_text
