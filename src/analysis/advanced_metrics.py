"""
Advanced metrics for TSSP model evaluation:
- Sensitivity Analysis
- Expected Value of Perfect Information (EVPI)
- Expected Mission Value (EMV)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from src.optimization import TSSPModel


def calculate_evpi(
    tssp_model: TSSPModel,
    behavior_classes: List[str],
    behavior_probabilities: Dict[Tuple[str, str], float],
    sources: List[str],
    tasks: List[str],
    stage1_costs: Dict[Tuple[str, str], float],
    recourse_costs: Dict[str, float],
    solver_name: str = 'glpk'
) -> Dict:
    """
    Calculate Expected Value of Perfect Information (EVPI).
    
    EVPI measures the maximum amount a decision maker would be willing to pay
    for perfect information about uncertain parameters (behavior probabilities).
    
    EVPI = E[V(perfect_info)] - E[V(current_info)]
    
    Where:
    - E[V(perfect_info)] = Expected value with perfect information (wait-and-see)
    - E[V(current_info)] = Expected value with current information (here-and-now)
    
    Parameters:
    -----------
    tssp_model : TSSPModel
        Solved TSSP model with current information (here-and-now solution)
    behavior_classes : List[str]
        List of behavior classes
    behavior_probabilities : Dict[Tuple[str, str], float]
        Behavior probabilities for each source
    sources : List[str]
        List of source IDs
    tasks : List[str]
        List of task IDs
    stage1_costs : Dict[Tuple[str, str], float]
        Stage 1 costs
    recourse_costs : Dict[str, float]
        Recourse costs by behavior class
    solver_name : str
        Solver to use
    
    Returns:
    --------
    Dict with EVPI metrics
    """
    # Current solution value (here-and-now)
    current_value = tssp_model.solution.get('objective_value', 0.0)
    
    # Calculate wait-and-see value (perfect information)
    # For each scenario (behavior realization), solve optimally
    # Note: This can be computationally expensive. We sample scenarios or use approximation.
    wait_and_see_values = []
    scenario_details = []
    
    # Sample scenarios: for each behavior class, find the most likely source
    # This reduces computational complexity while maintaining accuracy
    sampled_scenarios = []
    for behavior in behavior_classes:
        # Find source with highest probability for this behavior
        max_prob = 0.0
        best_source = None
        for source in sources:
            prob = behavior_probabilities.get((source, behavior), 0.0)
            if prob > max_prob:
                max_prob = prob
                best_source = source
        
        if best_source and max_prob > 1e-6:
            sampled_scenarios.append((best_source, behavior, max_prob))
    
    # If we have many sources, limit to top scenarios by probability
    if len(sampled_scenarios) > 20:
        sampled_scenarios.sort(key=lambda x: x[2], reverse=True)
        sampled_scenarios = sampled_scenarios[:20]
    
    print(f"  Calculating wait-and-see value for {len(sampled_scenarios)} scenarios...")
    
    for source, behavior, prob in sampled_scenarios:
        # Create deterministic scenario: this source exhibits this behavior with probability 1
        # Other sources keep their original probabilities (simplified approach)
        deterministic_probs = behavior_probabilities.copy()
        # Set this source's behavior to deterministic
        for b in behavior_classes:
            if b == behavior:
                deterministic_probs[(source, b)] = 1.0
            else:
                deterministic_probs[(source, b)] = 0.0
        
        # Solve with perfect information (deterministic behavior for this source)
        try:
            perfect_info_model = TSSPModel(
                sources=sources,
                tasks=tasks,
                behavior_classes=behavior_classes,
                behavior_probabilities=deterministic_probs,
                stage1_costs=stage1_costs,
                recourse_costs=recourse_costs
            )
            perfect_info_model.build_model()
            success = perfect_info_model.solve(solver_name=solver_name, verbose=False)
            
            if success:
                ws_value = perfect_info_model.solution.get('objective_value', 0.0)
                wait_and_see_values.append(prob * ws_value)
                scenario_details.append({
                    'source': source,
                    'behavior': behavior,
                    'probability': prob,
                    'value': ws_value,
                    'weighted_value': prob * ws_value
                })
        except Exception as e:
            print(f"    Warning: Could not solve perfect info scenario for {source}-{behavior}: {e}")
    
    # Expected value with perfect information (wait-and-see)
    # If we sampled, we approximate by scaling
    wait_and_see_value = sum(wait_and_see_values)
    
    # Scale to account for sampled scenarios (approximation)
    if len(sampled_scenarios) < len(sources) * len(behavior_classes):
        total_sampled_prob = sum(prob for _, _, prob in sampled_scenarios)
        if total_sampled_prob > 0:
            # Scale factor to approximate full scenario space
            scale_factor = min(1.0, total_sampled_prob)
            wait_and_see_value = wait_and_see_value / scale_factor if scale_factor > 0 else wait_and_see_value
    
    # EVPI = Wait-and-see value - Here-and-now value
    # Note: For minimization, EVPI = current_value - wait_and_see_value
    # (we save more with perfect information)
    evpi = current_value - wait_and_see_value
    
    return {
        'current_value': current_value,
        'wait_and_see_value': wait_and_see_value,
        'evpi': evpi,
        'evpi_percentage': (evpi / current_value * 100) if current_value > 0 else 0.0,
        'scenario_details': scenario_details,
        'interpretation': (
            f"EVPI = {evpi:.2f} means we would save up to {evpi:.2f} units "
            f"({evpi/current_value*100:.1f}%) with perfect information about behaviors."
        ) if current_value > 0 else "EVPI calculation completed."
    }


def calculate_emv(
    tssp_model: TSSPModel,
    information_values: Optional[Dict[Tuple[str, str], float]] = None
) -> Dict:
    """
    Calculate Expected Mission Value (EMV).
    
    EMV represents the expected value of the mission/objective, considering:
    - Costs (Stage 1 + Stage 2)
    - Information values (if provided)
    
    EMV = -Total_Cost + Information_Value (for maximization)
    or
    EMV = Information_Value - Total_Cost (net value)
    
    Parameters:
    -----------
    tssp_model : TSSPModel
        Solved TSSP model
    information_values : Optional[Dict[Tuple[str, str], float]]
        Information values for source-task assignments
    
    Returns:
    --------
    Dict with EMV metrics
    """
    if tssp_model.solution is None:
        raise ValueError("Model must be solved before calculating EMV")
    
    # Total expected cost
    total_cost = tssp_model.get_total_cost()
    stage1_cost = tssp_model.get_stage1_cost()
    stage2_cost = tssp_model.get_stage2_expected_cost()
    
    # Calculate information value from assignments
    information_value = 0.0
    if information_values:
        for (s, t), assigned in tssp_model.solution.get('assignments', {}).items():
            if assigned:
                info_val = information_values.get((s, t), 0.0)
                information_value += info_val
    
    # EMV = Information Value - Total Cost (net mission value)
    emv = information_value - total_cost
    
    # Alternative: EMV as negative cost (for minimization perspective)
    emv_negative = -total_cost
    
    return {
        'total_cost': total_cost,
        'stage1_cost': stage1_cost,
        'stage2_cost': stage2_cost,
        'information_value': information_value,
        'emv': emv,
        'emv_negative': emv_negative,
        'emv_per_source': emv / len(tssp_model.solution.get('assignments', {})) if tssp_model.solution.get('assignments') else 0.0,
        'interpretation': (
            f"EMV = {emv:.2f} represents the net mission value "
            f"(Information Value: {information_value:.2f} - Total Cost: {total_cost:.2f})."
        )
    }


def sensitivity_analysis(
    tssp_model: TSSPModel,
    behavior_classes: List[str],
    behavior_probabilities: Dict[Tuple[str, str], float],
    sources: List[str],
    tasks: List[str],
    stage1_costs: Dict[Tuple[str, str], float],
    recourse_costs: Dict[str, float],
    variation_range: float = 0.2,
    num_points: int = 11,
    solver_name: str = 'glpk',
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Perform sensitivity analysis on key parameters.
    
    Analyzes how changes in:
    1. Behavior probabilities
    2. Recourse costs
    3. Stage 1 costs
    
    affect the optimal solution and objective value.
    
    Parameters:
    -----------
    tssp_model : TSSPModel
        Baseline solved TSSP model
    behavior_classes : List[str]
        List of behavior classes
    behavior_probabilities : Dict[Tuple[str, str], float]
        Baseline behavior probabilities
    sources : List[str]
        List of source IDs
    tasks : List[str]
        List of task IDs
    stage1_costs : Dict[Tuple[str, str], float]
        Baseline Stage 1 costs
    recourse_costs : Dict[str, float]
        Baseline recourse costs
    variation_range : float
        Range of variation (e.g., 0.2 = ±20%)
    num_points : int
        Number of points to test
    solver_name : str
        Solver to use
    output_dir : Optional[Path]
        Directory to save sensitivity plots
    
    Returns:
    --------
    Dict with sensitivity analysis results
    """
    baseline_value = tssp_model.solution.get('objective_value', 0.0)
    
    results = {
        'baseline_value': baseline_value,
        'recourse_cost_sensitivity': {},
        'behavior_prob_sensitivity': {},
        'stage1_cost_sensitivity': {}
    }
    
    # 1. Sensitivity to recourse costs
    print("Performing sensitivity analysis on recourse costs...")
    recourse_sensitivity = {}
    for behavior in behavior_classes:
        if behavior == 'cooperative':  # Skip cooperative (always 0)
            continue
        
        base_cost = recourse_costs.get(behavior, 0.0)
        if base_cost == 0:
            continue
        
        variations = np.linspace(
            base_cost * (1 - variation_range),
            base_cost * (1 + variation_range),
            num_points
        )
        objective_values = []
        
        for var_cost in variations:
            modified_recourse = recourse_costs.copy()
            modified_recourse[behavior] = var_cost
            
            try:
                test_model = TSSPModel(
                    sources=sources,
                    tasks=tasks,
                    behavior_classes=behavior_classes,
                    behavior_probabilities=behavior_probabilities,
                    stage1_costs=stage1_costs,
                    recourse_costs=modified_recourse
                )
                test_model.build_model()
                success = test_model.solve(solver_name=solver_name, verbose=False)
                
                if success:
                    obj_val = test_model.solution.get('objective_value', 0.0)
                    objective_values.append(obj_val)
                else:
                    objective_values.append(np.nan)
            except Exception as e:
                objective_values.append(np.nan)
        
        recourse_sensitivity[behavior] = {
            'variations': variations.tolist(),
            'objective_values': objective_values,
            'percent_change': [(v - baseline_value) / baseline_value * 100 if baseline_value > 0 else 0 
                              for v in objective_values]
        }
    
    results['recourse_cost_sensitivity'] = recourse_sensitivity
    
    # 2. Sensitivity to behavior probabilities (for a sample source)
    print("Performing sensitivity analysis on behavior probabilities...")
    if sources:
        sample_source = sources[0]
        prob_sensitivity = {}
        
        for behavior in behavior_classes:
            base_prob = behavior_probabilities.get((sample_source, behavior), 0.0)
            if base_prob < 1e-6:
                continue
            
            variations = np.linspace(
                max(0.0, base_prob - variation_range),
                min(1.0, base_prob + variation_range),
                num_points
            )
            objective_values = []
            
            for var_prob in variations:
                # Adjust probabilities to maintain sum = 1
                modified_probs = behavior_probabilities.copy()
                prob_diff = var_prob - base_prob
                
                # Redistribute difference proportionally among other behaviors
                other_behaviors = [b for b in behavior_classes if b != behavior]
                other_probs_sum = sum(
                    behavior_probabilities.get((sample_source, b), 0.0) 
                    for b in other_behaviors
                )
                
                if other_probs_sum > 1e-6:
                    for other_b in other_behaviors:
                        other_base = behavior_probabilities.get((sample_source, other_b), 0.0)
                        modified_probs[(sample_source, other_b)] = max(0.0, other_base - prob_diff * (other_base / other_probs_sum))
                
                modified_probs[(sample_source, behavior)] = var_prob
                
                # Normalize to ensure sum = 1
                total = sum(modified_probs.get((sample_source, b), 0.0) for b in behavior_classes)
                if total > 1e-6:
                    for b in behavior_classes:
                        modified_probs[(sample_source, b)] = modified_probs.get((sample_source, b), 0.0) / total
                
                try:
                    test_model = TSSPModel(
                        sources=sources,
                        tasks=tasks,
                        behavior_classes=behavior_classes,
                        behavior_probabilities=modified_probs,
                        stage1_costs=stage1_costs,
                        recourse_costs=recourse_costs
                    )
                    test_model.build_model()
                    success = test_model.solve(solver_name=solver_name, verbose=False)
                    
                    if success:
                        obj_val = test_model.solution.get('objective_value', 0.0)
                        objective_values.append(obj_val)
                    else:
                        objective_values.append(np.nan)
                except Exception as e:
                    objective_values.append(np.nan)
            
            prob_sensitivity[behavior] = {
                'variations': variations.tolist(),
                'objective_values': objective_values,
                'percent_change': [(v - baseline_value) / baseline_value * 100 if baseline_value > 0 else 0 
                                  for v in objective_values]
            }
        
        results['behavior_prob_sensitivity'] = prob_sensitivity
    
    # 3. Sensitivity to Stage 1 costs (for a sample source-task pair)
    print("Performing sensitivity analysis on Stage 1 costs...")
    if sources and tasks:
        sample_source = sources[0]
        sample_task = tasks[0]
        base_stage1 = stage1_costs.get((sample_source, sample_task), 0.0)
        
        if base_stage1 > 0:
            variations = np.linspace(
                base_stage1 * (1 - variation_range),
                base_stage1 * (1 + variation_range),
                num_points
            )
            objective_values = []
            
            for var_cost in variations:
                modified_stage1 = stage1_costs.copy()
                modified_stage1[(sample_source, sample_task)] = var_cost
                
                try:
                    test_model = TSSPModel(
                        sources=sources,
                        tasks=tasks,
                        behavior_classes=behavior_classes,
                        behavior_probabilities=behavior_probabilities,
                        stage1_costs=modified_stage1,
                        recourse_costs=recourse_costs
                    )
                    test_model.build_model()
                    success = test_model.solve(solver_name=solver_name, verbose=False)
                    
                    if success:
                        obj_val = test_model.solution.get('objective_value', 0.0)
                        objective_values.append(obj_val)
                    else:
                        objective_values.append(np.nan)
                except Exception as e:
                    objective_values.append(np.nan)
            
            results['stage1_cost_sensitivity'] = {
                'variations': variations.tolist(),
                'objective_values': objective_values,
                'percent_change': [(v - baseline_value) / baseline_value * 100 if baseline_value > 0 else 0 
                                  for v in objective_values]
            }
    
    # Create visualizations
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_sensitivity_analysis(results, output_dir)
    
    return results


def plot_sensitivity_analysis(
    sensitivity_results: Dict,
    output_dir: Path
):
    """
    Plot sensitivity analysis results.
    
    Parameters:
    -----------
    sensitivity_results : Dict
        Results from sensitivity_analysis()
    output_dir : Path
        Directory to save plots
    """
    # Plot recourse cost sensitivity
    if sensitivity_results.get('recourse_cost_sensitivity'):
        fig, axes = plt.subplots(1, len(sensitivity_results['recourse_cost_sensitivity']), 
                                figsize=(5 * len(sensitivity_results['recourse_cost_sensitivity']), 5))
        if len(sensitivity_results['recourse_cost_sensitivity']) == 1:
            axes = [axes]
        
        for idx, (behavior, data) in enumerate(sensitivity_results['recourse_cost_sensitivity'].items()):
            ax = axes[idx] if len(axes) > 1 else axes[0]
            variations = data['variations']
            obj_vals = data['objective_values']
            
            # Filter out NaN values
            valid_indices = [i for i, v in enumerate(obj_vals) if not np.isnan(v)]
            valid_vars = [variations[i] for i in valid_indices]
            valid_vals = [obj_vals[i] for i in valid_indices]
            
            ax.plot(valid_vars, valid_vals, 'o-', linewidth=2, markersize=6)
            ax.axhline(y=sensitivity_results['baseline_value'], color='r', 
                      linestyle='--', label='Baseline', linewidth=1.5)
            ax.set_xlabel(f'Recourse Cost for {behavior}')
            ax.set_ylabel('Objective Value')
            ax.set_title(f'Sensitivity to {behavior} Recourse Cost')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sensitivity_recourse_costs.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved sensitivity plot to: {output_dir / 'sensitivity_recourse_costs.png'}")
    
    # Plot behavior probability sensitivity
    if sensitivity_results.get('behavior_prob_sensitivity'):
        fig, axes = plt.subplots(1, len(sensitivity_results['behavior_prob_sensitivity']), 
                                figsize=(5 * len(sensitivity_results['behavior_prob_sensitivity']), 5))
        if len(sensitivity_results['behavior_prob_sensitivity']) == 1:
            axes = [axes]
        
        for idx, (behavior, data) in enumerate(sensitivity_results['behavior_prob_sensitivity'].items()):
            ax = axes[idx] if len(axes) > 1 else axes[0]
            variations = data['variations']
            obj_vals = data['objective_values']
            
            valid_indices = [i for i, v in enumerate(obj_vals) if not np.isnan(v)]
            valid_vars = [variations[i] for i in valid_indices]
            valid_vals = [obj_vals[i] for i in valid_indices]
            
            ax.plot(valid_vars, valid_vals, 's-', linewidth=2, markersize=6, color='green')
            ax.axhline(y=sensitivity_results['baseline_value'], color='r', 
                      linestyle='--', label='Baseline', linewidth=1.5)
            ax.set_xlabel(f'Probability of {behavior}')
            ax.set_ylabel('Objective Value')
            ax.set_title(f'Sensitivity to {behavior} Probability')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sensitivity_behavior_probs.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved sensitivity plot to: {output_dir / 'sensitivity_behavior_probs.png'}")


def generate_advanced_metrics_report(
    evpi_results: Dict,
    emv_results: Dict,
    sensitivity_results: Dict,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate a comprehensive report of advanced metrics.
    
    Parameters:
    -----------
    evpi_results : Dict
        Results from calculate_evpi()
    emv_results : Dict
        Results from calculate_emv()
    sensitivity_results : Dict
        Results from sensitivity_analysis()
    output_path : Optional[Path]
        Path to save report
    
    Returns:
    --------
    str
        Report text
    """
    report = []
    report.append("=" * 60)
    report.append("ADVANCED METRICS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # EVPI Section
    report.append("EXPECTED VALUE OF PERFECT INFORMATION (EVPI):")
    report.append(f"  Current Value (Here-and-Now): {evpi_results.get('current_value', 0):.2f}")
    report.append(f"  Wait-and-See Value (Perfect Info): {evpi_results.get('wait_and_see_value', 0):.2f}")
    report.append(f"  EVPI: {evpi_results.get('evpi', 0):.2f}")
    report.append(f"  EVPI Percentage: {evpi_results.get('evpi_percentage', 0):.2f}%")
    report.append(f"  Interpretation: {evpi_results.get('interpretation', 'N/A')}")
    report.append("")
    
    # EMV Section
    report.append("EXPECTED MISSION VALUE (EMV):")
    report.append(f"  Total Cost: {emv_results.get('total_cost', 0):.2f}")
    report.append(f"  Stage 1 Cost: {emv_results.get('stage1_cost', 0):.2f}")
    report.append(f"  Stage 2 Cost: {emv_results.get('stage2_cost', 0):.2f}")
    report.append(f"  Information Value: {emv_results.get('information_value', 0):.2f}")
    report.append(f"  EMV (Net Mission Value): {emv_results.get('emv', 0):.2f}")
    report.append(f"  EMV per Source: {emv_results.get('emv_per_source', 0):.2f}")
    report.append(f"  Interpretation: {emv_results.get('interpretation', 'N/A')}")
    report.append("")
    
    # Sensitivity Analysis Section
    report.append("SENSITIVITY ANALYSIS:")
    report.append(f"  Baseline Objective Value: {sensitivity_results.get('baseline_value', 0):.2f}")
    report.append("")
    
    if sensitivity_results.get('recourse_cost_sensitivity'):
        report.append("  Recourse Cost Sensitivity:")
        for behavior, data in sensitivity_results['recourse_cost_sensitivity'].items():
            obj_vals = [v for v in data['objective_values'] if not np.isnan(v)]
            if obj_vals:
                min_val = min(obj_vals)
                max_val = max(obj_vals)
                range_val = max_val - min_val
                report.append(f"    {behavior:15s}: Range = {range_val:.2f} "
                            f"(Min: {min_val:.2f}, Max: {max_val:.2f})")
        report.append("")
    
    if sensitivity_results.get('behavior_prob_sensitivity'):
        report.append("  Behavior Probability Sensitivity:")
        for behavior, data in sensitivity_results['behavior_prob_sensitivity'].items():
            obj_vals = [v for v in data['objective_values'] if not np.isnan(v)]
            if obj_vals:
                min_val = min(obj_vals)
                max_val = max(obj_vals)
                range_val = max_val - min_val
                report.append(f"    {behavior:15s}: Range = {range_val:.2f} "
                            f"(Min: {min_val:.2f}, Max: {max_val:.2f})")
        report.append("")
    
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Advanced metrics report saved to: {output_path}")
    
    return report_text


def calculate_efficiency_frontier(
    sources: List[str],
    tasks: List[str],
    behavior_classes: List[str],
    behavior_probabilities: Dict[Tuple[str, str], float],
    stage1_costs: Dict[Tuple[str, str], float],
    recourse_costs: Dict[str, float],
    source_capacities: Optional[Dict[str, int]] = None,
    task_requirements: Optional[Dict[str, int]] = None,
    n_scenarios: int = 20,
    solver_name: str = 'glpk'
) -> Dict:
    """
    Calculate the efficiency frontier for Stage 1 vs Stage 2 cost trade-off.
    
    The efficiency frontier shows Pareto-optimal allocations where you cannot
    reduce Stage 1 cost without increasing Stage 2 cost, and vice versa.
    
    Parameters:
    -----------
    sources : List[str]
        List of source IDs
    tasks : List[str]
        List of task IDs
    behavior_classes : List[str]
        List of behavior classes
    behavior_probabilities : Dict[Tuple[str, str], float]
        Behavior probabilities P(b|s) for each source-behavior pair
    stage1_costs : Dict[Tuple[str, str], float]
        Stage 1 costs c(s,t) for each source-task pair
    recourse_costs : Dict[str, float]
        Recourse costs q(b) for each behavior class
    source_capacities : Optional[Dict[str, int]]
        Maximum number of tasks each source can handle (default: unlimited)
    task_requirements : Optional[Dict[str, int]]
        Minimum number of sources required for each task (default: 1)
    n_scenarios : int
        Number of allocation scenarios to generate (default: 20)
    solver_name : str
        Solver to use for optimization (default: 'glpk')
    
    Returns:
    --------
    Dict with:
        - 'frontier_points': List of (stage1_cost, stage2_cost) tuples on the frontier
        - 'dominated_points': List of (stage1_cost, stage2_cost) tuples dominated by frontier
        - 'all_points': List of all (stage1_cost, stage2_cost, scenario_name) tuples
        - 'frontier_scenarios': List of scenario names on the frontier
    """
    print(f"Calculating efficiency frontier with {n_scenarios} allocation scenarios...")
    
    # Default capacities and requirements
    if source_capacities is None:
        source_capacities = {s: len(tasks) for s in sources}  # Unlimited capacity
    if task_requirements is None:
        task_requirements = {t: 1 for t in tasks}  # At least 1 source per task
    
    all_points = []
    frontier_points = []
    dominated_points = []
    frontier_scenarios = []
    
    # Generate multiple allocation scenarios
    scenarios = []
    
    # Scenario 1: Optimal TSSP solution (baseline)
    try:
        baseline_model = TSSPModel(
            sources=sources,
            tasks=tasks,
            behavior_classes=behavior_classes,
            behavior_probabilities=behavior_probabilities,
            stage1_costs=stage1_costs,
            recourse_costs=recourse_costs
        )
        baseline_model.build_model()
        if baseline_model.solve(solver_name=solver_name, verbose=False):
            stage1 = baseline_model.get_stage1_cost()
            stage2 = baseline_model.get_stage2_expected_cost()
            scenarios.append({
                'name': 'Optimal TSSP',
                'stage1_cost': stage1,
                'stage2_cost': stage2,
                'model': baseline_model
            })
    except Exception as e:
        print(f"Warning: Could not solve baseline TSSP model: {e}")
    
    # Scenario 2-N: Different allocation strategies
    # Strategy: Risk-averse (prioritize high-reliability sources)
    try:
        # Sort sources by reliability (inverse of Stage 1 cost)
        source_avg_costs = {}
        for s in sources:
            costs = [stage1_costs.get((s, t), 0.0) for t in tasks]
            source_avg_costs[s] = np.mean(costs) if costs else 10.0
        
        sorted_sources = sorted(sources, key=lambda s: source_avg_costs[s])
        
        # Create risk-averse allocation (use low-cost sources first)
        risk_averse_assignments = {}
        task_assignments = {t: [] for t in tasks}
        
        for s in sorted_sources[:len(sources)]:
            capacity = source_capacities.get(s, len(tasks))
            assigned = 0
            for t in tasks:
                if assigned < capacity and len(task_assignments[t]) < task_requirements.get(t, 1):
                    risk_averse_assignments[(s, t)] = 1
                    task_assignments[t].append(s)
                    assigned += 1
        
        # Calculate costs for this allocation
        stage1_risk_averse = sum(
            stage1_costs.get((s, t), 0.0) * risk_averse_assignments.get((s, t), 0)
            for s in sources for t in tasks
        )
        
        # Estimate Stage 2 cost (simplified: assume recourse proportional to non-cooperative probability)
        stage2_risk_averse = 0.0
        for s in sources:
            for t in tasks:
                if risk_averse_assignments.get((s, t), 0) > 0:
                    for b in behavior_classes:
                        prob = behavior_probabilities.get((s, b), 0.0)
                        cost = recourse_costs.get(b, 0.0)
                        # Simplified: assume recourse action = 0.5 if non-cooperative
                        if cost > 0:
                            stage2_risk_averse += prob * cost * 0.5
        
        scenarios.append({
            'name': 'Risk-Averse (High Reliability)',
            'stage1_cost': stage1_risk_averse,
            'stage2_cost': stage2_risk_averse,
            'model': None
        })
    except Exception as e:
        print(f"Warning: Could not create risk-averse scenario: {e}")
    
    # Strategy: Risk-seeking (use low-reliability sources, lower Stage 1 cost)
    try:
        sorted_sources_high_cost = sorted(sources, key=lambda s: source_avg_costs[s], reverse=True)
        
        risk_seeking_assignments = {}
        task_assignments = {t: [] for t in tasks}
        
        for s in sorted_sources_high_cost[:len(sources)]:
            capacity = source_capacities.get(s, len(tasks))
            assigned = 0
            for t in tasks:
                if assigned < capacity and len(task_assignments[t]) < task_requirements.get(t, 1):
                    risk_seeking_assignments[(s, t)] = 1
                    task_assignments[t].append(s)
                    assigned += 1
        
        stage1_risk_seeking = sum(
            stage1_costs.get((s, t), 0.0) * risk_seeking_assignments.get((s, t), 0)
            for s in sources for t in tasks
        )
        
        stage2_risk_seeking = 0.0
        for s in sources:
            for t in tasks:
                if risk_seeking_assignments.get((s, t), 0) > 0:
                    for b in behavior_classes:
                        prob = behavior_probabilities.get((s, b), 0.0)
                        cost = recourse_costs.get(b, 0.0)
                        if cost > 0:
                            stage2_risk_seeking += prob * cost * 0.7  # Higher recourse for risk-seeking
        
        scenarios.append({
            'name': 'Risk-Seeking (Low Reliability)',
            'stage1_cost': stage1_risk_seeking,
            'stage2_cost': stage2_risk_seeking,
            'model': None
        })
    except Exception as e:
        print(f"Warning: Could not create risk-seeking scenario: {e}")
    
    # Generate additional random/interpolated scenarios
    if len(scenarios) >= 2:
        # Interpolate between scenarios
        for i in range(n_scenarios - len(scenarios)):
            alpha = i / (n_scenarios - len(scenarios) + 1)
            
            # Interpolate between risk-averse and risk-seeking
            if len(scenarios) >= 2:
                s1 = scenarios[1] if len(scenarios) > 1 else scenarios[0]
                s2 = scenarios[2] if len(scenarios) > 2 else scenarios[0]
                
                interp_stage1 = s1['stage1_cost'] * (1 - alpha) + s2['stage1_cost'] * alpha
                interp_stage2 = s1['stage2_cost'] * (1 - alpha) + s2['stage2_cost'] * alpha
                
                scenarios.append({
                    'name': f'Interpolated {i+1}',
                    'stage1_cost': interp_stage1,
                    'stage2_cost': interp_stage2,
                    'model': None
                })
    
    # Collect all points
    for scenario in scenarios:
        all_points.append((
            scenario['stage1_cost'],
            scenario['stage2_cost'],
            scenario['name']
        ))
    
    # Find Pareto-optimal frontier
    # A point is on the frontier if no other point has both lower Stage 1 AND lower Stage 2 cost
    for i, (s1_i, s2_i, name_i) in enumerate(all_points):
        is_dominated = False
        for j, (s1_j, s2_j, name_j) in enumerate(all_points):
            if i != j:
                # Point j dominates point i if both costs are lower
                if s1_j < s1_i and s2_j < s2_i:
                    is_dominated = True
                    break
        
        if not is_dominated:
            frontier_points.append((s1_i, s2_i))
            frontier_scenarios.append(name_i)
        else:
            dominated_points.append((s1_i, s2_i))
    
    # Sort frontier points by Stage 1 cost for visualization
    frontier_points = sorted(frontier_points, key=lambda x: x[0])
    
    print(f"Efficiency frontier calculated: {len(frontier_points)} frontier points, "
          f"{len(dominated_points)} dominated points")
    
    return {
        'frontier_points': frontier_points,
        'dominated_points': dominated_points,
        'all_points': all_points,
        'frontier_scenarios': frontier_scenarios,
        'scenarios': scenarios
    }


def plot_efficiency_frontier(
    frontier_data: Dict,
    output_path: Optional[Path] = None,
    show_dominated: bool = True
) -> plt.Figure:
    """
    Plot the efficiency frontier.
    
    Parameters:
    -----------
    frontier_data : Dict
        Results from calculate_efficiency_frontier
    output_path : Optional[Path]
        Path to save the plot
    show_dominated : bool
        Whether to show dominated points (default: True)
    
    Returns:
    --------
    matplotlib Figure
    """
    frontier_points = frontier_data['frontier_points']
    dominated_points = frontier_data['dominated_points']
    all_points = frontier_data['all_points']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot dominated points (if requested)
    if show_dominated and dominated_points:
        dom_s1, dom_s2 = zip(*dominated_points)
        ax.scatter(dom_s1, dom_s2, c='lightgray', alpha=0.5, s=50,
                  label='Dominated Points', marker='o', edgecolors='gray', linewidths=0.5)
    
    # Plot frontier points
    if frontier_points:
        front_s1, front_s2 = zip(*frontier_points)
        ax.scatter(front_s1, front_s2, c='#3b82f6', s=100, marker='o',
                  label='Efficiency Frontier', edgecolors='#1e40af', linewidths=2, zorder=5)
        
        # Connect frontier points with line
        ax.plot(front_s1, front_s2, 'b-', linewidth=2, alpha=0.6, zorder=4)
    
    # Highlight optimal TSSP point if available
    for s1, s2, name in all_points:
        if 'Optimal TSSP' in name:
            ax.scatter(s1, s2, c='#10b981', s=200, marker='*',
                      label='Optimal TSSP Solution', edgecolors='#047857', linewidths=2, zorder=6)
            break
    
    ax.set_xlabel('Stage 1 Cost (Strategic Tasking)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Stage 2 Cost (Recourse)', fontsize=12, fontweight='bold')
    ax.set_title('Efficiency Frontier: Stage 1 vs Stage 2 Cost Trade-off', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # Add annotation explaining the frontier
    ax.text(0.02, 0.98, 
            'Points on the frontier are Pareto-optimal:\n'
            'Cannot reduce Stage 1 cost without increasing Stage 2 cost,\n'
            'and vice versa.',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Efficiency frontier plot saved to: {output_path}")
    
    return fig
