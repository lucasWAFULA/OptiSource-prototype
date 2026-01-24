"""Analysis and visualization modules."""

from .cost_analysis import (
    analyze_costs,
    plot_cost_by_behavior,
    plot_cost_by_source,
    plot_cost_pie_chart,
    generate_cost_report
)

from .advanced_metrics import (
    calculate_evpi,
    calculate_emv,
    sensitivity_analysis,
    plot_sensitivity_analysis,
    generate_advanced_metrics_report,
    calculate_efficiency_frontier,
    plot_efficiency_frontier
)

from .allocation_benchmarks import (
    alloc_deterministic_greedy,
    alloc_uniform,
    evaluate_allocation_efficiency,
)

__all__ = [
    'analyze_costs',
    'plot_cost_by_behavior',
    'plot_cost_by_source',
    'plot_cost_pie_chart',
    'generate_cost_report',
    'calculate_evpi',
    'calculate_emv',
    'sensitivity_analysis',
    'plot_sensitivity_analysis',
    'generate_advanced_metrics_report',
    'calculate_efficiency_frontier',
    'plot_efficiency_frontier',
    'alloc_deterministic_greedy',
    'alloc_uniform',
    'evaluate_allocation_efficiency',
]
