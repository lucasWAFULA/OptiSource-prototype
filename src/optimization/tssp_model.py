"""
Two-Stage Stochastic Programming (TSSP) model for HUMINT source-task assignment.

This module implements the Pyomo optimization model that combines ML predictions
with stochastic optimization to make optimal source-task assignments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Objective, Constraint, Binary, NonNegativeReals,
    minimize, value, SolverFactory, TerminationCondition
)
from pyomo.opt import SolverStatus, TerminationCondition as PyomoTerminationCondition


class TSSPModel:
    """
    Two-Stage Stochastic Programming model for HUMINT source-task assignment.
    
    Stage 1: Strategic tasking decisions (assign sources to tasks)
    Stage 2: Recourse decisions (adjustments based on realized behaviors)
    """
    
    def __init__(
        self,
        sources: List[str],
        tasks: List[str],
        behavior_classes: List[str],
        behavior_probabilities: Dict[Tuple[str, str], float],
        stage1_costs: Dict[Tuple[str, str], float],
        recourse_costs: Dict[str, float],
        information_values: Optional[Dict[Tuple[str, str], float]] = None
    ):
        """
        Initialize TSSP model.
        
        Parameters:
        -----------
        sources : List[str]
            List of source IDs
        tasks : List[str]
            List of task IDs
        behavior_classes : List[str]
            List of behavior class names
        behavior_probabilities : Dict[Tuple[str, str], float]
            Probability of each source exhibiting each behavior class
            Key: (source_id, behavior_class), Value: probability
        stage1_costs : Dict[Tuple[str, str], float]
            Stage 1 cost for assigning each source to each task
            Key: (source_id, task_id), Value: cost
        recourse_costs : Dict[str, float]
            Recourse cost for each behavior class
            Key: behavior_class, Value: cost
        information_values : Optional[Dict[Tuple[str, str], float]]
            Information value of assigning each source to each task
            Key: (source_id, task_id), Value: value
        """
        self.sources = sources
        self.tasks = tasks
        self.behavior_classes = behavior_classes
        self.behavior_probabilities = behavior_probabilities
        self.stage1_costs = stage1_costs
        self.recourse_costs = recourse_costs
        self.information_values = information_values or {}
        
        self.model = None
        self.solution = None
    
    def build_model(self):
        """Build the Pyomo optimization model."""
        model = ConcreteModel()
        
        # Sets
        model.S = Set(initialize=self.sources)  # Sources
        model.T = Set(initialize=self.tasks)    # Tasks
        model.B = Set(initialize=self.behavior_classes)  # Behavior classes
        
        # Pre-compute non-cooperative behavior classes (those with recourse cost > 0)
        non_cooperative_behaviors = [
            b for b in self.behavior_classes 
            if self.recourse_costs.get(b, 0) > 0
        ]
        model.B_noncoop = Set(initialize=non_cooperative_behaviors)  # Non-cooperative behaviors
        
        # Parameters
        # Stage 1 costs
        def stage1_cost_init(m, s, t):
            return self.stage1_costs.get((s, t), 0.0)
        model.Stage1Cost = Param(model.S, model.T, initialize=stage1_cost_init)
        
        # Recourse costs
        model.RecourseCost = Param(model.B, initialize=self.recourse_costs)
        
        # Behavior probabilities
        def behavior_prob_init(m, s, b):
            return self.behavior_probabilities.get((s, b), 0.0)
        model.BehaviorProb = Param(model.S, model.B, initialize=behavior_prob_init)
        
        # Pre-compute non-cooperative probabilities for each source
        # This helps in creating constraints that ensure recourse when needed
        def non_coop_prob_init(m, s):
            return sum(
                self.behavior_probabilities.get((s, b), 0.0)
                for b in non_cooperative_behaviors
            )
        model.NonCoopProb = Param(model.S, initialize=non_coop_prob_init)
        
        # Information values (optional)
        if self.information_values:
            def info_value_init(m, s, t):
                return self.information_values.get((s, t), 0.0)
            model.InformationValue = Param(model.S, model.T, initialize=info_value_init)
        
        # Decision Variables
        # Stage 1: Binary assignment of source s to task t
        model.x = Var(model.S, model.T, domain=Binary)
        
        # Stage 2: Recourse action intensity
        # y[s, t, b] represents the recourse action intensity for source s on task t
        # when behavior b is realized. This is the adjustment needed after uncertainty is resolved.
        model.y = Var(model.S, model.T, model.B, domain=NonNegativeReals)
        
        # Objective Function
        # Minimize: Stage 1 costs + Expected Stage 2 recourse costs
        # Following the formulation: min_x sum_i c_i x_i + E_ω[Q(x, ω)]
        # where Q(x, ω) = min_y sum_i q_i(ω) y_i(ω)
        def objective_rule(m):
            # Stage 1: Strategic tasking costs (here-and-now decisions)
            stage1 = sum(
                m.Stage1Cost[s, t] * m.x[s, t]
                for s in m.S for t in m.T
            )
            
            # Stage 2: Expected recourse costs (after uncertainty is realized)
            # Following the formulation: E[q_i] = Σ_ω p(ω) × q_i(ω) × y_i(ω)
            # where:
            #   - p(ω) = BehaviorProb[s, b] (probability of behavior scenario b for source s)
            #   - q_i(ω) = RecourseCost[b] (cost per unit recourse for behavior b)
            #   - y_i(ω) = y[s, t, b] (recourse action intensity)
            # Expected cost: Σ_s Σ_t Σ_b BehaviorProb[s, b] × RecourseCost[b] × y[s, t, b]
            stage2 = sum(
                m.BehaviorProb[s, b] *  # p(ω): Probability of scenario (behavior b) for source s
                m.RecourseCost[b] *      # q_i(ω): Recourse cost per unit for behavior b
                m.y[s, t, b]             # y_i(ω): Recourse action intensity (decision variable)
                for s in m.S for t in m.T for b in m.B
            )
            return stage1 + stage2
        
        model.Obj = Objective(rule=objective_rule, sense=minimize)
        
        # Constraints
        # Each source can be assigned to at most one task
        def source_capacity_rule(m, s):
            return sum(m.x[s, t] for t in m.T) <= 1
        model.SourceCapacity = Constraint(model.S, rule=source_capacity_rule)
        
        # Each task needs at least one source assigned
        def task_coverage_rule(m, t):
            return sum(m.x[s, t] for s in m.S) >= 1
        model.TaskCoverage = Constraint(model.T, rule=task_coverage_rule)
        
        # Recourse can only occur if source is assigned to task
        # y[s, t, b] <= x[s, t] for all behaviors
        def recourse_feasibility_rule(m, s, t, b):
            return m.y[s, t, b] <= m.x[s, t]
        model.RecourseFeasibility = Constraint(
            model.S, model.T, model.B, rule=recourse_feasibility_rule
        )
        
        # Force recourse to be zero when recourse cost is zero (cooperative behavior)
        # q_i(ω) = 0 for cooperative behavior, so y[s, t, 'cooperative'] = 0
        def zero_recourse_cooperative_rule(m, s, t, b):
            if value(m.RecourseCost[b]) == 0:
                return m.y[s, t, b] == 0
            else:
                return Constraint.Skip
        model.ZeroRecourseCooperative = Constraint(
            model.S, model.T, model.B, rule=zero_recourse_cooperative_rule
        )
        
        # Ensure recourse is triggered when non-cooperative behaviors have positive probability
        # Following the formulation: if source is assigned and has non-cooperative behavior probability,
        # then recourse must be positive to ensure Stage 2 cost is non-zero
        # This implements: q_i(ω) > 0 if source is uncooperative, ensuring recourse is needed
        def recourse_trigger_rule(m, s, t):
            # Sum of recourse for non-cooperative behaviors only
            non_coop_recourse = sum(m.y[s, t, b] for b in m.B_noncoop)
            
            # If source is assigned and has non-zero non-cooperative probability,
            # then recourse must be triggered
            # Following the formulation: recourse should be triggered when non-cooperative behaviors occur
            # Constraint ensures Stage 2 cost is non-zero when non-cooperative behaviors are present
            # We use: non_coop_recourse >= x[s, t] * NonCoopProb[s] to ensure recourse is triggered
            # This means recourse intensity should be at least proportional to the probability
            if value(m.NonCoopProb[s]) > 1e-6:
                # Minimum recourse should be proportional to assignment and non-cooperative probability
                # When x[s,t] = 1 and NonCoopProb[s] > 0, recourse must be triggered
                # This ensures E[q_i] = Σ prob × cost × y is non-zero for non-cooperative behaviors
                return non_coop_recourse >= m.x[s, t] * m.NonCoopProb[s]
            else:
                # Source is fully cooperative (all behaviors have zero recourse cost)
                return Constraint.Skip
        model.RecourseTrigger = Constraint(
            model.S, model.T, rule=recourse_trigger_rule
        )
        
        self.model = model
        return model
    
    def solve(self, solver_name: str = 'glpk', verbose: bool = False, timelimit: int = 600) -> bool:
        """
        Solve the optimization model.
        
        Parameters:
        -----------
        solver_name : str
            Name of the solver to use (default: 'glpk')
        verbose : bool
            Whether to print solver output
        
        Returns:
        --------
        bool
            True if solution is optimal, False otherwise
        """
        if self.model is None:
            self.build_model()
        
        solver = SolverFactory(solver_name)
        if solver is None:
            raise ValueError(f"Solver '{solver_name}' not available. "
                           f"Please install a solver (e.g., 'pip install glpk' or use 'cbc')")
        
        # Add a time limit to prevent endless runs (default 600 seconds)
        solver_options = {}
        # Only CBC supports a time limit via 'seconds' option
        if solver_name.lower() == 'cbc':
            solver_options['seconds'] = timelimit
            results = solver.solve(self.model, tee=verbose, options=solver_options)
        else:
            # For GLPK and others, do not set a time limit (unsupported)
            results = solver.solve(self.model, tee=verbose)
        
        # Check solution status
        if (results.solver.status == SolverStatus.ok and
            results.solver.termination_condition == PyomoTerminationCondition.optimal):
            self.solution = {
                'status': 'optimal',
                'objective_value': value(self.model.Obj),
                'assignments': {},
                'recourse': {}
            }
            
            # Extract assignments
            for s in self.model.S:
                for t in self.model.T:
                    if self.model.x[s, t].value is not None and self.model.x[s, t].value > 0.5:
                        self.solution['assignments'][(s, t)] = 1
            
            # Extract recourse decisions
            for s in self.model.S:
                for t in self.model.T:
                    for b in self.model.B:
                        if (self.model.y[s, t, b].value is not None and
                            self.model.y[s, t, b].value > 1e-6):
                            self.solution['recourse'][(s, t, b)] = self.model.y[s, t, b].value
            
            return True
        else:
            self.solution = {
                'status': 'failed',
                'message': f"Solver status: {results.solver.status}, "
                          f"Termination: {results.solver.termination_condition}"
            }
            return False
    
    def get_stage1_cost(self) -> float:
        """Calculate total Stage 1 cost from solution."""
        if self.solution is None or self.solution['status'] != 'optimal':
            return 0.0
        
        total = 0.0
        for (s, t), _ in self.solution['assignments'].items():
            total += self.stage1_costs.get((s, t), 0.0)
        return total
    
    def get_stage2_expected_cost(self) -> float:
        """Calculate total Stage 2 expected recourse cost from solution."""
        if self.solution is None or self.solution['status'] != 'optimal':
            return 0.0
        
        total = 0.0
        for (s, t, b), y_value in self.solution['recourse'].items():
            prob = self.behavior_probabilities.get((s, b), 0.0)
            cost = self.recourse_costs.get(b, 0.0)
            total += prob * cost * y_value
        return total
    
    def get_total_cost(self) -> float:
        """Get total expected cost (Stage 1 + Stage 2)."""
        return self.get_stage1_cost() + self.get_stage2_expected_cost()
    
    def get_cost_decomposition(self) -> Dict:
        """
        Get detailed cost decomposition.
        
        Returns:
        --------
        Dictionary with cost breakdown by behavior class and source
        """
        if self.solution is None or self.solution['status'] != 'optimal':
            return {}
        
        stage1_cost = self.get_stage1_cost()
        stage2_cost = self.get_stage2_expected_cost()
        total_cost = stage1_cost + stage2_cost
        
        # Breakdown by behavior class
        behavior_breakdown = {}
        for b in self.behavior_classes:
            behavior_cost = 0.0
            for (s, t, b_class), y_value in self.solution['recourse'].items():
                if b_class == b:
                    prob = self.behavior_probabilities.get((s, b), 0.0)
                    cost = self.recourse_costs.get(b, 0.0)
                    behavior_cost += prob * cost * y_value
            behavior_breakdown[b] = behavior_cost
        
        # Breakdown by source
        source_breakdown = {}
        for s in self.sources:
            source_cost = 0.0
            for (s_assign, t, b), y_value in self.solution['recourse'].items():
                if s_assign == s:
                    prob = self.behavior_probabilities.get((s, b), 0.0)
                    cost = self.recourse_costs.get(b, 0.0)
                    source_cost += prob * cost * y_value
            if source_cost > 0:
                source_breakdown[s] = source_cost
        
        return {
            'stage1_cost': stage1_cost,
            'stage2_expected_cost': stage2_cost,
            'total_cost': total_cost,
            'stage2_proportion': stage2_cost / total_cost if total_cost > 0 else 0.0,
            'by_behavior': behavior_breakdown,
            'by_source': source_breakdown
        }

    def evaluate_assignment(
        self,
        assignments: Dict[Tuple[str, str], int],
        solver_name: str = 'glpk',
        verbose: bool = False
    ) -> Dict:
        """
        Evaluate a fixed source–task assignment by solving the recourse subproblem
        (Stage 2) and returning Stage 1 + Stage 2 costs. Uses the same cost model
        as the full TSSP for fair comparison.

        Parameters
        ----------
        assignments : Dict[Tuple[str, str], int]
            Map (source_id, task_id) -> 0 or 1. Must satisfy: each task >= 1 source,
            each source <= 1 task.
        solver_name : str
            Solver for the recourse LP (default: 'glpk').
        verbose : bool
            Whether to print solver output.

        Returns
        -------
        Dict with keys:
            - success : bool
            - stage1_cost : float
            - stage2_expected_cost : float
            - total_cost : float
            - n_assignments : int
            - message : str (if success is False)
        """
        stage1_cost = sum(
            self.stage1_costs.get((s, t), 0.0) * v
            for (s, t), v in assignments.items()
            if v
        )
        n_assign = sum(1 for v in assignments.values() if v)

        # Build a temporary model, fix x, solve for y only
        eval_model = TSSPModel(
            sources=self.sources,
            tasks=self.tasks,
            behavior_classes=self.behavior_classes,
            behavior_probabilities=self.behavior_probabilities,
            stage1_costs=self.stage1_costs,
            recourse_costs=self.recourse_costs,
            information_values=self.information_values,
        )
        eval_model.build_model()
        m = eval_model.model

        for s in m.S:
            for t in m.T:
                val = 1.0 if assignments.get((s, t), 0) else 0.0
                m.x[s, t].fix(val)

        solver = SolverFactory(solver_name)
        if solver is None:
            return {
                'success': False,
                'stage1_cost': stage1_cost,
                'stage2_expected_cost': float('nan'),
                'total_cost': float('nan'),
                'n_assignments': n_assign,
                'message': f"Solver '{solver_name}' not available.",
            }

        try:
            results = solver.solve(m, tee=verbose)
        except Exception as e:
            return {
                'success': False,
                'stage1_cost': stage1_cost,
                'stage2_expected_cost': float('nan'),
                'total_cost': float('nan'),
                'n_assignments': n_assign,
                'message': str(e),
            }

        if (results.solver.status != SolverStatus.ok or
                results.solver.termination_condition != PyomoTerminationCondition.optimal):
            return {
                'success': False,
                'stage1_cost': stage1_cost,
                'stage2_expected_cost': float('nan'),
                'total_cost': float('nan'),
                'n_assignments': n_assign,
                'message': (f"Solver status: {results.solver.status}, "
                            f"termination: {results.solver.termination_condition}"),
            }

        # Reconstruct solution for cost computation
        recourse = {}
        for s in m.S:
            for t in m.T:
                for b in m.B:
                    if m.y[s, t, b].value is not None and m.y[s, t, b].value > 1e-6:
                        recourse[(s, t, b)] = m.y[s, t, b].value

        stage2_cost = 0.0
        for (s, t, b), yval in recourse.items():
            prob = self.behavior_probabilities.get((s, b), 0.0)
            cost = self.recourse_costs.get(b, 0.0)
            stage2_cost += prob * cost * yval

        total = stage1_cost + stage2_cost
        return {
            'success': True,
            'stage1_cost': stage1_cost,
            'stage2_expected_cost': stage2_cost,
            'total_cost': total,
            'n_assignments': n_assign,
            'message': None,
        }
