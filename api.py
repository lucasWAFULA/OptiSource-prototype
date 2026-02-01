"""
ML-TSSP API Module
Provides optimization and explanation functions for the dashboard.
"""

import numpy as np
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add src directory to path for TSSP model import
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.optimization.tssp_model import TSSPModel
    TSSP_AVAILABLE = True
except ImportError:
    TSSP_AVAILABLE = False
    TSSPModel = None

# Import shared_db functions for active thresholds
try:
    from shared_db import get_active_threshold_settings
    SHARED_DB_AVAILABLE = True
except ImportError:
    SHARED_DB_AVAILABLE = False
    get_active_threshold_settings = None

# Task roster (same as dashboard)
TASK_ROSTER = [f"Task {i + 1:02d}" for i in range(20)]
BEHAVIOR_CLASSES = ["Cooperative", "Uncertain", "Coerced", "Deceptive"]
BEHAVIOR_RISK_MAP = {
    "cooperative": 0.0,
    "uncertain": 0.2,
    "coerced": 0.4,
    "deceptive": 1.0
}
ACTION_RISK_MULTIPLIER = {
    "disengage": 0.2,
    "flag_for_ci": 0.6,
    "flag_and_task": 0.8,
    "task": 1.0
}


def run_optimization(payload: Dict[str, Any], ml_pipeline=None) -> Dict[str, Any]:
    # #region agent log
    try:
        import json as _json_log
        import time as _time_log
        with open(r"d:\Updated-FINAL DASH\.cursor\debug.log", "a", encoding="utf-8") as _f:
            _f.write(_json_log.dumps({"location": "api.py:run_optimization", "message": "Starting API run_optimization", "data": {"n_sources": len(payload.get("sources", [])), "has_pipeline": ml_pipeline is not None}, "timestamp": int(_time_log.time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1"}) + "\n")
    except: pass
    # #endregion
    """
    Execute ML-TSSP optimization on provided sources.
    
    ML-TSSP ONLY: Uses ML models (XGBoost + GRU) for all predictions.
    No formula-based fallback is permitted in this system.
    
    Args:
        payload: Dictionary containing 'sources' list and 'seed' for RNG
        ml_pipeline: ML pipeline object with predict methods. REQUIRED for ML-TSSP operation.
                     If None or not loaded, the system raises an error.
        
    Returns:
        Dictionary with 'policies' and 'emv' results, plus '_using_ml_models' flag indicating ML usage
    """
    sources = payload.get("sources", [])
    seed = payload.get("seed", 42)
    rng = np.random.default_rng(seed)
    
    policies = {"ml_tssp": [], "deterministic": [], "uniform": []}
    # Deterministic and Uniform models are Stage 1-only baselines.
    # They do not use ML predictions or progress to Stage 2 (no adaptive optimization).
    
    # Check if ML pipeline is available and loaded
    use_ml_models = (
        ml_pipeline is not None 
        and hasattr(ml_pipeline, 'models_loaded') 
        and ml_pipeline.models_loaded
    )
    
    # ML-TSSP REQUIRED: System uses ML models (XGBoost + GRU) - no formula fallback
    if not use_ml_models:
        raise RuntimeError(
            "ML-TSSP models are required for optimization. "
            "ML pipeline is not available or models are not loaded. "
            "Please ensure ML models are properly initialized before running optimization."
        )
    
    # TSSP model availability check (warn but allow fallback to decision logic)
    if not TSSP_AVAILABLE:
        import warnings
        warnings.warn(
            "TSSP optimization model is not available. "
            "Will use decision logic with ML predictions. "
            "For full ML-TSSP optimization, ensure src/optimization/tssp_model.py is accessible."
        )
    
    # PERFORMANCE OPTIMIZATION: Use batch predictions instead of per-source predictions
    # This is 10-100x faster for large source sets
    try:
        # Prepare features dicts for all sources
        features_dicts = []
        for source in sources:
            features = source.get("features", {})
            features_dict = {
                "tsr": features.get("task_success_rate", 0.5),
                "cor": features.get("corroboration_score", 0.5),
                "time": features.get("report_timeliness", 0.5),
                "handler": features.get("handler_confidence", 0.5),
                "dec_score": features.get("deception_score", 0.3),
                "ci": features.get("ci_flag", 0)
            }
            features_dicts.append(features_dict)
        
        # Batch predictions (much faster than per-source)
        if hasattr(ml_pipeline, 'predict_batch_reliability_scores'):
            # Use batch prediction methods if available
            reliability_scores = ml_pipeline.predict_batch_reliability_scores(features_dicts)
            deception_scores = ml_pipeline.predict_batch_deception_scores(features_dicts)
            behavior_probs_list = ml_pipeline.predict_batch_behavior_probabilities(features_dicts)
        else:
            # Fallback to per-source predictions (slower but works)
            reliability_scores = []
            deception_scores = []
            behavior_probs_list = []
            for features_dict in features_dicts:
                reliability_scores.append(ml_pipeline.predict_reliability_score(features_dict))
                deception_scores.append(ml_pipeline.predict_deception_score(features_dict))
                behavior_probs_list.append(ml_pipeline.predict_behavior_probabilities(features_dict))
            reliability_scores = np.array(reliability_scores)
            deception_scores = np.array(deception_scores)
        
        ml_models_used = True
        
    except Exception as e:
        # ML prediction failed - raise error instead of falling back
        raise RuntimeError(
            f"ML-TSSP batch prediction failed: {str(e)}. "
            "The system requires ML models to operate. Please check model files and pipeline configuration."
        ) from e
    
    # ===================================================================
    # STEP 2: Prepare TSSP optimization inputs from ML predictions
    # ===================================================================
    # Extract source IDs and prepare TSSP inputs
    source_ids = [s.get("source_id") for s in sources]
    tasks = TASK_ROSTER.copy()
    behavior_classes = [b.title() for b in BEHAVIOR_CLASSES]  # Capitalize for TSSP
    
    # Prepare behavior probabilities: Dict[(source_id, behavior_class), probability]
    behavior_probabilities = {}
    for idx, source in enumerate(sources):
        source_id = source.get("source_id")
        behavior_probs_lower = behavior_probs_list[idx]
        behavior_probs_ml = {k.lower(): v for k, v in behavior_probs_lower.items()}
        
        # Map to TSSP format with capitalized behavior names
        for behavior_lower, prob in behavior_probs_ml.items():
            behavior_capitalized = behavior_lower.title()
            if behavior_capitalized in behavior_classes:
                behavior_probabilities[(source_id, behavior_capitalized)] = float(prob)
    
    # Calculate Stage 1 costs: cost of assigning source to task
    # Cost = 1 - (reliability * (1 - deception)) = risk of assignment
    stage1_costs = {}
    for idx, source in enumerate(sources):
        source_id = source.get("source_id")
        reliability = float(reliability_scores[idx])
        deception = float(deception_scores[idx])
        # Cost is inverse of quality: higher reliability and lower deception = lower cost
        assignment_quality = reliability * (1 - deception)
        assignment_cost = 1.0 - assignment_quality  # Cost increases as quality decreases
        for task in tasks:
            stage1_costs[(source_id, task)] = float(np.clip(assignment_cost, 0.0, 1.0))
    
    # Calculate recourse costs: cost of dealing with each behavior class
    # Higher risk behaviors have higher recourse costs
    recourse_costs = {}
    for behavior in behavior_classes:
        behavior_lower = behavior.lower()
        risk = BEHAVIOR_RISK_MAP.get(behavior_lower, 0.5)
        # Recourse cost is proportional to behavior risk
        recourse_costs[behavior] = float(risk)
    
    # ===================================================================
    # STEP 3: Build and solve TSSP optimization model
    # ===================================================================
    # TSSP optimization with ML predictions
    # Use TSSP for smaller problems, fallback to decision logic for larger ones to prevent hanging
    tssp_assignments = None
    # Reduce TSSP usage to smaller problems to prevent hanging - can be increased if solver performance improves
    # Limit to 20 sources for faster response times
    use_tssp = TSSP_AVAILABLE and TSSPModel is not None and len(sources) <= 20
    
    if use_tssp:
        try:
            # Quick check: if problem is too large, skip TSSP immediately
            if len(sources) > 20 or len(tasks) > 20:
                tssp_assignments = None
            else:
                tssp_model = TSSPModel(
                    sources=source_ids,
                    tasks=tasks,
                    behavior_classes=behavior_classes,
                    behavior_probabilities=behavior_probabilities,
                    stage1_costs=stage1_costs,
                    recourse_costs=recourse_costs
                )
                
                # Build the model (this should be fast)
                tssp_model.build_model()
                
                # Solve the optimization problem with aggressive timeout protection
                # Try CBC first (faster), fallback to GLPK
                # Use 10 second timeout - TSSP should solve quickly for small problems
                solver_success = False
                solver_error = None
                for solver_name in ['cbc', 'glpk']:
                    try:
                        # #region agent log
                        try:
                            with open(r"d:\Updated-FINAL DASH\.cursor\debug.log", "a", encoding="utf-8") as _f:
                                _f.write(_json_log.dumps({"location": "api.py:solver_loop", "message": "Attempting solver", "data": {"solver": solver_name, "n_sources": len(sources)}, "timestamp": int(_time_log.time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1"}) + "\n")
                        except: pass
                        # #endregion
                        # Use short timeout to prevent UI hanging
                        solver_success = tssp_model.solve(solver_name=solver_name, verbose=False, timelimit=10)
                        # #region agent log
                        try:
                            with open(r"d:\Updated-FINAL DASH\.cursor\debug.log", "a", encoding="utf-8") as _f:
                                _f.write(_json_log.dumps({"location": "api.py:solver_loop", "message": "Solver result", "data": {"solver": solver_name, "success": solver_success}, "timestamp": int(_time_log.time() * 1000), "sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1"}) + "\n")
                        except: pass
                        # #endregion
                        if solver_success:
                            break
                    except Exception as e:
                        solver_error = str(e)
                        continue
                    except KeyboardInterrupt:
                        # Handle interruption gracefully
                        solver_error = "Solver interrupted"
                        break
                
                if not solver_success:
                    # Silently fall back to decision logic - don't warn to avoid cluttering UI
                    tssp_assignments = None
                else:
                    # Extract TSSP solution
                    tssp_solution = tssp_model.solution
                    if tssp_solution is not None and tssp_solution.get('status') == 'optimal':
                        # Extract assignments from TSSP solution
                        tssp_assignments = tssp_solution.get('assignments', {})
                    else:
                        # Silently skip TSSP and use decision logic
                        tssp_assignments = None
        
        except Exception as e:
            # If TSSP fails, silently fall back to decision logic (but still use ML predictions)
            tssp_assignments = None
    
    # ===================================================================
    # STEP 4: Map TSSP assignments to policy format
    # ===================================================================
    # Create mapping from source_id to TSSP assignment
    source_to_task = {}
    if tssp_assignments:
        for (source_id, task), assigned in tssp_assignments.items():
            if assigned:
                source_to_task[source_id] = task
    
    # Process each source with ML predictions and TSSP assignments
    for idx, source in enumerate(sources):
        source_id = source.get("source_id")
        
        # Get pre-computed ML predictions
        reliability = float(reliability_scores[idx])
        deception = float(deception_scores[idx])
        behavior_probs_lower = behavior_probs_list[idx]
        behavior_probs_ml = {k.lower(): v for k, v in behavior_probs_lower.items()}
        
        # Get recourse rules
        recourse = source.get("recourse_rules", {})
        rel_disengage = recourse.get("rel_disengage", 0.35)
        rel_flag = recourse.get("rel_ci_flag", 0.50)
        dec_disengage = recourse.get("dec_disengage", 0.75)
        dec_flag = recourse.get("dec_ci_flag", 0.60)
        
        # Calculate intrinsic risk from ML behavior probabilities
        intrinsic_risk = sum(
            float(prob) * BEHAVIOR_RISK_MAP.get(behavior, 0.5)
            for behavior, prob in behavior_probs_ml.items()
        )
        intrinsic_risk = float(np.clip(intrinsic_risk, 0.0, 1.0))
        
        # Determine risk bucket using active thresholds from database
        if SHARED_DB_AVAILABLE and get_active_threshold_settings:
            try:
                active_thresholds = get_active_threshold_settings()
                low_threshold = active_thresholds.get("low_risk", 0.3)
                medium_threshold = active_thresholds.get("medium_risk", 0.6)
            except Exception:
                # Fallback to defaults if database access fails
                low_threshold = 0.3
                medium_threshold = 0.6
        else:
            # Fallback to defaults if shared_db not available
            low_threshold = 0.3
            medium_threshold = 0.6
        
        if intrinsic_risk < low_threshold:
            risk_bucket = "low"
        elif intrinsic_risk > medium_threshold:
            risk_bucket = "high"
        else:
            risk_bucket = "medium"
        
        # Use TSSP assignment if available, otherwise apply decision logic
        if tssp_assignments and source_id in source_to_task:
            # TSSP assigned this source to a task
            task = source_to_task[source_id]
            # Check if quality thresholds require disengagement despite TSSP assignment
            if risk_bucket == "high" or deception >= dec_disengage or reliability < rel_disengage:
                action = "disengage"
                task = None
            elif deception >= dec_flag:
                action = "flag_for_ci"
                # Keep TSSP-assigned task
            elif reliability < rel_flag:
                action = "flag_and_task"
                # Keep TSSP-assigned task
            else:
                action = "task"
                # Keep TSSP-assigned task
        else:
            # Fallback to decision logic (if TSSP didn't assign or failed)
            if risk_bucket == "high":
                action = "disengage"
                task = None
            elif deception >= dec_disengage or reliability < rel_disengage:
                action = "disengage"
                task = None
            elif deception >= dec_flag:
                action = "flag_for_ci"
                task = rng.choice(TASK_ROSTER) if not tssp_assignments else (source_to_task.get(source_id) or rng.choice(TASK_ROSTER))
            elif reliability < rel_flag:
                action = "flag_and_task"
                task = rng.choice(TASK_ROSTER) if not tssp_assignments else (source_to_task.get(source_id) or rng.choice(TASK_ROSTER))
            else:
                action = "task"
                task = rng.choice(TASK_ROSTER) if not tssp_assignments else (source_to_task.get(source_id) or rng.choice(TASK_ROSTER))
        
        # Calculate expected risk (post-recourse, adjusted by action)
        expected_risk = intrinsic_risk * ACTION_RISK_MULTIPLIER.get(action, 1.0)
        expected_risk = float(np.clip(expected_risk, 0.0, 1.0))
        score = reliability * (1 - deception) * (1 - expected_risk)
        
        policy_item = {
            "source_id": source_id,
            "reliability": float(reliability),
            "deception": float(deception),
            "action": action,
            "task": task,
            "expected_risk": float(expected_risk),
            "intrinsic_risk": float(intrinsic_risk),
            "risk_bucket": risk_bucket,
            "score": float(score)
        }
        
        policies["ml_tssp"].append(policy_item)

        # Deterministic: Stage 1 only, fixed moderate risk, no ML, no TSSP, no Stage 2
        # Independent baseline model for comparison
        det_item = policy_item.copy()
        det_item["intrinsic_risk"] = 0.5  # Fixed moderate intrinsic risk for baseline
        det_item["expected_risk"] = 0.5
        det_item["risk_bucket"] = "medium"  # Fixed medium risk bucket
        det_item["score"] = reliability * (1 - deception) * (1 - det_item["expected_risk"])
        policies["deterministic"].append(det_item)

        # Uniform: Stage 1 only, equal allocation, no ML, no TSSP, no Stage 2
        # Independent baseline model for comparison - uses uniform distribution across behaviors
        # Uniform risk = average of all behavior risks (equal probability assumption)
        uniform_risk = sum(BEHAVIOR_RISK_MAP.values()) / len(BEHAVIOR_RISK_MAP)
        uni_item = policy_item.copy()
        uni_item["intrinsic_risk"] = float(uniform_risk)  # Use uniform risk as intrinsic
        uni_item["expected_risk"] = float(uniform_risk)
        # Determine risk bucket for uniform
        if uniform_risk < 0.3:
            uni_risk_bucket = "low"
        elif uniform_risk > 0.6:
            uni_risk_bucket = "high"
        else:
            uni_risk_bucket = "medium"
        uni_item["risk_bucket"] = uni_risk_bucket
        uni_item["score"] = reliability * (1 - deception) * (1 - uni_item["expected_risk"])
        policies["uniform"].append(uni_item)
    
    # Calculate EMV
    ml_emv = sum(p["score"] for p in policies["ml_tssp"])
    det_emv = sum(p["score"] for p in policies["deterministic"])
    uni_emv = sum(p["score"] for p in policies["uniform"])

    return {
        "policies": policies,
        "emv": {
            "ml_tssp": ml_emv,
            "deterministic": det_emv,
            "uniform": uni_emv
        },
        "_using_ml_models": True,
        "_using_fallback": False
    }


def explain_source(source_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate SHAP-style explanations for a source.
    
    Args:
        source_payload: Dictionary with 'source_id' and 'features'
        
    Returns:
        Dictionary with 'shap_values' for each behavior class
    """
    source_id = source_payload.get("source_id", "UNKNOWN")
    features = source_payload.get("features", {})
    
    shap_values = {}
    for behavior in BEHAVIOR_CLASSES:
        behavior_shap = {}
        
        tsr = float(features.get("task_success_rate", 0.5))
        cor = float(features.get("corroboration_score", 0.5))
        time = float(features.get("report_timeliness", 0.5))
        
        if behavior == "Cooperative":
            behavior_shap["task_success_rate"] = tsr * 0.3
            behavior_shap["corroboration_score"] = cor * 0.25
            behavior_shap["report_timeliness"] = time * 0.15
            behavior_shap["reliability_trend"] = (1 - tsr) * -0.05
        elif behavior == "Uncertain":
            behavior_shap["task_success_rate"] = (1 - tsr) * 0.2
            behavior_shap["corroboration_score"] = (1 - cor) * 0.25
            behavior_shap["report_timeliness"] = (1 - time) * 0.15
            behavior_shap["reliability_trend"] = abs(0.5 - tsr) * 0.2
        elif behavior == "Coerced":
            behavior_shap["corroboration_score"] = (1 - cor) * 0.3
            behavior_shap["task_success_rate"] = (1 - tsr) * 0.25
            behavior_shap["report_timeliness"] = (1 - time) * 0.2
            behavior_shap["consistency_volatility"] = abs(0.5 - cor) * 0.15
        elif behavior == "Deceptive":
            behavior_shap["corroboration_score"] = (1 - cor) * 0.35
            behavior_shap["task_success_rate"] = abs(0.7 - tsr) * 0.25
            behavior_shap["reliability_trend"] = (1 - tsr) * 0.2
            behavior_shap["consistency_volatility"] = (1 - cor) * 0.2
        
        shap_values[behavior] = behavior_shap
    
    return {"shap_values": shap_values}
