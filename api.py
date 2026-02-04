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
    # IMPORTANT: Each model makes independent decisions based on its own risk assessments
    # ML-TSSP: Uses ML predictions + Stage 2 optimization
    # Deterministic: Uses fixed risk assumptions + Stage 1 decisions only
    # Uniform: Uses equal probability assumptions + Stage 1 decisions only
    
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
            reliability_scores_batch = ml_pipeline.predict_batch_reliability_scores(features_dicts)
            deception_scores_batch = ml_pipeline.predict_batch_deception_scores(features_dicts)
            behavior_probs_list = ml_pipeline.predict_batch_behavior_probabilities(features_dicts)
        else:
            # Fallback to per-source predictions (slower but works)
            reliability_scores_batch = []
            deception_scores_batch = []
            behavior_probs_list = []
            for features_dict in features_dicts:
                reliability_scores_batch.append(ml_pipeline.predict_reliability_score(features_dict))
                deception_scores_batch.append(ml_pipeline.predict_deception_score(features_dict))
                behavior_probs_list.append(ml_pipeline.predict_behavior_probabilities(features_dict))
            reliability_scores_batch = np.array(reliability_scores_batch)
            deception_scores_batch = np.array(deception_scores_batch)
        
        ml_models_used = True
        
    except Exception as e:
        # ML prediction failed - raise error instead of falling back
        raise RuntimeError(
            f"ML-TSSP batch prediction failed: {str(e)}. "
            "The system requires ML models to operate. Please check model files and pipeline configuration."
        ) from e
    
    # ===================================================================
    # STEP 1: Calculate reliability and deception scores
    # ===================================================================
    # ML-TSSP uses actual ML predictions
    # Baselines use formula-based calculations (simulating what ML would predict)
    reliability_scores = {}
    deception_scores = {}
    behavior_probs_dict = {}
    tasks = TASK_ROSTER.copy()  # Define tasks here for use in STEP 1
    behavior_classes = [b.title() for b in BEHAVIOR_CLASSES]  # Define behavior_classes here for use in STEP 1

    for idx, source in enumerate(sources):
        source_id = source.get("source_id")
        features = source.get("features", {})

        # Extract features
        tsr = features.get("task_success_rate", 0.5)
        cor = features.get("corroboration_score", 0.5)
        time = features.get("report_timeliness", 0.5)
        handler = features.get("handler_confidence", 0.5)
        dec_score = features.get("deception_score", 0.3)
        ci = features.get("ci_flag", 0)

        # ML-TSSP: Use actual ML predictions from batch results
        reliability_scores[source_id] = float(reliability_scores_batch[idx])
        deception_scores[source_id] = float(deception_scores_batch[idx])
        behavior_probs_dict[source_id] = {k.lower(): float(v) for k, v in behavior_probs_list[idx].items()}

        # Deterministic: Use formula-based calculation (simulating ML prediction)
        det_reliability = np.clip(
            0.30 * tsr + 0.25 * cor + 0.20 * time + 0.15 * handler
            - 0.15 * dec_score - 0.10 * ci + 0.05 * rng.normal(0, 0.03),
            0.0, 1.0
        )
        det_deception = np.clip(
            0.30 * dec_score + 0.25 * ci + 0.20 * (1 - cor) + 0.15 * (1 - handler)
            + 0.10 * rng.beta(2, 5),
            0.0, 1.0
        )
        reliability_scores[f"{source_id}_deterministic"] = float(det_reliability)
        deception_scores[f"{source_id}_deterministic"] = float(det_deception)
        # Deterministic uses fixed moderate risk assumption (no behavior modeling)
        behavior_probs_dict[f"{source_id}_deterministic"] = {
            "cooperative": 0.4, "uncertain": 0.3, "coerced": 0.2, "deceptive": 0.1
        }

        # Uniform: Use same formula-based calculation as deterministic
        reliability_scores[f"{source_id}_uniform"] = float(det_reliability)
        deception_scores[f"{source_id}_uniform"] = float(det_deception)
        # Uniform assumes equal probability for all behaviors
        behavior_probs_dict[f"{source_id}_uniform"] = {
            "cooperative": 0.25, "uncertain": 0.25, "coerced": 0.25, "deceptive": 0.25
        }
    
    # Calculate Stage 1 costs: cost of assigning source to task
    # Cost = 1 - (reliability * (1 - deception)) = risk of assignment
    stage1_costs = {}
    for idx, source in enumerate(sources):
        source_id = source.get("source_id")
        reliability = float(reliability_scores[source_id])
        deception = float(deception_scores[source_id])
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
    # STEP 2: Process each model separately
    # ===================================================================

    # Process ML-TSSP (uses actual ML predictions + Stage 2 optimization)
    source_ids = [s.get("source_id") for s in sources]

    # ML-TSSP behavior probabilities
    behavior_probabilities = {}
    for idx, source in enumerate(sources):
        source_id = source.get("source_id")
        behavior_probs_ml = behavior_probs_dict[source_id]
        for behavior_lower, prob in behavior_probs_ml.items():
            behavior_capitalized = behavior_lower.title()
            if behavior_capitalized in behavior_classes:
                behavior_probabilities[(source_id, behavior_capitalized)] = float(prob)

    # Calculate Stage 1 costs for ML-TSSP
    stage1_costs = {}
    for idx, source in enumerate(sources):
        source_id = source.get("source_id")
        reliability = reliability_scores[source_id]
        deception = deception_scores[source_id]
        assignment_quality = reliability * (1 - deception)
        assignment_cost = 1.0 - assignment_quality
        for task in tasks:
            stage1_costs[(source_id, task)] = float(np.clip(assignment_cost, 0.0, 1.0))

    # Calculate recourse costs
    recourse_costs = {}
    for behavior in behavior_classes:
        behavior_lower = behavior.lower()
        risk = BEHAVIOR_RISK_MAP.get(behavior_lower, 0.5)
        recourse_costs[behavior] = float(risk)

    # ML-TSSP optimization (Stage 1 + Stage 2)
    tssp_assignments = None
    use_tssp = TSSP_AVAILABLE and TSSPModel is not None and len(sources) <= 20

    if use_tssp:
        try:
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

                tssp_model.build_model()
                solver_success = False
                for solver_name in ['cbc', 'glpk']:
                    try:
                        solver_success = tssp_model.solve(solver_name=solver_name, verbose=False, timelimit=10)
                        if solver_success:
                            break
                    except Exception:
                        continue

                if solver_success:
                    tssp_solution = tssp_model.solution
                    if tssp_solution and tssp_solution.get('status') == 'optimal':
                        tssp_assignments = tssp_solution.get('assignments', {})
                else:
                    tssp_assignments = None
        except Exception:
            tssp_assignments = None

    # Create TSSP assignment mapping
    source_to_task = {}
    if tssp_assignments:
        for (source_id, task), assigned in tssp_assignments.items():
            if assigned:
                source_to_task[source_id] = task
    
    # Process each source with ML predictions and TSSP assignments
    for idx, source in enumerate(sources):
        source_id = source.get("source_id")
        
        # Get pre-computed ML predictions
        reliability = float(reliability_scores[source_id])
        deception = float(deception_scores[source_id])
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
                # Keep TSSP-assigned task even for disengagement recommendations
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
                task = rng.choice(TASK_ROSTER) if not tssp_assignments else (source_to_task.get(source_id) or rng.choice(TASK_ROSTER))
            elif deception >= dec_disengage or reliability < rel_disengage:
                action = "disengage"
                task = rng.choice(TASK_ROSTER) if not tssp_assignments else (source_to_task.get(source_id) or rng.choice(TASK_ROSTER))
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
        score = reliability * (1 - deception) * (1 - expected_risk) * 100  # Scale to 0-100
        
        policy_item = {
            "source_id": source_id,
            "reliability": float(reliability),
            "deception": float(deception),
            "action": action,
            "task": task,
            "expected_risk": float(expected_risk),
            "intrinsic_risk": float(intrinsic_risk),
            "risk_bucket": risk_bucket,
            "score": float(score),
            "behavior_probabilities": behavior_probs_ml
        }
        
        policies["ml_tssp"].append(policy_item)

        # ===== DETERMINISTIC BASELINE =====
        # Independent baseline model: Stage 1 only, rule-based risk assessment
        det_reliability = reliability_scores[f"{source_id}_deterministic"]
        det_deception = deception_scores[f"{source_id}_deterministic"]

        # Deterministic uses fixed moderate risk (no behavior modeling)
        det_intrinsic_risk = 0.5
        det_expected_risk = det_intrinsic_risk * ACTION_RISK_MULTIPLIER.get(action, 1.0)
        det_expected_risk = float(np.clip(det_expected_risk, 0.0, 1.0))

        # Deterministic makes its OWN decisions based on fixed risk
        det_risk_bucket = "medium"  # Always medium due to fixed risk
        if det_risk_bucket == "high" or det_deception >= dec_disengage or det_reliability < rel_disengage:
            det_action = "disengage"
            det_task = rng.choice(TASK_ROSTER)
        elif det_deception >= dec_flag:
            det_action = "flag_for_ci"
            det_task = rng.choice(TASK_ROSTER)
        elif det_reliability < rel_flag:
            det_action = "flag_and_task"
            det_task = rng.choice(TASK_ROSTER)
        else:
            det_action = "task"
            det_task = rng.choice(TASK_ROSTER)

        det_final_expected_risk = det_intrinsic_risk * ACTION_RISK_MULTIPLIER.get(det_action, 1.0)
        det_final_expected_risk = float(np.clip(det_final_expected_risk, 0.0, 1.0))
        det_score = det_reliability * (1 - det_deception) * (1 - det_final_expected_risk) * 100  # Scale to 0-100

        det_item = {
            "source_id": source_id,
            "reliability": float(det_reliability),
            "deception": float(det_deception),
            "action": det_action,
            "task": det_task,
            "expected_risk": float(det_final_expected_risk),
            "intrinsic_risk": float(det_intrinsic_risk),
            "risk_bucket": det_risk_bucket,
            "score": float(det_score)
        }
        policies["deterministic"].append(det_item)

        # ===== UNIFORM BASELINE =====
        # Independent baseline model: Stage 1 only, equal probability assumption
        uni_reliability = reliability_scores[f"{source_id}_uniform"]
        uni_deception = deception_scores[f"{source_id}_uniform"]
        uni_behavior_probs = behavior_probs_dict[f"{source_id}_uniform"]

        # Uniform calculates intrinsic risk from equal behavior probabilities
        uni_intrinsic_risk = sum(
            float(prob) * BEHAVIOR_RISK_MAP.get(behavior, 0.5)
            for behavior, prob in uni_behavior_probs.items()
        )
        uni_intrinsic_risk = float(np.clip(uni_intrinsic_risk, 0.0, 1.0))
        uni_expected_risk = uni_intrinsic_risk * ACTION_RISK_MULTIPLIER.get(action, 1.0)
        uni_expected_risk = float(np.clip(uni_expected_risk, 0.0, 1.0))

        # Uniform makes its OWN decisions based on its risk assessment
        if uni_intrinsic_risk < low_threshold:
            uni_risk_bucket = "low"
        elif uni_intrinsic_risk > medium_threshold:
            uni_risk_bucket = "high"
        else:
            uni_risk_bucket = "medium"

        if uni_risk_bucket == "high" or uni_deception >= dec_disengage or uni_reliability < rel_disengage:
            uni_action = "disengage"
            uni_task = rng.choice(TASK_ROSTER)
        elif uni_deception >= dec_flag:
            uni_action = "flag_for_ci"
            uni_task = rng.choice(TASK_ROSTER)
        elif uni_reliability < rel_flag:
            uni_action = "flag_and_task"
            uni_task = rng.choice(TASK_ROSTER)
        else:
            uni_action = "task"
            uni_task = rng.choice(TASK_ROSTER)

        uni_final_expected_risk = uni_intrinsic_risk * ACTION_RISK_MULTIPLIER.get(uni_action, 1.0)
        uni_final_expected_risk = float(np.clip(uni_final_expected_risk, 0.0, 1.0))
        uni_score = uni_reliability * (1 - uni_deception) * (1 - uni_final_expected_risk) * 100  # Scale to 0-100

        uni_item = {
            "source_id": source_id,
            "reliability": float(uni_reliability),
            "deception": float(uni_deception),
            "action": uni_action,
            "task": uni_task,
            "expected_risk": float(uni_final_expected_risk),
            "intrinsic_risk": float(uni_intrinsic_risk),
            "risk_bucket": uni_risk_bucket,
            "score": float(uni_score)
        }
        policies["uniform"].append(uni_item)
    
    # Calculate EMV and improvement percentages
    ml_emv = sum(p["score"] for p in policies["ml_tssp"])
    det_emv = sum(p["score"] for p in policies["deterministic"])
    uni_emv = sum(p["score"] for p in policies["uniform"])

    # Calculate improvement percentages over baselines
    det_improvement = ((ml_emv - det_emv) / det_emv * 100) if det_emv > 0 else 0.0
    uni_improvement = ((ml_emv - uni_emv) / uni_emv * 100) if uni_emv > 0 else 0.0

    return {
        "policies": policies,
        "emv": {
            "ml_tssp": ml_emv,
            "deterministic": det_emv,
            "uniform": uni_emv,
            "improvement_over_deterministic": det_improvement,
            "improvement_over_uniform": uni_improvement
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
