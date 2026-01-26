"""
ML-TSSP API Module
Provides optimization and explanation functions for the dashboard.
"""

import numpy as np
from typing import Dict, List, Any

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
    """
    Execute ML-TSSP optimization on provided sources.
    
    PRIMARY METHOD: Uses ML models (XGBoost + GRU) for all predictions.
    FALLBACK ONLY: Formula-based calculations are used ONLY when ML models are unavailable or fail.
    
    The system is ML-TSSP driven - formulas are a safety fallback, not the primary method.
    
    Args:
        payload: Dictionary containing 'sources' list and 'seed' for RNG
        ml_pipeline: ML pipeline object with predict methods. REQUIRED for ML-TSSP operation.
                     If None or not loaded, system falls back to formulas (not recommended for production).
        
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
    
    # Process each source with pre-computed predictions
    for idx, source in enumerate(sources):
        # Get pre-computed predictions
        reliability = float(reliability_scores[idx])
        deception = float(deception_scores[idx])
        behavior_probs_lower = behavior_probs_list[idx]
        
        # Convert to lowercase keys format expected by this function
        behavior_probs_ml = {k.lower(): v for k, v in behavior_probs_lower.items()}
        
        # Get recourse rules
        recourse = source.get("recourse_rules", {})
        rel_disengage = recourse.get("rel_disengage", 0.35)
        rel_flag = recourse.get("rel_ci_flag", 0.50)
        dec_disengage = recourse.get("dec_disengage", 0.75)
        dec_flag = recourse.get("dec_ci_flag", 0.60)
        
        # Decision logic
        if deception >= dec_disengage or reliability < rel_disengage:
            action = "disengage"
            task = None
        elif deception >= dec_flag:
            action = "flag_for_ci"
            task = rng.choice(TASK_ROSTER)
        elif reliability < rel_flag:
            action = "flag_and_task"
            task = rng.choice(TASK_ROSTER)
        else:
            action = "task"
            task = rng.choice(TASK_ROSTER)
        
        # ML-TSSP REQUIRED: Use ML-predicted behavior probabilities (from XGBoost Classifier)
        if behavior_probs_ml is None:
            raise RuntimeError(
                f"ML-TSSP behavior probability prediction failed for source {source.get('source_id', 'UNKNOWN')}. "
                "The system requires ML models to operate."
            )
        behavior_probs = behavior_probs_ml

        # Calculate intrinsic risk (before recourse/action adjustment)
        intrinsic_risk = sum(
            float(prob) * BEHAVIOR_RISK_MAP.get(behavior, 0.5)
            for behavior, prob in behavior_probs.items()
        )
        intrinsic_risk = float(np.clip(intrinsic_risk, 0.0, 1.0))
        
        # Calculate expected risk (post-recourse, adjusted by action)
        expected_risk = intrinsic_risk * ACTION_RISK_MULTIPLIER.get(action, 1.0)
        expected_risk = float(np.clip(expected_risk, 0.0, 1.0))
        score = reliability * (1 - deception) * (1 - expected_risk)
        
        # Determine risk bucket from intrinsic risk
        if intrinsic_risk < 0.3:
            risk_bucket = "low"
        elif intrinsic_risk > 0.6:
            risk_bucket = "high"
        else:
            risk_bucket = "medium"
        
        policy_item = {
            "source_id": source.get("source_id"),
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

        # Deterministic: Stage 1 only, fixed moderate risk, no ML, no Stage 2
        det_item = policy_item.copy()
        det_item["intrinsic_risk"] = 0.5  # Fixed moderate intrinsic risk for baseline
        det_item["expected_risk"] = 0.5
        det_item["risk_bucket"] = "medium"  # Fixed medium risk bucket
        det_item["score"] = reliability * (1 - deception) * (1 - det_item["expected_risk"])
        policies["deterministic"].append(det_item)

        # Uniform: Stage 1 only, equal allocation, no ML, no Stage 2
        if behavior_probs:
            uniform_risk = sum(
                BEHAVIOR_RISK_MAP.get(behavior, 0.5)
                for behavior in behavior_probs.keys()
            ) / len(behavior_probs)
        else:
            uniform_risk = 0.5
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
        "_using_ml_models": ml_models_used,
        "_using_fallback": not ml_models_used
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
