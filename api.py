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
    
    # Track if ML models were successfully used (for at least one source)
    ml_models_used = False
    
    for source in sources:
        features = source.get("features", {})
        tsr = features.get("task_success_rate", 0.5)
        cor = features.get("corroboration_score", 0.5)
        time = features.get("report_timeliness", 0.5)
        handler = features.get("handler_confidence", 0.5)
        dec_score = features.get("deception_score", 0.3)
        ci = features.get("ci_flag", 0)
        
        # Prepare features dict for ML pipeline
        features_dict = {
            "tsr": tsr,
            "cor": cor,
            "time": time,
            "handler": handler,
            "dec_score": dec_score,
            "ci": ci
        }
        
        # PRIMARY METHOD: Use ML-TSSP models (XGBoost + GRU) for all predictions
        reliability = None
        deception = None
        behavior_probs_ml = None
        ml_prediction_failed = False
        
        if use_ml_models:
            try:
                # PRIMARY: Use GRU Regressor for reliability score (ML-TSSP)
                reliability = ml_pipeline.predict_reliability_score(features_dict)
                
                # PRIMARY: Use GRU Regressor for deception score (ML-TSSP)
                deception = ml_pipeline.predict_deception_score(features_dict)
                
                # PRIMARY: Use XGBoost Classifier for behavior probabilities (ML-TSSP)
                behavior_probs_lower = ml_pipeline.predict_behavior_probabilities(features_dict)
                
                # Convert to lowercase keys format expected by this function
                behavior_probs_ml = {k.lower(): v for k, v in behavior_probs_lower.items()}
                
                # Mark that ML models were successfully used (PRIMARY METHOD)
                ml_models_used = True
                
            except Exception:
                # ML prediction failed - will fall back to formulas (NOT RECOMMENDED)
                ml_prediction_failed = True
                reliability = None
                deception = None
                behavior_probs_ml = None
        
        # FALLBACK ONLY: Use formula-based calculations ONLY if ML not available or failed
        # This is a safety fallback, not the primary method
        if reliability is None or deception is None:
            # Formula-based reliability calculation (fallback)
            reliability = np.clip(
                0.30 * tsr + 0.25 * cor + 0.20 * time + 0.15 * handler
                - 0.15 * dec_score - 0.10 * ci + 0.05 * rng.normal(0, 0.03),
                0.0, 1.0
            )
            
            # Formula-based deception calculation (fallback)
            deception = np.clip(
                0.30 * dec_score + 0.25 * ci + 0.20 * (1 - cor) + 0.15 * (1 - handler)
                + 0.10 * rng.beta(2, 5),
                0.0, 1.0
            )
        
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
        
        # PRIMARY: Use ML behavior probabilities (XGBoost), FALLBACK: formulas only if ML unavailable
        if behavior_probs_ml is not None:
            # PRIMARY METHOD: Use ML-predicted behavior probabilities (from XGBoost Classifier)
            behavior_probs = behavior_probs_ml
        else:
            # FALLBACK ONLY: Formula-based behavior probabilities (used only when ML unavailable)
            cooperative_prob = max(0.0, min(1.0, reliability * (1 - deception) * 1.2))
            uncertain_prob = max(0.0, min(1.0, (1 - reliability) * 0.4))
            coerced_prob = max(0.0, min(1.0, deception * 0.3))
            deceptive_prob = max(0.0, min(1.0, deception * 0.5))

            total = cooperative_prob + uncertain_prob + coerced_prob + deceptive_prob
            if total > 0:
                behavior_probs = {
                    "cooperative": cooperative_prob / total,
                    "uncertain": uncertain_prob / total,
                    "coerced": coerced_prob / total,
                    "deceptive": deceptive_prob / total
                }
            else:
                behavior_probs = {
                    "cooperative": 0.25,
                    "uncertain": 0.25,
                    "coerced": 0.25,
                    "deceptive": 0.25
                }

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
