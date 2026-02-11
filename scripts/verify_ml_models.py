"""
Verify that ML models load and operate successfully.
Run from project root: python scripts/verify_ml_models.py
"""
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    from src.utils.config import MODELS_DIR

    # 1. Check required files exist
    required = [
        "classification_model.pkl",
        "classification_model_label_encoder.pkl",
        "reliability_scaler.pkl",
        "deception_scaler.pkl",
    ]
    optional_models = [
        "reliability_model.keras", "reliability_model.h5", "reliability_model.pkl",
        "deception_model.keras", "deception_model.h5", "deception_model.pkl",
    ]
    missing = []
    for f in required:
        if not (MODELS_DIR / f).exists():
            missing.append(f)
    has_reliability = any((MODELS_DIR / f).exists() for f in ["reliability_model.keras", "reliability_model.h5", "reliability_model.pkl"])
    has_deception = any((MODELS_DIR / f).exists() for f in ["deception_model.keras", "deception_model.h5", "deception_model.pkl"])
    if not has_reliability:
        missing.append("reliability_model (.keras, .h5, or .pkl)")
    if not has_deception:
        missing.append("deception_model (.keras, .h5, or .pkl)")

    if missing:
        print("FAIL: Missing model files in models/:")
        for m in missing:
            print("  -", m)
        print("\nTrain models first (e.g. run the pipeline or main.py) or copy trained files into models/.")
        return 1

    # 2. Load pipeline and models
    print("Loading pipeline and models...")
    try:
        from src.dashboard_integration import get_dashboard_pipeline
        pipeline = get_dashboard_pipeline()
        loaded = pipeline.load_models()
    except Exception as e:
        print("FAIL: Error during load:", e)
        return 1

    if not loaded or not getattr(pipeline, "models_loaded", False):
        print("FAIL: load_models() returned False or models_loaded is False. Check warnings above.")
        return 1

    print("Models loaded successfully.")

    # 3. Quick prediction test
    test_features = {
        "task_success_rate": 0.7,
        "corroboration_score": 0.6,
        "report_timeliness": 0.8,
        "handler_confidence": 0.7,
        "deception_score": 0.2,
        "ci_flag": 0,
    }
    try:
        prob = pipeline.predict_behavior_probabilities(test_features)
        rel = pipeline.predict_reliability_score(test_features)
        dec = pipeline.predict_deception_score(test_features)
        print("Predictions OK: behavior probs keys =", list(prob.keys()), "reliability =", round(rel, 4), "deception =", round(dec, 4))
    except Exception as e:
        print("FAIL: Prediction error:", e)
        return 1

    print("\nOK: ML models load and operate successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
