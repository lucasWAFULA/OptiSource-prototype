"""
Dashboard integration module for ML-TSSP pipeline.

This module provides a lightweight interface for the Streamlit dashboard
to interact with the ML-TSSP pipeline, including model loading and prediction.
"""

from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import joblib

from src.utils.config import MODELS_DIR, PROJECT_ROOT
from src.ml import ClassificationModelTrainer, RegressionModelTrainer


class DashboardPipeline:
    """
    Lightweight pipeline wrapper for dashboard use.
    Loads models and provides prediction methods.
    """
    
    def __init__(self):
        """Initialize dashboard pipeline."""
        self.classification_trainer = None
        self.reliability_trainer = None
        self.deception_trainer = None
        self.label_encoder = None
        self.reliability_scaler = None
        self.deception_scaler = None
        self.models_loaded = False
        
    def load_models(self) -> bool:
        """
        Load all ML models from the models directory.
        
        Returns:
        --------
        bool
            True if all models loaded successfully, False otherwise
        """
        try:
            missing = []
            def _load_keras_or_h5(model_path_keras: Path, model_path_h5: Path, label: str):
                if model_path_keras.exists():
                    try:
                        self.reliability_trainer.load_model(model_path_keras) if label == "reliability" else self.deception_trainer.load_model(model_path_keras)
                        print(f"Loaded {label} model from {model_path_keras}")
                        return True
                    except Exception as e:
                        print(f"Warning: Could not load {label} .keras model: {e}")
                if model_path_h5.exists():
                    try:
                        self.reliability_trainer.load_model(model_path_h5) if label == "reliability" else self.deception_trainer.load_model(model_path_h5)
                        print(f"Loaded {label} model from {model_path_h5}")
                        return True
                    except Exception as e:
                        print(f"Warning: Could not load {label} .h5 model: {e}")
                return False
            # Load XGBoost classifier
            self.classification_trainer = ClassificationModelTrainer()
            classifier_path = MODELS_DIR / "classification_model.pkl"
            if not classifier_path.exists():
                missing.append(str(classifier_path))
                print(f"Warning: Classification model not found at {classifier_path}")
            else:
                self.classification_trainer.load_model(classifier_path)
                self.classification_trainer.best_model_name = 'xgboost'
            
            # Load label encoder
            encoder_path = MODELS_DIR / "classification_model_label_encoder.pkl"
            if not encoder_path.exists():
                missing.append(str(encoder_path))
                print(f"Warning: Label encoder not found at {encoder_path}")
            else:
                self.label_encoder = joblib.load(encoder_path)
            
            # Load reliability GRU model and scaler (prefer .keras, fallback .h5)
            self.reliability_trainer = RegressionModelTrainer()
            reliability_model_path = MODELS_DIR / "reliability_model.keras"
            reliability_h5_path = MODELS_DIR / "reliability_model.h5"
            if _load_keras_or_h5(reliability_model_path, reliability_h5_path, "reliability"):
                self.reliability_trainer.best_model_name = 'gru'
            else:
                print(f"Warning: Reliability model not found at {reliability_model_path} or {reliability_h5_path}")
            
            reliability_scaler_path = MODELS_DIR / "reliability_scaler.pkl"
            if not reliability_scaler_path.exists():
                missing.append(str(reliability_scaler_path))
                print(f"Warning: Reliability scaler not found at {reliability_scaler_path}")
            else:
                self.reliability_scaler = joblib.load(reliability_scaler_path)
            
            # Load deception GRU model and scaler (prefer .keras, fallback .h5)
            self.deception_trainer = RegressionModelTrainer()
            deception_model_path = MODELS_DIR / "deception_model.keras"
            deception_h5_path = MODELS_DIR / "deception_model.h5"
            if _load_keras_or_h5(deception_model_path, deception_h5_path, "deception"):
                self.deception_trainer.best_model_name = 'gru'
            else:
                print(f"Warning: Deception model not found at {deception_model_path} or {deception_h5_path}")
            
            deception_scaler_path = MODELS_DIR / "deception_scaler.pkl"
            if not deception_scaler_path.exists():
                missing.append(str(deception_scaler_path))
                print(f"Warning: Deception scaler not found at {deception_scaler_path}")
            else:
                self.deception_scaler = joblib.load(deception_scaler_path)

            # Fallback to pickle models if GRU .h5 failed or unavailable
            if self.reliability_trainer.best_model is None:
                reliability_pkl = MODELS_DIR / "reliability_model.pkl"
                if reliability_pkl.exists():
                    try:
                        self.reliability_trainer.best_model = joblib.load(reliability_pkl)
                        self.reliability_trainer.best_model_name = 'xgboost'
                        print(f"Loaded reliability model from {reliability_pkl}")
                    except Exception as e:
                        print(f"Warning: Could not load reliability .pkl model: {e}")
                else:
                    missing.append(str(reliability_pkl))

            if self.deception_trainer.best_model is None:
                deception_pkl = MODELS_DIR / "deception_model.pkl"
                if deception_pkl.exists():
                    try:
                        self.deception_trainer.best_model = joblib.load(deception_pkl)
                        self.deception_trainer.best_model_name = 'xgboost'
                        print(f"Loaded deception model from {deception_pkl}")
                    except Exception as e:
                        print(f"Warning: Could not load deception .pkl model: {e}")
                else:
                    missing.append(str(deception_pkl))

            if missing:
                print(f"Missing required files: {missing}")
                self.models_loaded = False
                return False
            
            self.models_loaded = True
            print("All ML models loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
            return False
    
    def predict_behavior_probabilities(self, features: Dict) -> Dict[str, float]:
        """
        Predict behavior class probabilities using XGBoost classifier.
        
        Parameters:
        -----------
        features : Dict
            Dictionary of feature values (e.g., {'tsr': 0.8, 'cor': 0.7, ...})
            
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping behavior classes to probabilities
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Convert features dict to array format expected by classifier
        # This assumes features are in the correct order
        # You may need to adjust based on your feature list
        feature_list = ['tsr', 'cor', 'time', 'handler', 'dec_score', 'ci']
        feature_array = np.array([[features.get(f, 0.0) for f in feature_list]])
        
        # Get predictions
        probabilities = self.classification_trainer.best_model.predict_proba(feature_array)[0]
        
        # Map to behavior classes using label encoder
        classes = self.label_encoder.classes_
        result = {cls.lower(): float(prob) for cls, prob in zip(classes, probabilities)}
        
        return result
    
    def predict_reliability_score(self, features: Dict) -> float:
        """
        Predict reliability score using GRU regressor.
        
        Parameters:
        -----------
        features : Dict
            Dictionary of feature values
            
        Returns:
        --------
        float
            Predicted reliability score (0-1)
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Convert features to array and scale
        feature_list = ['tsr', 'cor', 'time', 'handler', 'dec_score', 'ci']
        feature_array = np.array([[features.get(f, 0.0) for f in feature_list]])
        
        # If GRU model, scale + reshape
        if getattr(self.reliability_trainer, "best_model_name", "") == "gru" and self.reliability_scaler is not None:
            scaled_features = self.reliability_scaler.transform(feature_array)
            scaled_features = scaled_features.reshape(scaled_features.shape[0], 1, scaled_features.shape[1])
            prediction = self.reliability_trainer.best_model.predict(scaled_features, verbose=0)[0][0]
        else:
            # Fallback to non-GRU regressor (no scaling/reshape)
            prediction = self.reliability_trainer.best_model.predict(feature_array)[0]
        
        # Clip to [0, 1] range
        return float(np.clip(prediction, 0.0, 1.0))
    
    def predict_deception_score(self, features: Dict) -> float:
        """
        Predict deception score using GRU regressor.
        
        Parameters:
        -----------
        features : Dict
            Dictionary of feature values
            
        Returns:
        --------
        float
            Predicted deception score (0-1)
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Convert features to array and scale
        feature_list = ['tsr', 'cor', 'time', 'handler', 'dec_score', 'ci']
        feature_array = np.array([[features.get(f, 0.0) for f in feature_list]])
        
        # If GRU model, scale + reshape
        if getattr(self.deception_trainer, "best_model_name", "") == "gru" and self.deception_scaler is not None:
            scaled_features = self.deception_scaler.transform(feature_array)
            scaled_features = scaled_features.reshape(scaled_features.shape[0], 1, scaled_features.shape[1])
            prediction = self.deception_trainer.best_model.predict(scaled_features, verbose=0)[0][0]
        else:
            # Fallback to non-GRU regressor (no scaling/reshape)
            prediction = self.deception_trainer.best_model.predict(feature_array)[0]
        
        # Clip to [0, 1] range
        return float(np.clip(prediction, 0.0, 1.0))


def get_dashboard_pipeline() -> DashboardPipeline:
    """
    Get or create a dashboard pipeline instance.
    
    Returns:
    --------
    DashboardPipeline
        Initialized pipeline instance
    """
    pipeline = DashboardPipeline()
    return pipeline
