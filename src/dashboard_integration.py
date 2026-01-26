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

from src.utils.config import MODELS_DIR, PROJECT_ROOT, REGRESSION_FEATURES_FILE, CLASSIFICATION_FEATURES_FILE
from src.ml import ClassificationModelTrainer, RegressionModelTrainer
from src.data.preprocessing import load_features_from_file


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
        self.regression_features = None  # Will be loaded when models are loaded
        self.classification_features = None  # Will be loaded when models are loaded
        
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

            # Load classification features list for XGBoost classifier
            try:
                self.classification_features = load_features_from_file(CLASSIFICATION_FEATURES_FILE)
                print(f"Loaded {len(self.classification_features)} classification features: {self.classification_features}")
            except Exception as e:
                print(f"Warning: Could not load classification features file: {e}")
                # Fallback to default features
                self.classification_features = [
                    "task_success_rate", "corroboration_score", "report_timeliness",
                    "handler_confidence", "deception_score", "ci_flag", "report_accuracy",
                    "report_frequency", "access_level", "information_value", "handling_cost_kes",
                    "threat_relevant_features", "reliability_score", "scenario_probability"
                ]
            
            # Load regression features list for GRU models
            try:
                self.regression_features = load_features_from_file(REGRESSION_FEATURES_FILE)
                print(f"Loaded {len(self.regression_features)} regression features: {self.regression_features}")
            except Exception as e:
                print(f"Warning: Could not load regression features file: {e}")
                # Fallback to default features
                self.regression_features = [
                    "task_success_rate", "corroboration_score", "report_timeliness",
                    "handler_confidence", "ci_flag", "report_accuracy", "report_frequency",
                    "access_level", "information_value", "handling_cost_kes",
                    "threat_relevant_features", "scenario_probability"
                ]
            
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
            Dictionary of feature values (can use short names like 'tsr' or full names)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping behavior classes to probabilities
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Map short feature names to full names
        feature_mapping = {
            'tsr': 'task_success_rate',
            'cor': 'corroboration_score',
            'time': 'report_timeliness',
            'handler': 'handler_confidence',
            'dec_score': 'deception_score',
            'ci': 'ci_flag'
        }
        
        # Build feature array with all required classification features
        if self.classification_features is None:
            raise RuntimeError("Classification features not loaded. Call load_models() first.")
        
        feature_array = []
        for feat_name in self.classification_features:
            # Try full name first, then short name, then default value
            value = features.get(feat_name)
            if value is None:
                # Try short name mapping
                short_name = next((k for k, v in feature_mapping.items() if v == feat_name), None)
                if short_name:
                    value = features.get(short_name)
            if value is None:
                # Use default values for missing features
                if feat_name in ['report_accuracy', 'report_frequency', 'access_level', 
                                'information_value', 'handling_cost_kes', 'threat_relevant_features', 
                                'reliability_score', 'scenario_probability']:
                    # Default values based on typical ranges
                    if feat_name == 'report_accuracy':
                        value = features.get('tsr', features.get('task_success_rate', 0.5))  # Use TSR as proxy
                    elif feat_name == 'report_frequency':
                        value = 0.5  # Default moderate frequency
                    elif feat_name == 'access_level':
                        value = 0.5  # Default moderate access
                    elif feat_name == 'information_value':
                        value = 0.5  # Default moderate value
                    elif feat_name == 'handling_cost_kes':
                        value = 5000.0  # Default cost in KES
                    elif feat_name == 'threat_relevant_features':
                        value = 0.5  # Default moderate threat relevance
                    elif feat_name == 'reliability_score':
                        # Try to get from features if available, otherwise use TSR as proxy
                        value = features.get('reliability', features.get('tsr', features.get('task_success_rate', 0.5)))
                    elif feat_name == 'scenario_probability':
                        value = 0.5  # Default moderate probability
                    else:
                        value = 0.0
                else:
                    value = 0.0
            feature_array.append(float(value))
        
        feature_array = np.array([feature_array])
        
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
            Dictionary of feature values (can use short names like 'tsr' or full names)
            
        Returns:
        --------
        float
            Predicted reliability score (0-1)
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Map short feature names to full names
        feature_mapping = {
            'tsr': 'task_success_rate',
            'cor': 'corroboration_score',
            'time': 'report_timeliness',
            'handler': 'handler_confidence',
            'dec_score': 'deception_score',
            'ci': 'ci_flag'
        }
        
        # Build feature array with all required regression features
        if self.regression_features is None:
            raise RuntimeError("Regression features not loaded. Call load_models() first.")
        
        feature_array = []
        for feat_name in self.regression_features:
            # Try full name first, then short name, then default value
            value = features.get(feat_name)
            if value is None:
                # Try short name mapping
                short_name = next((k for k, v in feature_mapping.items() if v == feat_name), None)
                if short_name:
                    value = features.get(short_name)
            if value is None:
                # Use default values for missing features
                if feat_name in ['report_accuracy', 'report_frequency', 'access_level', 
                                'information_value', 'handling_cost_kes', 'threat_relevant_features', 
                                'scenario_probability']:
                    # Default values based on typical ranges
                    if feat_name == 'report_accuracy':
                        value = features.get('tsr', features.get('task_success_rate', 0.5))  # Use TSR as proxy
                    elif feat_name == 'report_frequency':
                        value = 0.5  # Default moderate frequency
                    elif feat_name == 'access_level':
                        value = 0.5  # Default moderate access
                    elif feat_name == 'information_value':
                        value = 0.5  # Default moderate value
                    elif feat_name == 'handling_cost_kes':
                        value = 5000.0  # Default cost in KES
                    elif feat_name == 'threat_relevant_features':
                        value = 0.5  # Default moderate threat relevance
                    elif feat_name == 'scenario_probability':
                        value = 0.5  # Default moderate probability
                    else:
                        value = 0.0
                else:
                    value = 0.0
            feature_array.append(float(value))
        
        feature_array = np.array([feature_array])
        
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
            Dictionary of feature values (can use short names like 'tsr' or full names)
            
        Returns:
        --------
        float
            Predicted deception score (0-1)
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Map short feature names to full names
        feature_mapping = {
            'tsr': 'task_success_rate',
            'cor': 'corroboration_score',
            'time': 'report_timeliness',
            'handler': 'handler_confidence',
            'dec_score': 'deception_score',
            'ci': 'ci_flag'
        }
        
        # Build feature array with all required regression features
        if self.regression_features is None:
            raise RuntimeError("Regression features not loaded. Call load_models() first.")
        
        feature_array = []
        for feat_name in self.regression_features:
            # Try full name first, then short name, then default value
            value = features.get(feat_name)
            if value is None:
                # Try short name mapping
                short_name = next((k for k, v in feature_mapping.items() if v == feat_name), None)
                if short_name:
                    value = features.get(short_name)
            if value is None:
                # Use default values for missing features
                if feat_name in ['report_accuracy', 'report_frequency', 'access_level', 
                                'information_value', 'handling_cost_kes', 'threat_relevant_features', 
                                'scenario_probability']:
                    # Default values based on typical ranges
                    if feat_name == 'report_accuracy':
                        value = features.get('tsr', features.get('task_success_rate', 0.5))  # Use TSR as proxy
                    elif feat_name == 'report_frequency':
                        value = 0.5  # Default moderate frequency
                    elif feat_name == 'access_level':
                        value = 0.5  # Default moderate access
                    elif feat_name == 'information_value':
                        value = 0.5  # Default moderate value
                    elif feat_name == 'handling_cost_kes':
                        value = 5000.0  # Default cost in KES
                    elif feat_name == 'threat_relevant_features':
                        value = 0.5  # Default moderate threat relevance
                    elif feat_name == 'scenario_probability':
                        value = 0.5  # Default moderate probability
                    else:
                        value = 0.0
                else:
                    value = 0.0
            feature_array.append(float(value))
        
        feature_array = np.array([feature_array])
        
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
    
    def predict_batch_reliability_scores(self, features_list: List[Dict]) -> np.ndarray:
        """
        Predict reliability scores for multiple sources in batch (much faster).
        
        Parameters:
        -----------
        features_list : List[Dict]
            List of feature dictionaries, one per source
            
        Returns:
        --------
        np.ndarray
            Array of predicted reliability scores
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Map short feature names to full names
        feature_mapping = {
            'tsr': 'task_success_rate',
            'cor': 'corroboration_score',
            'time': 'report_timeliness',
            'handler': 'handler_confidence',
            'dec_score': 'deception_score',
            'ci': 'ci_flag'
        }
        
        if self.regression_features is None:
            raise RuntimeError("Regression features not loaded. Call load_models() first.")
        
        # Build batch feature array
        batch_features = []
        for features in features_list:
            feature_array = []
            for feat_name in self.regression_features:
                value = features.get(feat_name)
                if value is None:
                    short_name = next((k for k, v in feature_mapping.items() if v == feat_name), None)
                    if short_name:
                        value = features.get(short_name)
                if value is None:
                    # Use same defaults as single prediction
                    if feat_name == 'report_accuracy':
                        value = features.get('tsr', features.get('task_success_rate', 0.5))
                    elif feat_name == 'report_frequency':
                        value = 0.5
                    elif feat_name == 'access_level':
                        value = 0.5
                    elif feat_name == 'information_value':
                        value = 0.5
                    elif feat_name == 'handling_cost_kes':
                        value = 5000.0
                    elif feat_name == 'threat_relevant_features':
                        value = 0.5
                    elif feat_name == 'scenario_probability':
                        value = 0.5
                    else:
                        value = 0.0
                feature_array.append(float(value))
            batch_features.append(feature_array)
        
        batch_array = np.array(batch_features)
        
        # Batch prediction with GRU model
        if getattr(self.reliability_trainer, "best_model_name", "") == "gru" and self.reliability_scaler is not None:
            scaled_features = self.reliability_scaler.transform(batch_array)
            scaled_features = scaled_features.reshape(scaled_features.shape[0], 1, scaled_features.shape[1])
            predictions = self.reliability_trainer.best_model.predict(scaled_features, verbose=0).flatten()
        else:
            predictions = self.reliability_trainer.best_model.predict(batch_array)
        
        return np.clip(predictions, 0.0, 1.0)
    
    def predict_batch_deception_scores(self, features_list: List[Dict]) -> np.ndarray:
        """
        Predict deception scores for multiple sources in batch (much faster).
        
        Parameters:
        -----------
        features_list : List[Dict]
            List of feature dictionaries, one per source
            
        Returns:
        --------
        np.ndarray
            Array of predicted deception scores
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Map short feature names to full names
        feature_mapping = {
            'tsr': 'task_success_rate',
            'cor': 'corroboration_score',
            'time': 'report_timeliness',
            'handler': 'handler_confidence',
            'dec_score': 'deception_score',
            'ci': 'ci_flag'
        }
        
        if self.regression_features is None:
            raise RuntimeError("Regression features not loaded. Call load_models() first.")
        
        # Build batch feature array
        batch_features = []
        for features in features_list:
            feature_array = []
            for feat_name in self.regression_features:
                value = features.get(feat_name)
                if value is None:
                    short_name = next((k for k, v in feature_mapping.items() if v == feat_name), None)
                    if short_name:
                        value = features.get(short_name)
                if value is None:
                    # Use same defaults as single prediction
                    if feat_name == 'report_accuracy':
                        value = features.get('tsr', features.get('task_success_rate', 0.5))
                    elif feat_name == 'report_frequency':
                        value = 0.5
                    elif feat_name == 'access_level':
                        value = 0.5
                    elif feat_name == 'information_value':
                        value = 0.5
                    elif feat_name == 'handling_cost_kes':
                        value = 5000.0
                    elif feat_name == 'threat_relevant_features':
                        value = 0.5
                    elif feat_name == 'scenario_probability':
                        value = 0.5
                    else:
                        value = 0.0
                feature_array.append(float(value))
            batch_features.append(feature_array)
        
        batch_array = np.array(batch_features)
        
        # Batch prediction with GRU model
        if getattr(self.deception_trainer, "best_model_name", "") == "gru" and self.deception_scaler is not None:
            scaled_features = self.deception_scaler.transform(batch_array)
            scaled_features = scaled_features.reshape(scaled_features.shape[0], 1, scaled_features.shape[1])
            predictions = self.deception_trainer.best_model.predict(scaled_features, verbose=0).flatten()
        else:
            predictions = self.deception_trainer.best_model.predict(batch_array)
        
        return np.clip(predictions, 0.0, 1.0)
    
    def predict_batch_behavior_probabilities(self, features_list: List[Dict]) -> List[Dict[str, float]]:
        """
        Predict behavior probabilities for multiple sources in batch (much faster).
        
        Parameters:
        -----------
        features_list : List[Dict]
            List of feature dictionaries, one per source
            
        Returns:
        --------
        List[Dict[str, float]]
            List of behavior probability dictionaries, one per source
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Map short feature names to full names
        feature_mapping = {
            'tsr': 'task_success_rate',
            'cor': 'corroboration_score',
            'time': 'report_timeliness',
            'handler': 'handler_confidence',
            'dec_score': 'deception_score',
            'ci': 'ci_flag'
        }
        
        if self.classification_features is None:
            raise RuntimeError("Classification features not loaded. Call load_models() first.")
        
        # Build batch feature array
        batch_features = []
        for features in features_list:
            feature_array = []
            for feat_name in self.classification_features:
                value = features.get(feat_name)
                if value is None:
                    short_name = next((k for k, v in feature_mapping.items() if v == feat_name), None)
                    if short_name:
                        value = features.get(short_name)
                if value is None:
                    # Use same defaults as single prediction
                    if feat_name == 'report_accuracy':
                        value = features.get('tsr', features.get('task_success_rate', 0.5))
                    elif feat_name == 'report_frequency':
                        value = 0.5
                    elif feat_name == 'access_level':
                        value = 0.5
                    elif feat_name == 'information_value':
                        value = 0.5
                    elif feat_name == 'handling_cost_kes':
                        value = 5000.0
                    elif feat_name == 'threat_relevant_features':
                        value = 0.5
                    elif feat_name == 'reliability_score':
                        value = features.get('reliability', features.get('tsr', features.get('task_success_rate', 0.5)))
                    elif feat_name == 'scenario_probability':
                        value = 0.5
                    else:
                        value = 0.0
                feature_array.append(float(value))
            batch_features.append(feature_array)
        
        batch_array = np.array(batch_features)
        
        # Batch prediction with XGBoost
        probabilities = self.classification_trainer.best_model.predict_proba(batch_array)
        classes = self.label_encoder.classes_
        
        # Convert to list of dictionaries
        results = []
        for prob_row in probabilities:
            result = {cls.lower(): float(prob) for cls, prob in zip(classes, prob_row)}
            results.append(result)
        
        return results


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
