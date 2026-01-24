"""
Regression models for HUMINT source performance metrics.

This module implements regression models including XGBoost, GRU,
and baseline models for predicting reliability and deception scores.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBRegressor

try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. GRU models will be disabled.")


class RegressionModelTrainer:
    """Trainer for regression models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Train XGBoost regressor.
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data
        params : Optional hyperparameters
        
        Returns:
        --------
        Dictionary with model, metrics, and predictions
        """
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
            }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        
        self.models['xgboost'] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred
        }
        
        return self.models['xgboost']
    
    def train_linear_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Train linear regression baseline.
        
        Returns:
        --------
        Dictionary with model, metrics, and predictions
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        
        self.models['linear'] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred
        }
        
        return self.models['linear']
    
    def train_gru(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        epochs: int = 50,
        batch_size: int = 32,
        hidden_units: int = 64,
        dropout: float = 0.3,
        patience: int = 10
    ) -> Dict[str, Any]:
        """
        Train GRU model for regression.
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data
        epochs : Number of training epochs
        batch_size : Batch size
        hidden_units : Number of GRU hidden units
        dropout : Dropout rate
        patience : Early stopping patience
        
        Returns:
        --------
        Dictionary with model, metrics, and predictions
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for GRU models")
        
        # Reshape data for GRU (samples, timesteps, features)
        num_features = X_train.shape[1]
        
        X_train_reshaped = X_train.values.reshape(X_train.shape[0], 1, num_features)
        X_test_reshaped = X_test.values.reshape(X_test.shape[0], 1, num_features)
        
        # Build model
        input_layer = Input(shape=(1, num_features))
        gru_layer = GRU(units=hidden_units, activation='relu')(input_layer)
        dropout_layer = Dropout(dropout)(gru_layer)
        dense_layer = Dense(32, activation='relu')(dropout_layer)
        output_layer = Dense(1, activation='linear')(dense_layer)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_reshaped, y_train, test_size=0.1, random_state=self.random_state
        )
        
        history = model.fit(
            X_train_split, y_train_split,
            validation_data=(X_val_split, y_val_split),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Predictions
        y_pred = model.predict(X_test_reshaped, verbose=0).flatten()
        
        metrics = self._calculate_metrics(y_test, y_pred)
        
        self.models['gru'] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'history': history.history
        }
        
        return self.models['gru']
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
        }
    
    def get_best_model(self) -> Tuple[Any, str]:
        """
        Get the best performing model based on R² score.
        
        Returns:
        --------
        Tuple of (model, model_name)
        """
        if not self.models:
            raise ValueError("No models trained yet")
        
        best_r2 = -np.inf
        best_name = None
        
        for name, result in self.models.items():
            r2 = result['metrics']['r2']
            if r2 > best_r2:
                best_r2 = r2
                best_name = name
        
        self.best_model = self.models[best_name]['model']
        self.best_model_name = best_name
        
        return self.best_model, best_name
    
    def save_model(self, model_path: Path):
        """
        Save the best model.
        
        Parameters:
        -----------
        model_path : Path to save model
        """
        if self.best_model is None:
            self.get_best_model()
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.best_model_name == 'gru':
            self.best_model.save(str(model_path))
        else:
            joblib.dump(self.best_model, model_path)
        
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: Path):
        """
        Load a saved model.
        
        Parameters:
        -----------
        model_path : Path to saved model
        """
        model_path = Path(model_path)
        
        if model_path.suffix in ['.h5', '.keras'] or model_path.is_dir():
            if not TENSORFLOW_AVAILABLE:
                raise ImportError("TensorFlow required to load GRU model")
            from tensorflow.keras.models import load_model
            self.best_model = load_model(str(model_path), compile=False)
            self.best_model_name = 'gru'
        else:
            self.best_model = joblib.load(model_path)
            self.best_model_name = 'xgboost'  # Default assumption
        
        print(f"Model loaded from: {model_path}")
