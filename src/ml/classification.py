"""
Classification models for HUMINT source behavior prediction.

This module implements various classification models including XGBoost,
GRU, and baseline models for predicting source behavior classes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. GRU models will be disabled.")


class ClassificationModelTrainer:
    """Trainer for classification models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.label_encoder = None
        self.scaler = None
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
        Train XGBoost classifier.
        
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
                'eval_metric': 'mlogloss',
            }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        self.models['xgboost'] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        return self.models['xgboost']
    
    def train_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Dict]:
        """
        Train baseline classification models.
        
        Returns:
        --------
        Dictionary of model results
        """
        baseline_models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            'svm': SVC(probability=True, random_state=self.random_state),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state)
        }
        
        results = {}
        for name, model in baseline_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            self.models[name] = results[name]
        
        return results
    
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
        Train GRU model for classification.
        
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
        num_classes = len(np.unique(y_train))
        
        X_train_reshaped = X_train.values.reshape(X_train.shape[0], 1, num_features)
        X_test_reshaped = X_test.values.reshape(X_test.shape[0], 1, num_features)
        
        # Build model
        input_layer = Input(shape=(1, num_features))
        gru_layer = GRU(units=hidden_units, activation='relu')(input_layer)
        dropout_layer = Dropout(dropout)(gru_layer)
        dense_layer = Dense(32, activation='relu')(dropout_layer)
        output_layer = Dense(num_classes, activation='softmax')(dense_layer)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
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
        y_pred_proba = model.predict(X_test_reshaped, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        self.models['gru'] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'history': history.history
        }
        
        return self.models['gru']
    
    def apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE for class imbalance.
        
        Returns:
        --------
        Resampled X_train, y_train
        """
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
        y_resampled = pd.Series(y_resampled)
        
        return X_resampled, y_resampled
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        if y_pred_proba is not None:
            from sklearn.metrics import roc_auc_score
            try:
                if len(np.unique(y_true)) > 2:
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovr', average='weighted'
                    )
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    def get_best_model(self) -> Tuple[Any, str]:
        """
        Get the best performing model based on F1 score.
        
        Returns:
        --------
        Tuple of (model, model_name)
        """
        if not self.models:
            raise ValueError("No models trained yet")
        
        best_f1 = -1
        best_name = None
        
        for name, result in self.models.items():
            f1 = result['metrics']['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_name = name
        
        self.best_model = self.models[best_name]['model']
        self.best_model_name = best_name
        
        return self.best_model, best_name
    
    def save_model(self, model_path: Path, label_encoder: Optional[LabelEncoder] = None):
        """
        Save the best model and label encoder.
        
        Parameters:
        -----------
        model_path : Path to save model
        label_encoder : Label encoder to save
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
        
        # Save label encoder if provided
        if label_encoder is not None:
            encoder_path = model_path.parent / f"{model_path.stem}_label_encoder.pkl"
            joblib.dump(label_encoder, encoder_path)
            print(f"Label encoder saved to: {encoder_path}")
        
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
