"""
Preprocessing utilities for ML-TSSP training pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


DEFAULT_CLASS_FEATURES = [
    "task_success_rate",
    "corroboration_score",
    "report_timeliness",
    "handler_confidence",
    "deception_score",
    "ci_flag",
]

DEFAULT_REG_FEATURES = [
    "task_success_rate",
    "corroboration_score",
    "report_timeliness",
    "handler_confidence",
    "ci_flag",
    "information_value",
    "report_accuracy",
    "report_frequency",
    "access_level",
]


def load_features_from_file(feature_file: str | Path) -> List[str]:
    """
    Load a feature list from a text file (one feature per line).
    Falls back to defaults if file is missing or empty.
    """
    try:
        path = Path(feature_file)
        if not path.exists():
            return DEFAULT_CLASS_FEATURES
        raw = path.read_text(encoding="utf-8").splitlines()
        features = [line.strip() for line in raw if line.strip() and not line.startswith("#")]
        return features if features else DEFAULT_CLASS_FEATURES
    except Exception:
        return DEFAULT_CLASS_FEATURES


def prepare_classification_data(
    df: pd.DataFrame,
    feature_file: Optional[str | Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Prepare classification dataset for behavior prediction.
    """
    features = load_features_from_file(feature_file) if feature_file else DEFAULT_CLASS_FEATURES
    features = [f for f in features if f in df.columns]
    if not features:
        raise ValueError("No valid classification features found in dataset.")

    X = df[features].copy()
    y = df["behavior_class"].astype(str).values

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    return X_train, y_train, X_test, y_test, encoder


def prepare_regression_data(
    df: pd.DataFrame,
    feature_file: Optional[str | Path] = None,
    target_col: str = "reliability_score",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Prepare regression dataset for reliability/deception prediction.
    """
    features = load_features_from_file(feature_file) if feature_file else DEFAULT_REG_FEATURES
    features = [f for f in features if f in df.columns]
    if not features:
        raise ValueError("No valid regression features found in dataset.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df[features].copy()
    y = df[target_col].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, y_train, X_test, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features for GRU models.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
"""
Data preprocessing utilities for HUMINT source performance dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_features_from_file(feature_file: Path) -> List[str]:
    """
    Load feature names from a text file.
    
    Parameters:
    -----------
    feature_file : Path
        Path to the feature file (one feature per line)
    
    Returns:
    --------
    List[str]
        List of feature names
    """
    with open(feature_file, 'r') as f:
        features = [line.strip() for line in f if line.strip()]
    return features


def prepare_classification_data(
    df: pd.DataFrame,
    feature_file: Optional[Path] = None,
    target_col: str = 'behavior_class',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare data for classification task.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_file : Optional[Path]
        Path to file containing feature names. If None, uses default features.
    target_col : str
        Name of target column (default: 'behavior_class')
    test_size : float
        Proportion of data for testing (default: 0.2)
    random_state : int
        Random seed (default: 42)
    
    Returns:
    --------
    Tuple containing X_train, y_train, X_test, y_test
    """
    if feature_file:
        features = load_features_from_file(feature_file)
    else:
        # Default classification features
        features = [
            'task_success_rate', 'corroboration_score', 'report_timeliness',
            'handler_confidence', 'deception_score', 'ci_flag',
            'report_accuracy', 'report_frequency', 'access_level',
            'information_value', 'handling_cost_kes', 'threat_relevant_features',
            'reliability_score', 'scenario_probability'
        ]
    
    # Ensure all features exist
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        print(f"Warning: Missing features: {missing}")
    
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y), index=y.index)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    return X_train, y_train, X_test, y_test, label_encoder


def prepare_regression_data(
    df: pd.DataFrame,
    feature_file: Optional[Path] = None,
    target_col: str = 'reliability_score',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare data for regression task.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_file : Optional[Path]
        Path to file containing feature names. If None, uses default features.
    target_col : str
        Name of target column (default: 'reliability_score')
    test_size : float
        Proportion of data for testing (default: 0.2)
    random_state : int
        Random seed (default: 42)
    
    Returns:
    --------
    Tuple containing X_train, y_train, X_test, y_test
    """
    if feature_file:
        features = load_features_from_file(feature_file)
    else:
        # Default regression features (excluding target)
        features = [
            'task_success_rate', 'corroboration_score', 'report_timeliness',
            'handler_confidence', 'ci_flag', 'report_accuracy',
            'report_frequency', 'access_level', 'information_value',
            'handling_cost_kes', 'threat_relevant_features', 'scenario_probability'
        ]
    
    # Remove target from features if present
    features = [f for f in features if f != target_col]
    
    # Ensure all features exist
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        print(f"Warning: Missing features: {missing}")
    
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, y_train, X_test, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    scaler : Optional[StandardScaler]
        Pre-fitted scaler. If None, fits a new one.
    
    Returns:
    --------
    Tuple containing scaled X_train, scaled X_test, and the scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
    else:
        X_train_scaled = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, scaler
