# Model Selection and Integration

## Best Performing Models

Based on performance evaluation, the following models are used in the ML-TSSP pipeline:

### Classification: XGBoost Classifier
- **Purpose**: Predict behavior class probabilities
- **Performance**: Highest F1-score and accuracy among tested models
- **Output**: Probability distribution over behavior classes (cooperative, deceptive, coerced, uncertain)
- **Usage in TSSP**: Behavior probabilities feed directly into Stage 2 expected recourse cost calculation

### Regression: GRU (Gated Recurrent Unit)
- **Purpose**: Predict reliability and deception scores
- **Performance**: Superior R² scores compared to XGBoost for both targets
  - Reliability Score: R² > 0.97
  - Deception Score: R² > 0.77
- **Output**: Continuous scores for reliability and deception
- **Usage in TSSP**: 
  - Reliability scores inform Stage 1 cost calculations
  - Both scores contribute to information value calculations

## Model Integration Flow

```
┌─────────────────────────────────────────┐
│  Source Features (Dataset)              │
└──────────────┬──────────────────────────┘
               │
               ├──────────────────────────┐
               │                          │
               ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│  XGBoost Classifier      │  │  GRU Regressor           │
│  (Behavior Prediction)   │  │  (Reliability/Deception) │
└──────────────┬───────────┘  └──────────────┬───────────┘
               │                             │
               │                             │
               ▼                             ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│  Behavior Probabilities  │  │  Reliability Score       │
│  P(b|s) for each source  │  │  Deception Score         │
└──────────────┬───────────┘  └──────────────┬───────────┘
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  TSSP Optimization   │
                   │  - Stage 1 Costs     │
                   │  - Stage 2 Recourse  │
                   └──────────────────────┘
```

## Implementation Details

### Classification Model (XGBoost)

**Training**:
- Uses SMOTE for class imbalance handling
- Trained on all classification features from `training_features_nc.txt`
- Outputs probability distribution over 4 behavior classes

**Integration**:
```python
# Behavior probabilities feed into TSSP Stage 2
behavior_probabilities[(source_id, behavior_class)] = xgb_model.predict_proba(X)[class_idx]
```

### Regression Models (GRU)

**Training**:
- Separate GRU models for reliability and deception
- Features scaled using StandardScaler before training
- Input shape: (samples, 1, features) for GRU compatibility
- Early stopping to prevent overfitting

**Integration**:
```python
# Reliability predictions used in Stage 1 costs
reliability = gru_reliability_model.predict(X_scaled)

# Both scores used in information value calculation
info_value = (predicted_reliability + information_value) / 2
```

## Why These Models?

### XGBoost for Classification
- **Interpretability**: Feature importance analysis
- **Performance**: Consistently highest F1-scores
- **Efficiency**: Fast training and prediction
- **Robustness**: Handles class imbalance well with SMOTE

### GRU for Regression
- **Performance**: Significantly better R² scores
  - Reliability: GRU R² = 0.974 vs XGBoost R² = 0.962
  - Deception: GRU R² = 0.775 vs XGBoost R² = 0.734
- **Sequence Modeling**: Can capture temporal dependencies in features
- **Deep Learning**: Better at learning complex non-linear relationships

## Model Persistence

Models are saved in `models/` directory:
- `classification_model.pkl`: XGBoost classifier
- `classification_model_label_encoder.pkl`: Label encoder for classes
- `reliability_model.h5`: GRU model for reliability
- `reliability_scaler.pkl`: Scaler for reliability features
- `deception_model.h5`: GRU model for deception
- `deception_scaler.pkl`: Scaler for deception features

## Usage in Pipeline

The pipeline automatically:
1. Trains XGBoost classifier for behavior prediction
2. Trains GRU models for reliability and deception
3. Uses predictions (not raw data) in TSSP optimization
4. Ensures all TSSP inputs come from ML model outputs

This ensures the optimization model uses the best available predictions rather than ground truth or raw data values.
