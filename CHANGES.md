# Implementation Updates

## Model Selection Changes

### Classification Model
- **Before**: Trained multiple models (XGBoost, Random Forest, SVM, KNN, Decision Tree) and selected best
- **After**: Uses **only XGBoost Classifier** (best performing model)
- **Changes**:
  - Removed baseline model training
  - Directly uses XGBoost as the classification model
  - Model saved as `classification_model.pkl`

### Regression Models
- **Before**: Used XGBoost Regressor for both reliability and deception
- **After**: Uses **GRU (Gated Recurrent Unit)** for both targets (best performing)
- **Changes**:
  - `train_regression_models()` now calls `train_gru()` instead of `train_xgboost()`
  - Features are scaled before GRU training
  - Scalers are saved separately (`reliability_scaler.pkl`, `deception_scaler.pkl`)
  - Models saved as `.h5` files (Keras format)

### TSSP Input Integration
- **Before**: Used raw dataset values for reliability/deception scores
- **After**: Uses **ML model predictions** exclusively
- **Changes**:
  - Behavior probabilities from **XGBoost Classifier** predictions
  - Reliability scores from **GRU Regressor** predictions
  - Deception scores from **GRU Regressor** predictions
  - Stage 1 costs adjusted based on predicted reliability
  - Information values calculated using predicted reliability

## Key Files Modified

1. **`src/pipeline.py`**:
   - `train_classification_model()`: Removed baseline models, uses only XGBoost
   - `train_regression_models()`: Changed to use GRU instead of XGBoost
   - `prepare_tssp_inputs()`: Now uses ML predictions instead of raw data
   - Added scaler storage and loading

2. **Model Files**:
   - Classification: `models/classification_model.pkl` (XGBoost)
   - Reliability: `models/reliability_model.h5` (GRU) + `reliability_scaler.pkl`
   - Deception: `models/deception_model.h5` (GRU) + `deception_scaler.pkl`

## Data Flow

```
Dataset Features
    ↓
┌─────────────────────────────────────┐
│  XGBoost Classifier                 │ → Behavior Probabilities → TSSP Stage 2
│  (Classification)                   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  GRU Regressor (Reliability)        │ → Reliability Scores → TSSP Stage 1 & Info Value
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  GRU Regressor (Deception)          │ → Deception Scores → TSSP Info Value
└─────────────────────────────────────┘
```

## Verification

To verify the implementation:
1. Run the pipeline: `python main.py`
2. Check console output - should show:
   - "Training XGBoost Classifier (best performing model)..."
   - "Training Reliability Score Model with GRU (best performing model)..."
   - "Training Deception Score Model with GRU (best performing model)..."
   - "Generating behavior probabilities from XGBoost Classifier..."
   - "Generating reliability and deception scores from GRU models..."
3. Check model files in `models/` directory
4. Verify TSSP uses predictions (check `prepare_tssp_inputs` output)

## Notes

- GRU models require TensorFlow/Keras
- If TensorFlow is not available, the code will raise an error (GRU is required)
- Model loading now handles both `.pkl` (XGBoost) and `.h5` (GRU) formats
- Scalers must be loaded alongside GRU models for proper predictions
