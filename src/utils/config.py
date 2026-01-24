"""
Configuration settings for the HUMINT ML-TSSP project.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
if not PROJECT_ROOT.exists():
    # Fallback if running from different location
    PROJECT_ROOT = Path.cwd()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Dataset file
DATASET_FILE = PROJECT_ROOT / "humint_source_dataset_15000_enhanced.csv"

# Feature files
CLASSIFICATION_FEATURES_FILE = PROJECT_ROOT / "training_features_nc.txt"
REGRESSION_FEATURES_FILE = PROJECT_ROOT / "training_features_reg.txt"

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# ML Model parameters
XGBOOST_CLASSIFIER_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'eval_metric': 'mlogloss',
}

XGBOOST_REGRESSOR_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
}

# GRU Model parameters
GRU_HIDDEN_UNITS = 64
GRU_DROPOUT = 0.3
GRU_EPOCHS = 50
GRU_BATCH_SIZE = 32
GRU_PATIENCE = 10

# Behavior classes
BEHAVIOR_CLASSES = ['cooperative', 'deceptive', 'coerced', 'uncertain']

# Stage 2 recourse penalties q(b) by behavior class
# These represent the recourse cost per unit of recourse action
# Following the formulation: q_i(ω) where ω is the behavior scenario
# Cost structure:
#   - cooperative: 0.0   (no recourse needed)
#   - uncertain: 20.0    (low recourse penalty)
#   - coerced: 40.0      (medium recourse penalty)
#   - deceptive: 100.0   (high recourse penalty)
# Expected Stage 2 cost: E[q_i] = Σ_s Σ_t Σ_b P(b|s) × q(b) × y[s,t,b]
RECOURSE_COSTS = {
    'cooperative': 0.0,    # No recourse needed
    'uncertain': 20.0,     # Low recourse penalty for uncertainty
    'coerced': 40.0,       # Medium recourse penalty for coercion
    'deceptive': 100.0,   # High recourse penalty for deception
}
