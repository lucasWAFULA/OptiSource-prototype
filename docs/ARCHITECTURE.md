# System Architecture

## Overview

The HUMINT ML-TSSP system is a hybrid approach combining machine learning predictions with two-stage stochastic optimization for intelligent source-task assignment.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Generation Layer                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Synthetic HUMINT Source Dataset (15,000+ sources)  │   │
│  │  - Performance metrics                               │   │
│  │  - Behavior classifications                          │   │
│  │  - Operational characteristics                      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Machine Learning Layer                      │
│  ┌──────────────────────┐  ┌──────────────────────┐       │
│  │  Classification      │  │  Regression          │       │
│  │  - XGBoost          │  │  - Reliability Score │       │
│  │  - GRU              │  │  - Deception Score   │       │
│  │  - Baseline Models │  │  - XGBoost/GRU       │       │
│  └──────────────────────┘  └──────────────────────┘       │
│                            ↓                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Behavior Probability Predictions                    │   │
│  │  P(behavior | source features)                      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              TSSP Optimization Layer                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Stage 1: Strategic Tasking                         │   │
│  │  - Binary decisions: x[s,t]                        │   │
│  │  - Minimize: Σ Stage1Cost[s,t] * x[s,t]           │   │
│  └──────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Stage 2: Recourse Decisions                        │   │
│  │  - Continuous: y[s,t,b]                             │   │
│  │  - Minimize: Σ P[b|s] * RecourseCost[b] * y[s,t,b] │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Analysis & Reporting Layer                   │
│  - Cost decomposition                                        │
│  - Attribution analysis                                      │
│  - Visualizations                                            │
│  - Reports                                                   │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Generation (`src/data/`)

**Purpose**: Generate synthetic HUMINT source datasets with realistic characteristics.

**Key Features**:
- Configurable number of sources
- Realistic feature distributions
- Behavior class assignment based on reliability/deception
- Scenario probability calculation

**Output**: CSV file with source features and labels

### 2. Machine Learning Models (`src/ml/`)

#### Classification Module
- **Primary Model**: XGBoost Classifier
- **Alternative**: GRU (Gated Recurrent Unit) for sequence modeling
- **Baselines**: Random Forest, SVM, KNN, Decision Tree
- **Handling**: SMOTE for class imbalance

**Output**: Behavior class probabilities for each source

#### Regression Module
- **Targets**: Reliability score, Deception score
- **Models**: XGBoost Regressor, Linear Regression, GRU Regressor
- **Evaluation**: R², MAE, MSE, RMSE

**Output**: Predicted reliability and deception scores

### 3. TSSP Optimization (`src/optimization/`)

**Mathematical Formulation**:

**Sets**:
- S: Sources {s₁, s₂, ..., sₙ}
- T: Tasks {t₁, t₂, ..., tₘ}
- B: Behavior classes {cooperative, uncertain, coerced, deceptive}

**Decision Variables**:
- `x[s,t] ∈ {0,1}`: Stage 1 assignment
- `y[s,t,b] ≥ 0`: Stage 2 recourse intensity

**Objective**:
```
Minimize: Σ(s∈S, t∈T) Stage1Cost[s,t] * x[s,t] +
          Σ(s∈S, t∈T, b∈B) P[b|s] * RecourseCost[b] * y[s,t,b]
```

**Constraints**:
1. Source capacity: Σ(t∈T) x[s,t] ≤ 1  ∀s∈S
2. Task coverage: Σ(s∈S) x[s,t] ≥ 1  ∀t∈T
3. Recourse feasibility: y[s,t,b] ≤ x[s,t]  ∀s∈S, t∈T, b∈B

### 4. Cost Analysis (`src/analysis/`)

**Functions**:
- Calculate Stage 1 and Stage 2 costs
- Verify against optimal objective value
- Attribute costs by behavior class
- Attribute costs by source
- Generate visualizations and reports

## Data Flow

1. **Data Generation** → Raw dataset with features
2. **Preprocessing** → Train/test splits, feature scaling
3. **ML Training** → Trained models and predictions
4. **TSSP Input Preparation** → Behavior probabilities, costs, values
5. **Optimization** → Optimal assignments and recourse decisions
6. **Analysis** → Cost breakdowns and insights

## Integration Points

### ML → TSSP
- Behavior probabilities feed into Stage 2 expectation
- Reliability/deception scores inform Stage 1 costs
- Information values derived from ML predictions

### TSSP → Analysis
- Solution variables extracted for cost calculation
- Assignment patterns analyzed
- Recourse decisions evaluated

## Scalability Considerations

- **Data**: Handles 15,000+ sources efficiently
- **ML**: Batch processing, model persistence
- **Optimization**: Configurable problem size (sources/tasks)
- **Solver**: Supports multiple solvers (GLPK, CBC, Gurobi, CPLEX)

## Extensibility

The modular architecture allows for:
- New ML models (add to `src/ml/`)
- Additional constraints (modify `src/optimization/tssp_model.py`)
- Custom analysis (extend `src/analysis/`)
- Alternative data sources (implement in `src/data/`)
