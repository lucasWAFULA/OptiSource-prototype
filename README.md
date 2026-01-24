# HUMINT Source Performance: ML-TSSP Model

A hybrid Machine Learning - Two-Stage Stochastic Programming (ML-TSSP) system for evaluating and optimizing HUMINT (Human Intelligence) source performance and task assignments.

## Overview

This project combines:
- **Machine Learning Models**: Classification (behavior prediction) and Regression (reliability/deception scores)
- **Two-Stage Stochastic Optimization**: Strategic tasking decisions with recourse for uncertain behaviors
- **Cost Analysis**: Comprehensive cost decomposition and attribution analysis

## Project Structure

```
.
├── src/
│   ├── data/              # Data generation and preprocessing
│   │   ├── data_generation.py
│   │   └── preprocessing.py
│   ├── ml/                # Machine learning models
│   │   ├── classification.py
│   │   └── regression.py
│   ├── optimization/       # TSSP optimization model
│   │   └── tssp_model.py
│   ├── analysis/          # Cost analysis and visualization
│   │   └── cost_analysis.py
│   ├── utils/             # Utilities and configuration
│   │   └── config.py
│   └── pipeline.py        # End-to-end pipeline
├── models/                 # Saved ML models (generated)
├── output/                 # Analysis outputs (generated)
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── main.py                 # Main execution script
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Updated-FINAL-DASH
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install optimization solver (choose one):
   - **GLPK** (open-source): `sudo apt-get install glpk-utils` (Linux) or download from [GLPK website](https://www.gnu.org/software/glpk/)
   - **CBC** (open-source): Included with Pyomo, or install separately
   - **Gurobi/CPLEX** (commercial): Requires license

## Usage

### Quick Start

Run the complete pipeline:
```bash
python main.py
```

### Command Line Options

```bash
python main.py --help
```

Options:
- `--n-sources N`: Number of sources in dataset (default: 15000)
- `--opt-sources N`: Number of sources for optimization (default: 100)
- `--opt-tasks N`: Number of tasks for optimization (default: 10)
- `--solver NAME`: Solver to use: 'glpk', 'cbc', 'gurobi', 'cplex' (default: 'glpk')
- `--data-path PATH`: Path to existing dataset
- `--skip-ml`: Skip ML training (use existing models)

### Example

```bash
# Run with custom parameters
python main.py --n-sources 20000 --opt-sources 200 --opt-tasks 15 --solver cbc

# Use existing dataset and skip ML training
python main.py --data-path data/humint_dataset.csv --skip-ml
```

## Pipeline Components

### 1. Data Generation
- Synthetic HUMINT source dataset with 15,000+ sources
- Features: task success rate, reliability, deception scores, behavior classes, etc.
- Configurable via `src/data/data_generation.py`

### 2. Machine Learning Models

#### Classification (Behavior Prediction)
- **XGBoost Classifier**: Primary model for behavior class prediction
- **Baseline Models**: Random Forest, SVM, KNN, Decision Tree
- **GRU Model**: Deep learning alternative (optional)
- **SMOTE**: Handles class imbalance

#### Regression (Performance Metrics)
- **XGBoost Regressor**: Predicts reliability and deception scores
- **Linear Regression**: Baseline model
- **GRU Regressor**: Deep learning alternative (optional)

### 3. TSSP Optimization

**Stage 1 (Here-and-Now Decisions)**:
- Binary assignment variables: `x[s, t]` - assign source `s` to task `t`
- Objective: Minimize strategic tasking costs
- Constraints: Source capacity, task coverage

**Stage 2 (Recourse Decisions)**:
- Continuous recourse variables: `y[s, t, b]` - recourse intensity
- Objective: Minimize expected recourse costs
- Constraints: Recourse feasibility, behavior dependencies

### 4. Cost Analysis
- Stage 1 vs Stage 2 cost decomposition
- Cost attribution by behavior class
- Cost attribution by source
- Visualizations and reports

## Output Files

After running the pipeline, outputs are saved in:

- `models/`: Trained ML models
  - `classification_model.pkl`
  - `reliability_model.pkl`
  - `deception_model.pkl`

- `output/`: Analysis results
  - `cost_by_behavior.png`
  - `cost_by_source.png`
  - `cost_pie_chart.png`
  - `cost_analysis_report.txt`

## Configuration

Edit `src/utils/config.py` to customize:
- Model hyperparameters
- Recourse costs by behavior class
- Feature lists
- Paths and directories

## Model Architecture

### ML-TSSP Integration Flow

```
Data Generation
    ↓
ML Training (Classification + Regression)
    ↓
Behavior Probability Predictions
    ↓
TSSP Optimization Model
    ↓
Cost Analysis & Reporting
```

### TSSP Mathematical Formulation

**Objective Function**:
```
Minimize: Σ(s,t) Stage1Cost[s,t] * x[s,t] + 
          Σ(s,t,b) BehaviorProb[s,b] * RecourseCost[b] * y[s,t,b]
```

**Constraints**:
- Each source assigned to at most one task
- Each task has at least one source
- Recourse only if source assigned to task

## Performance Metrics

### ML Models
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: MAE, MSE, RMSE, R²

### Optimization
- Optimal objective value
- Stage 1 and Stage 2 cost breakdown
- Assignment statistics

## Troubleshooting

### Solver Issues
If solver is not found:
1. Install GLPK: `sudo apt-get install glpk-utils` (Linux)
2. Or use CBC: Usually included with Pyomo
3. Check solver availability: `pyomo --help`

### Memory Issues
For large datasets:
- Reduce `--opt-sources` and `--opt-tasks`
- Use a more efficient solver (Gurobi/CPLEX)
- Process in batches

## Development

### Running Tests
```bash
pytest tests/
```

### Code Structure
- Modular design for easy extension
- Configuration-driven parameters
- Comprehensive error handling

## License

[Specify your license here]

## Citation

If you use this code, please cite:
```
HUMINT Source Performance: ML-TSSP Model
Hybrid Machine Learning - Two-Stage Stochastic Programming Approach
```

## Contact

[Your contact information]
