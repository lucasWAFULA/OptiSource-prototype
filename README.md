## Role-Based Access Control

This application implements robust role-based access control (RBAC) to ensure that users only access features appropriate to their responsibilities within the intelligence cycle. Each user is assigned a role, which determines their permissions and accessible dashboard features.

### User Roles

- **Case Officer (Handler):** Manages assigned sources, submits tasking, updates field notes, flags risk, and recommends engagement.
- **Intelligence Analyst:** Views reports, runs models, scores intelligence, and compares sources.
- **Tasking Coordinator:** Approves tasking, assigns tasks, manages priorities, and views capabilities.
- **Source Evaluation Officer:** Views longitudinal metrics, validates scores, and recommends disengagement.
- **Operations Oversight / Legal & Ethics:** Views audit logs, freezes operations, reviews exceptions, and accesses aggregated dashboards.
- **System Administrator:** Manages users, configures the system, and can export data in bulk. Cannot view intelligence content or source performance data.
- **Executive / Strategic Viewer:** Views aggregated dashboards for high-level insights.

### Permissions Matrix (Summary)

| Role                   | Key Permissions                                                                 |
|------------------------|-------------------------------------------------------------------------------|
| Case Officer           | view_assigned_sources, submit_tasking, update_field_notes, flag_risk, recommend_engagement |
| Analyst                | view_reports, run_models, score_intelligence, compare_sources                  |
| Tasking Coordinator    | approve_tasking, assign_tasks, view_capabilities, manage_priorities            |
| Evaluation Officer     | view_longitudinal_metrics, validate_scores, recommend_disengagement            |
| Oversight              | view_audit_logs, freeze_operations, review_exceptions, view_aggregated_dashboards |
| Admin                  | manage_users, configure_system, export_bulk                                    |
| Executive              | view_aggregated_dashboards                                                     |

### How It Works

- User roles are assigned at login and stored in the session.
- Each dashboard section checks permissions before displaying sensitive data or controls.
- Navigation and actions are dynamically enabled/disabled based on the user’s role.
- Admins cannot view intelligence content or source performance data for security and compliance.

For more details, see the role and permission logic in `dashboard.py`.

# HUMINT Source Performance: ML-TSSP Framework

**HUMINT ML-TSSP** is a robust, modular platform for evaluating, optimizing, and visualizing Human Intelligence (HUMINT) source performance and task assignments. Leveraging advanced machine learning and a two-stage stochastic programming (TSSP) optimization model, the system provides actionable insights through an interactive Streamlit dashboard. Designed for reliability, explainability, and operational flexibility, it supports both production and research use cases.


## Key Features

- **End-to-end pipeline**: Transforms raw or simulated HUMINT data into risk-aware, optimized task assignments.
- **ML-driven risk assessment**: Predicts source behavior (cooperative, uncertain, coerced, deceptive), reliability, and deception using XGBoost and GRU models.
- **TSSP optimization**: Assigns sources to tasks, balancing operational cost and risk under uncertainty, with recourse for adverse behaviors.
- **Interactive dashboard**: Visualizes assignments, risk buckets, cost breakdowns, and supports scenario analysis with custom recourse policies.
- **Failsafe fallback**: Remains fully functional using formula-based risk and behavior estimation if ML models are unavailable.


## Core Pipeline Overview

1. **Data Ingestion & Feature Engineering**
  - Accepts CSV uploads or generates synthetic HUMINT datasets (15,000+ sources typical).
  - Features include: task success rate, corroboration, timeliness, handler confidence, deception, CI flags, and more.

2. **Machine Learning Models**
  - **Behavior Classification**: XGBoost classifier predicts probabilities for each behavior class.
  - **Reliability & Deception Regression**: GRU (deep learning) regressors output continuous reliability and deception scores (0-1).
  - **Label encoding & scaling**: Ensures robust, production-ready inference.

3. **Two-Stage Stochastic Optimization (TSSP)**
  - **Stage 1**: Assign sources to tasks to minimize expected cost, subject to operational constraints.
  - **Stage 2 (Recourse)**: Adjusts for realized behaviors (e.g., disengage, flag, escalate) to minimize risk/cost after uncertainty is revealed.
  - **Risk buckets**: Each source is categorized as low, medium, or high risk based on ML-predicted probabilities and recourse policy.

4. **Cost & Risk Analysis**
  - Decomposes total cost by behavior class, source, and recourse action.
  - Computes expected loss (EMV), risk-adjusted assignment scores, and visualizes cost/risk tradeoffs.

5. **Interactive Dashboard (Streamlit)**
  - Upload data, tune recourse policies, and run scenario analysis in real time.
  - Visualizes assignments, risk buckets, KPIs, cost breakdowns, and model explanations (SHAP-style).
  - Fully functional fallback mode if ML models are missing (formula-based risk/behavior estimation).


## Additional Highlights

- **Production-ready**: Modular, robust, and cloud-deployable (Streamlit Cloud, GitHub Actions).
- **Explainable**: SHAP-style feature attributions for model predictions.
- **Customizable**: Recourse policies, risk thresholds, and solver options are user-tunable.
- **Failsafe**: Formula-based fallback ensures dashboard and optimization always work, even if models are missing.
- **Comprehensive outputs**: Assignment tables, risk/cost plots, downloadable reports, and more.


## Example Workflow

1. **Upload or generate HUMINT data**
2. **ML models predict**: behavior probabilities, reliability, deception
3. **TSSP optimizer assigns**: sources to tasks, minimizing expected loss
4. **Recourse actions**: disengage, flag, escalate, or assign based on risk
5. **Dashboard visualizes**: assignments, risk buckets, cost breakdowns, and allows scenario tuning


## Model Details

- **Behavior Classes**: Cooperative, Uncertain, Coerced, Deceptive
- **ML Models**:
  - XGBoost classifier for behavior prediction
  - GRU regressors for reliability and deception (with scaler support)
- **Optimization**:
  - Pyomo-based TSSP model
  - Stage 1: Assignment, Stage 2: Recourse (risk mitigation)
  - Risk buckets: Low (<0.3), Medium (0.3-0.6), High (>0.6)
- **Recourse Actions**: Disengage, flag for CI, escalate, assign
- **Fallback**: Formula-based risk/behavior estimation if models are missing


## Dashboard Highlights

- **KPI indicators**: Assignment quality, risk, and cost
- **Assignment tables**: Per-source actions, risk, and task
- **Risk/cost plots**: By behavior, source, and recourse action
- **Scenario analysis**: Tune recourse thresholds and instantly see impact
- **Health checks**: System status, model availability, and fallback mode


## Getting Started

1. **Local Deployment**:
  ```bash
  pip install -r requirements.txt
  streamlit run streamlit_app.py
  ```
2. **Streamlit Cloud**:
  - Connect this repository and set the main file to `streamlit_app.py`.
3. **Command-line Pipeline**:
  ```bash
  python main.py --n-sources 15000 --opt-sources 100 --opt-tasks 10 --solver glpk
  ```


## Outputs

- `models/`: Trained ML models (`classification_model.pkl`, `reliability_model.keras`, `deception_model.keras`, scalers)
- `output/`: Cost/risk plots, assignment tables, reports


## References

- See `dashboard.py` for full dashboard logic and fallback details
- See `src/pipeline.py` for ML-TSSP pipeline implementation


## License

This project is licensed under the MIT License. See the LICENSE file for details.


## Citation

If you use this code, please cite:

HUMINT Source Performance: ML-TSSP Model  
Hybrid Machine Learning - Two-Stage Stochastic Programming Approach


## Contact

For questions or collaboration, please contact:

**Lucas Wafula**  
Graduate Student, MSc. Data Science & Analytics  
Strathmore Institute of Mathematical Sciences  
Email: lucas.wafula@strathmore.edu

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
