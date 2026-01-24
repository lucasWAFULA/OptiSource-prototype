# User Guide

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Optimization solver (GLPK, CBC, Gurobi, or CPLEX)

### Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install optimization solver** (choose one):

   **Option A: GLPK (Recommended for beginners)**
   ```bash
   # Linux
   sudo apt-get install glpk-utils
   
   # macOS
   brew install glpk
   
   # Windows: Download from https://www.gnu.org/software/glpk/
   ```

   **Option B: CBC (Often included with Pyomo)**
   ```bash
   # Usually works out of the box with Pyomo
   ```

   **Option C: Gurobi/CPLEX (Commercial, requires license)**
   ```bash
   # Follow vendor-specific installation instructions
   ```

## Basic Usage

### Running the Complete Pipeline

The simplest way to run the system:

```bash
python main.py
```

This will:
1. Generate a dataset with 15,000 sources
2. Train ML models (classification and regression)
3. Prepare optimization inputs
4. Solve TSSP model with 100 sources and 10 tasks
5. Generate cost analysis and reports

### Customizing Parameters

```bash
# Use more sources in dataset
python main.py --n-sources 20000

# Optimize with more sources and tasks
python main.py --opt-sources 200 --opt-tasks 15

# Use a different solver
python main.py --solver cbc

# Use existing dataset
python main.py --data-path data/my_dataset.csv

# Skip ML training (use existing models)
python main.py --skip-ml
```

## Step-by-Step Usage

### 1. Generate Dataset Only

```python
from src.data import generate_humint_dataset

df = generate_humint_dataset(
    n_sources=15000,
    output_path="my_dataset.csv"
)
```

### 2. Train ML Models Only

```python
from src.pipeline import MLTSSPPipeline

pipeline = MLTSSPPipeline(data_path="my_dataset.csv")
pipeline.load_or_generate_data()
pipeline.train_classification_model()
pipeline.train_regression_models()
```

### 3. Run Optimization Only

```python
from src.pipeline import MLTSSPPipeline

pipeline = MLTSSPPipeline(data_path="my_dataset.csv")
pipeline.load_or_generate_data()
# ... load or train models ...

tssp_inputs = pipeline.prepare_tssp_inputs(n_sources=100, n_tasks=10)
pipeline.solve_tssp(tssp_inputs, solver_name='glpk')
```

### 4. Analyze Results

```python
analysis_results = pipeline.analyze_results()
print(analysis_results['decomposition'])
```

## Understanding Outputs

### Model Files (`models/`)

- `classification_model.pkl`: Trained behavior classifier
- `classification_model_label_encoder.pkl`: Label encoder for classes
- `reliability_model.pkl`: Reliability score predictor
- `deception_model.pkl`: Deception score predictor

### Analysis Outputs (`output/`)

- `cost_by_behavior.png`: Bar chart of costs by behavior class
- `cost_by_source.png`: Bar chart of top sources by cost
- `cost_pie_chart.png`: Pie chart of Stage 1 vs Stage 2 costs
- `cost_analysis_report.txt`: Detailed text report

### Report Contents

The cost analysis report includes:
- Cost verification (Stage 1 + Stage 2 = Optimal value)
- Total cost breakdown
- Stage 2 proportion of total cost
- Cost attribution by behavior class
- Top sources contributing to Stage 2 costs

## Configuration

### Modifying Recourse Costs

Edit `src/utils/config.py`:

```python
RECOURSE_COSTS = {
    'cooperative': 0,      # No recourse needed
    'deceptive': 50000,   # High cost
    'coerced': 30000,     # Medium cost
    'uncertain': 15000,   # Low cost
}
```

### Changing Model Parameters

Edit `src/utils/config.py`:

```python
XGBOOST_CLASSIFIER_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    # ... modify as needed
}
```

## Troubleshooting

### Solver Not Found

**Error**: `Solver 'glpk' not available`

**Solution**:
1. Install GLPK: `sudo apt-get install glpk-utils` (Linux)
2. Or use CBC: `python main.py --solver cbc`
3. Verify: `pyomo --help` should list available solvers

### Memory Issues

**Error**: Out of memory during optimization

**Solution**:
- Reduce problem size: `--opt-sources 50 --opt-tasks 5`
- Use a more efficient solver (Gurobi/CPLEX)
- Process in smaller batches

### Model Training Fails

**Error**: TensorFlow/GPU issues

**Solution**:
- Install CPU-only TensorFlow: `pip install tensorflow-cpu`
- Or disable GRU models (XGBoost will be used)

### Import Errors

**Error**: `ModuleNotFoundError`

**Solution**:
```bash
# Ensure you're in the project root directory
cd /path/to/Updated-FINAL-DASH

# Install dependencies
pip install -r requirements.txt

# Run from project root
python main.py
```

## Advanced Usage

### Custom Feature Lists

Edit `training_features_nc.txt` or `training_features_reg.txt` to modify feature sets.

### Adding New Behavior Classes

1. Update `BEHAVIOR_CLASSES` in `src/utils/config.py`
2. Update `RECOURSE_COSTS` dictionary
3. Regenerate dataset or update existing data

### Custom Solver Configuration

```python
from src.optimization import TSSPModel

# Build model
model = TSSPModel(...)
model.build_model()

# Use custom solver options
solver = SolverFactory('gurobi')
solver.options['TimeLimit'] = 300  # 5 minutes
results = solver.solve(model.model)
```

## Performance Tips

1. **For large datasets**: Use `--skip-ml` after initial training
2. **For faster optimization**: Reduce `--opt-sources` and `--opt-tasks`
3. **For better ML performance**: Increase `n_estimators` in config
4. **For production**: Use commercial solvers (Gurobi/CPLEX)

## Best Practices

1. **Start small**: Test with `--opt-sources 50 --opt-tasks 5`
2. **Save models**: Don't retrain unnecessarily
3. **Verify results**: Check cost verification in reports
4. **Monitor memory**: Watch for large optimization problems
5. **Use version control**: Track configuration changes

## Getting Help

- Check `README.md` for overview
- Review `docs/ARCHITECTURE.md` for system design
- Examine example outputs in `output/`
- Review code comments in source files
