"""
End-to-end pipeline for HUMINT ML-TSSP model.

This module integrates data generation, ML model training, and TSSP optimization
into a complete pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import sys

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_generation import generate_humint_dataset
from src.data.preprocessing import prepare_classification_data, prepare_regression_data
from src.ml import ClassificationModelTrainer, RegressionModelTrainer
from src.optimization import TSSPModel
from src.analysis import analyze_costs, generate_cost_report
from src.utils.config import (
    PROJECT_ROOT, MODELS_DIR, OUTPUT_DIR, BEHAVIOR_CLASSES, RECOURSE_COSTS,
    CLASSIFICATION_FEATURES_FILE, REGRESSION_FEATURES_FILE
)


class MLTSSPPipeline:
    """
    Complete pipeline for ML-TSSP HUMINT source performance evaluation.
    """
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        random_seed: int = 42
    ):
        """
        Initialize pipeline.
        
        Parameters:
        -----------
        data_path : Optional[Path]
            Path to dataset. If None, will generate new data.
        random_seed : int
            Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_seed = random_seed
        self.df = None
        self.classification_trainer = None
        self.reliability_trainer = None
        self.deception_trainer = None
        self.reliability_scaler = None
        self.deception_scaler = None
        self.tssp_model = None
        self.label_encoder = None
    
    def load_or_generate_data(self, n_sources: int = 15000) -> pd.DataFrame:
        """
        Load existing dataset or generate new one.
        
        Parameters:
        -----------
        n_sources : int
            Number of sources to generate if creating new dataset
        
        Returns:
        --------
        pd.DataFrame
            Dataset
        """
        if self.data_path and Path(self.data_path).exists():
            print(f"Loading dataset from: {self.data_path}")
            self.df = pd.read_csv(self.data_path)
        else:
            print(f"Generating new dataset with {n_sources} sources...")
            self.df = generate_humint_dataset(
                n_sources=n_sources,
                random_seed=self.random_seed,
                output_path=self.data_path or (PROJECT_ROOT / "humint_source_dataset_15000_enhanced.csv")
            )
        
        print(f"Dataset loaded: {len(self.df)} sources")
        return self.df
    
    def train_classification_model(
        self,
        use_smote: bool = True,
        save_model: bool = True
    ) -> Dict:
        """
        Train classification model for behavior prediction.
        
        Parameters:
        -----------
        use_smote : bool
            Whether to use SMOTE for class imbalance
        save_model : bool
            Whether to save the trained model
        
        Returns:
        --------
        Dictionary with training results
        """
        print("\n" + "="*60)
        print("TRAINING CLASSIFICATION MODEL")
        print("="*60)
        
        # Prepare data
        X_train, y_train, X_test, y_test, label_encoder = prepare_classification_data(
            self.df,
            feature_file=CLASSIFICATION_FEATURES_FILE,
            random_state=self.random_seed
        )
        
        # Apply SMOTE if requested
        if use_smote:
            print("Applying SMOTE for class imbalance...")
            self.classification_trainer = ClassificationModelTrainer(random_state=self.random_seed)
            X_train, y_train = self.classification_trainer.apply_smote(X_train, y_train)
        else:
            self.classification_trainer = ClassificationModelTrainer(random_state=self.random_seed)
        
        # Train XGBoost Classifier (best performing model for classification)
        print("\nTraining XGBoost Classifier (best performing model)...")
        xgb_results = self.classification_trainer.train_xgboost(
            X_train, y_train, X_test, y_test
        )
        print(f"XGBoost Accuracy: {xgb_results['metrics']['accuracy']:.4f}")
        print(f"XGBoost F1-Score: {xgb_results['metrics']['f1']:.4f}")
        print(f"XGBoost Precision: {xgb_results['metrics']['precision']:.4f}")
        print(f"XGBoost Recall: {xgb_results['metrics']['recall']:.4f}")
        
        # Set XGBoost as the best model
        self.classification_trainer.best_model = xgb_results['model']
        self.classification_trainer.best_model_name = 'xgboost'
        
        # Save model
        if save_model:
            model_path = MODELS_DIR / "classification_model.pkl"
            self.classification_trainer.save_model(model_path, label_encoder)
        
        # Store label encoder for later use
        self.label_encoder = label_encoder
        
        return {
            'best_model': 'xgboost',
            'metrics': xgb_results['metrics'],
            'label_encoder': label_encoder
        }
    
    def train_regression_models(
        self,
        save_models: bool = True
    ) -> Dict:
        """
        Train regression models for reliability and deception scores.
        
        Parameters:
        -----------
        save_models : bool
            Whether to save the trained models
        
        Returns:
        --------
        Dictionary with training results
        """
        print("\n" + "="*60)
        print("TRAINING REGRESSION MODELS")
        print("="*60)
        
        results = {}
        
        # Train reliability score model using GRU (best performing model for regression)
        print("\nTraining Reliability Score Model with GRU (best performing model)...")
        X_train, y_train, X_test, y_test = prepare_regression_data(
            self.df,
            feature_file=REGRESSION_FEATURES_FILE,
            target_col='reliability_score',
            random_state=self.random_seed
        )
        
        # Scale features for GRU
        from src.data.preprocessing import scale_features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        self.reliability_trainer = RegressionModelTrainer(random_state=self.random_seed)
        reliability_results = self.reliability_trainer.train_gru(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        print(f"Reliability GRU R²: {reliability_results['metrics']['r2']:.4f}")
        print(f"Reliability GRU RMSE: {reliability_results['metrics']['rmse']:.4f}")
        print(f"Reliability GRU MAE: {reliability_results['metrics']['mae']:.4f}")
        
        # Store scaler for later predictions
        self.reliability_scaler = scaler
        
        if save_models:
            model_path = MODELS_DIR / "reliability_model.keras"
            self.reliability_trainer.save_model(model_path)
            # Also save scaler
            scaler_path = MODELS_DIR / "reliability_scaler.pkl"
            joblib.dump(scaler, scaler_path)
        
        results['reliability'] = reliability_results['metrics']
        
        # Train deception score model using GRU (best performing model for regression)
        print("\nTraining Deception Score Model with GRU (best performing model)...")
        X_train, y_train, X_test, y_test = prepare_regression_data(
            self.df,
            feature_file=REGRESSION_FEATURES_FILE,
            target_col='deception_score',
            random_state=self.random_seed
        )
        
        # Scale features for GRU
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        self.deception_trainer = RegressionModelTrainer(random_state=self.random_seed)
        deception_results = self.deception_trainer.train_gru(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        print(f"Deception GRU R²: {deception_results['metrics']['r2']:.4f}")
        print(f"Deception GRU RMSE: {deception_results['metrics']['rmse']:.4f}")
        print(f"Deception GRU MAE: {deception_results['metrics']['mae']:.4f}")
        
        # Store scaler for later predictions
        self.deception_scaler = scaler
        
        if save_models:
            model_path = MODELS_DIR / "deception_model.keras"
            self.deception_trainer.save_model(model_path)
            # Also save scaler
            scaler_path = MODELS_DIR / "deception_scaler.pkl"
            joblib.dump(scaler, scaler_path)
        
        results['deception'] = deception_results['metrics']
        
        return results
    
    def prepare_tssp_inputs(
        self,
        n_sources: int = 100,
        n_tasks: int = 10
    ) -> Dict:
        """
        Prepare inputs for TSSP optimization model.
        
        Parameters:
        -----------
        n_sources : int
            Number of sources to include in optimization
        n_tasks : int
            Number of tasks to optimize
        
        Returns:
        --------
        Dictionary with TSSP model inputs
        """
        print("\n" + "="*60)
        print("PREPARING TSSP INPUTS")
        print("="*60)
        
        # Select subset of sources
        sources_df = self.df.head(n_sources).copy()
        sources = sources_df['source_id'].tolist()
        
        # Create tasks
        tasks = [f"TASK_{i:03d}" for i in range(1, n_tasks + 1)]
        
        # Get behavior probabilities P(b | s) from XGBoost Classifier predictions
        # Note: Using ML predictions (XGBoost) is preferred over scenario-based probabilities
        # as it provides more accurate behavior classification based on source features
        print("Generating behavior probabilities P(b | s) from XGBoost Classifier...")
        if self.classification_trainer is None or self.classification_trainer.best_model is None:
            raise ValueError("Classification model (XGBoost) must be trained first")
        
        # Prepare features for prediction
        from src.data.preprocessing import load_features_from_file
        features = load_features_from_file(CLASSIFICATION_FEATURES_FILE)
        available_features = [f for f in features if f in sources_df.columns]
        X_pred = sources_df[available_features]
        
        # Use XGBoost model to predict behavior probabilities
        # This gives P(b | s) for each source s and behavior class b
        xgb_model = self.classification_trainer.best_model
        proba = xgb_model.predict_proba(X_pred)
        
        # Create behavior probability dictionary: P(b | s)
        # Key: (source_id, behavior_class), Value: probability
        behavior_probabilities = {}
        for idx, source_id in enumerate(sources):
            for class_idx, behavior in enumerate(BEHAVIOR_CLASSES):
                behavior_probabilities[(source_id, behavior)] = float(proba[idx, class_idx])
        
        # Verify probabilities sum to 1.0 for each source (with small tolerance)
        for source_id in sources:
            total_prob = sum(
                behavior_probabilities.get((source_id, b), 0.0) 
                for b in BEHAVIOR_CLASSES
            )
            if abs(total_prob - 1.0) > 0.01:  # Allow small numerical errors
                print(f"Warning: Probabilities for source {source_id} sum to {total_prob:.4f}, not 1.0")
        
        # Get reliability and deception scores from GRU regression models
        print("Generating reliability and deception scores from GRU models...")
        if self.reliability_trainer is None or self.reliability_trainer.best_model is None:
            raise ValueError("Reliability GRU model must be trained first")
        if self.deception_trainer is None or self.deception_trainer.best_model is None:
            raise ValueError("Deception GRU model must be trained first")
        
        # Prepare features for regression predictions
        reg_features = load_features_from_file(REGRESSION_FEATURES_FILE)
        available_reg_features = [f for f in reg_features if f in sources_df.columns]
        X_reg_pred = sources_df[available_reg_features]
        
        # Predict reliability scores using GRU
        X_reg_pred_scaled = self.reliability_scaler.transform(X_reg_pred)
        X_reg_pred_reshaped = X_reg_pred_scaled.reshape(X_reg_pred_scaled.shape[0], 1, X_reg_pred_scaled.shape[1])
        reliability_predictions = self.reliability_trainer.best_model.predict(X_reg_pred_reshaped, verbose=0).flatten()
        
        # Predict deception scores using GRU
        X_reg_pred_scaled = self.deception_scaler.transform(X_reg_pred)
        X_reg_pred_reshaped = X_reg_pred_scaled.reshape(X_reg_pred_scaled.shape[0], 1, X_reg_pred_scaled.shape[1])
        deception_predictions = self.deception_trainer.best_model.predict(X_reg_pred_reshaped, verbose=0).flatten()
        
        # Store predictions in dataframe for later use
        sources_df['predicted_reliability'] = reliability_predictions
        sources_df['predicted_deception'] = deception_predictions
        
        # =====================================================
        # COST STRUCTURES
        # =====================================================
        # Stage 1 cost: c(s,t) = 10 * (1 - reliability[s])
        #   - Lower reliability → higher cost (inverse relationship)
        #   - Task-independent (same cost for all tasks)
        #   - Uses predicted reliability from GRU model
        print("Calculating Stage 1 costs using ML predictions...")
        stage1_costs = {}
        for idx, source_id in enumerate(sources):
            source_data = sources_df[sources_df['source_id'] == source_id].iloc[0]
            # Use predicted reliability from GRU model
            predicted_reliability = source_data['predicted_reliability']
            
            # Stage 1 cost formula: c(s,t) = 10 * (1 - reliability[s])
            # This ensures: lower reliability → higher cost
            base_cost = 10.0 * (1.0 - predicted_reliability)
            
            # Stage 1 cost is the same for all tasks (task-independent)
            # If you need task-specific costs, you can add task complexity factor here
            for task_id in tasks:
                stage1_costs[(source_id, task_id)] = round(base_cost, 2)
        
        # Stage 2 recourse penalties q(b) are defined in config.py:
        #   - cooperative: 0.0   (no recourse needed)
        #   - uncertain: 20.0    (low penalty)
        #   - coerced: 40.0      (medium penalty)
        #   - deceptive: 100.0   (high penalty)
        # Expected Stage 2 cost: E[q_i] = Σ_s Σ_t Σ_b P(b|s) × q(b) × y[s,t,b]
        
        # Information values (based on predicted reliability and information value)
        print("Calculating information values using ML predictions...")
        information_values = {}
        for idx, source_id in enumerate(sources):
            source_data = sources_df[sources_df['source_id'] == source_id].iloc[0]
            # Use predicted reliability from GRU model (not raw data)
            predicted_reliability = source_data['predicted_reliability']
            info_value = source_data.get('information_value', 0.5)
            # Combine predicted reliability with information value
            base_value = (predicted_reliability + info_value) / 2
            for task_id in tasks:
                information_values[(source_id, task_id)] = base_value
        
        print(f"\nSummary of ML predictions used in TSSP:")
        print(f"  - Behavior probabilities: XGBoost Classifier")
        print(f"  - Reliability scores: GRU Regressor (mean: {reliability_predictions.mean():.3f})")
        print(f"  - Deception scores: GRU Regressor (mean: {deception_predictions.mean():.3f})")
        
        print(f"Prepared inputs for {len(sources)} sources and {len(tasks)} tasks")
        
        return {
            'sources': sources,
            'tasks': tasks,
            'behavior_classes': BEHAVIOR_CLASSES,
            'behavior_probabilities': behavior_probabilities,
            'stage1_costs': stage1_costs,
            'recourse_costs': RECOURSE_COSTS,
            'information_values': information_values
        }
    
    def solve_tssp(
        self,
        tssp_inputs: Dict,
        solver_name: str = 'glpk',
        verbose: bool = False
    ) -> bool:
        """
        Build and solve TSSP optimization model.
        
        Parameters:
        -----------
        tssp_inputs : Dict
            Inputs for TSSP model
        solver_name : str
            Solver to use
        verbose : bool
            Whether to print solver output
        
        Returns:
        --------
        bool
            True if solved successfully
        """
        print("\n" + "="*60)
        print("SOLVING TSSP OPTIMIZATION MODEL")
        print("="*60)
        
        self.tssp_model = TSSPModel(**tssp_inputs)
        self.tssp_model.build_model()
        
        print(f"Model built with {len(tssp_inputs['sources'])} sources, "
              f"{len(tssp_inputs['tasks'])} tasks, "
              f"{len(tssp_inputs['behavior_classes'])} behavior classes")
        
        success = self.tssp_model.solve(solver_name=solver_name, verbose=verbose)
        
        if success:
            print(f"\n✓ Optimization solved successfully!")
            print(f"  Optimal Objective Value: {self.tssp_model.solution['objective_value']:.2f}")
            print(f"  Number of assignments: {len(self.tssp_model.solution['assignments'])}")
        else:
            print(f"\n✗ Optimization failed: {self.tssp_model.solution.get('message', 'Unknown error')}")
        
        return success
    
    def analyze_results(
        self,
        save_outputs: bool = True,
        include_advanced_metrics: bool = True,
        sensitivity_variation: float = 0.2,
        solver_name: str = 'glpk'
    ) -> Dict:
        """
        Analyze TSSP optimization results and generate reports.
        
        Parameters:
        -----------
        save_outputs : bool
            Whether to save analysis outputs
        include_advanced_metrics : bool
            Whether to calculate EVPI, EMV, and sensitivity analysis
        sensitivity_variation : float
            Range of variation for sensitivity analysis (e.g., 0.2 = ±20%)
        solver_name : str
            Solver to use for advanced metrics calculations
        
        Returns:
        --------
        Dictionary with analysis results
        """
        print("\n" + "="*60)
        print("ANALYZING RESULTS")
        print("="*60)
        
        if self.tssp_model is None or self.tssp_model.solution is None:
            raise ValueError("TSSP model must be solved before analysis")
        
        output_dir = OUTPUT_DIR if save_outputs else None
        analysis_results = analyze_costs(self.tssp_model, output_dir=output_dir)
        
        # Generate text report
        if save_outputs:
            report_path = OUTPUT_DIR / "cost_analysis_report.txt"
            report_text = generate_cost_report(
                analysis_results['decomposition'],
                analysis_results['verification'],
                output_path=report_path
            )
            print("\n" + report_text)
        
        # Advanced metrics: EVPI, EMV, and Sensitivity Analysis
        if include_advanced_metrics:
            print("\n" + "="*60)
            print("CALCULATING ADVANCED METRICS")
            print("="*60)
            
            from src.analysis.advanced_metrics import (
                calculate_evpi,
                calculate_emv,
                sensitivity_analysis,
                generate_advanced_metrics_report,
                calculate_efficiency_frontier,
                plot_efficiency_frontier
            )
            
            # Get TSSP inputs for advanced metrics
            tssp_inputs = self.prepare_tssp_inputs()
            
            # Calculate EVPI (Expected Value of Perfect Information)
            print("\nCalculating Expected Value of Perfect Information (EVPI)...")
            try:
                evpi_results = calculate_evpi(
                    tssp_model=self.tssp_model,
                    behavior_classes=tssp_inputs['behavior_classes'],
                    behavior_probabilities=tssp_inputs['behavior_probabilities'],
                    sources=tssp_inputs['sources'],
                    tasks=tssp_inputs['tasks'],
                    stage1_costs=tssp_inputs['stage1_costs'],
                    recourse_costs=tssp_inputs['recourse_costs'],
                    solver_name=solver_name
                )
                print(f"  EVPI: {evpi_results.get('evpi', 0):.2f}")
                print(f"  EVPI Percentage: {evpi_results.get('evpi_percentage', 0):.2f}%")
                analysis_results['evpi'] = evpi_results
            except Exception as e:
                print(f"  Warning: EVPI calculation failed: {e}")
                analysis_results['evpi'] = None
            
            # Calculate EMV (Expected Mission Value)
            print("\nCalculating Expected Mission Value (EMV)...")
            try:
                emv_results = calculate_emv(
                    tssp_model=self.tssp_model,
                    information_values=tssp_inputs.get('information_values')
                )
                print(f"  EMV: {emv_results.get('emv', 0):.2f}")
                print(f"  Information Value: {emv_results.get('information_value', 0):.2f}")
                analysis_results['emv'] = emv_results
            except Exception as e:
                print(f"  Warning: EMV calculation failed: {e}")
                analysis_results['emv'] = None
            
            # Sensitivity Analysis
            print("\nPerforming Sensitivity Analysis...")
            try:
                sensitivity_results = sensitivity_analysis(
                    tssp_model=self.tssp_model,
                    behavior_classes=tssp_inputs['behavior_classes'],
                    behavior_probabilities=tssp_inputs['behavior_probabilities'],
                    sources=tssp_inputs['sources'],
                    tasks=tssp_inputs['tasks'],
                    stage1_costs=tssp_inputs['stage1_costs'],
                    recourse_costs=tssp_inputs['recourse_costs'],
                    variation_range=sensitivity_variation,
                    solver_name=solver_name,
                    output_dir=output_dir
                )
                print("  Sensitivity analysis completed")
                analysis_results['sensitivity'] = sensitivity_results
            except Exception as e:
                print(f"  Warning: Sensitivity analysis failed: {e}")
                analysis_results['sensitivity'] = None
            
            # Efficiency Frontier Analysis
            print("\nCalculating Efficiency Frontier...")
            try:
                frontier_results = calculate_efficiency_frontier(
                    sources=tssp_inputs['sources'],
                    tasks=tssp_inputs['tasks'],
                    behavior_classes=tssp_inputs['behavior_classes'],
                    behavior_probabilities=tssp_inputs['behavior_probabilities'],
                    stage1_costs=tssp_inputs['stage1_costs'],
                    recourse_costs=tssp_inputs['recourse_costs'],
                    n_scenarios=20,
                    solver_name=solver_name
                )
                print(f"  Efficiency frontier calculated: {len(frontier_results['frontier_points'])} frontier points")
                
                # Generate plot if output directory is available
                if output_dir:
                    frontier_plot_path = output_dir / "efficiency_frontier.png"
                    plot_efficiency_frontier(frontier_results, output_path=frontier_plot_path)
                
                analysis_results['efficiency_frontier'] = frontier_results
            except Exception as e:
                print(f"  Warning: Efficiency frontier calculation failed: {e}")
                analysis_results['efficiency_frontier'] = None
            
            # Generate advanced metrics report
            if save_outputs and analysis_results.get('evpi') and analysis_results.get('emv') and analysis_results.get('sensitivity'):
                try:
                    advanced_report_path = OUTPUT_DIR / "advanced_metrics_report.txt"
                    advanced_report = generate_advanced_metrics_report(
                        evpi_results=analysis_results['evpi'],
                        emv_results=analysis_results['emv'],
                        sensitivity_results=analysis_results['sensitivity'],
                        output_path=advanced_report_path
                    )
                    print("\n" + advanced_report)
                except Exception as e:
                    print(f"  Warning: Advanced metrics report generation failed: {e}")
        
        return analysis_results
    
    def run_full_pipeline(
        self,
        n_sources: int = 15000,
        opt_n_sources: int = 100,
        opt_n_tasks: int = 10,
        train_ml: bool = True,
        solver_name: str = 'glpk'
    ) -> Dict:
        """
        Run the complete ML-TSSP pipeline.
        
        Parameters:
        -----------
        n_sources : int
            Number of sources in dataset
        opt_n_sources : int
            Number of sources for optimization
        opt_n_tasks : int
            Number of tasks for optimization
        train_ml : bool
            Whether to train ML models (if False, assumes models exist)
        solver_name : str
            Solver to use for optimization
        
        Returns:
        --------
        Dictionary with all results
        """
        print("\n" + "="*80)
        print("HUMINT ML-TSSP PIPELINE")
        print("="*80)
        
        results = {}
        
        # Step 1: Load or generate data
        self.load_or_generate_data(n_sources=n_sources)
        results['data'] = {'n_sources': len(self.df)}
        
        # Step 2: Train ML models
        if train_ml:
            classification_results = self.train_classification_model()
            regression_results = self.train_regression_models()
            results['ml'] = {
                'classification': classification_results,
                'regression': regression_results
            }
        else:
            # Load existing models
            print("\nLoading existing ML models...")
            self.classification_trainer = ClassificationModelTrainer()
            self.classification_trainer.load_model(MODELS_DIR / "classification_model.pkl")
            self.classification_trainer.best_model = self.classification_trainer.best_model
            self.classification_trainer.best_model_name = 'xgboost'
            
            # Load label encoder
            try:
                self.label_encoder = joblib.load(MODELS_DIR / "classification_model_label_encoder.pkl")
            except FileNotFoundError:
                print("Warning: Label encoder not found, will need to retrain classification model")
            
            # Load GRU models and scalers
            self.reliability_trainer = RegressionModelTrainer()
            reliability_model_path = MODELS_DIR / "reliability_model.keras"
            reliability_h5_path = MODELS_DIR / "reliability_model.h5"
            if reliability_model_path.exists():
                try:
                    self.reliability_trainer.load_model(reliability_model_path)
                except Exception:
                    if reliability_h5_path.exists():
                        self.reliability_trainer.load_model(reliability_h5_path)
                    else:
                        raise
                self.reliability_trainer.best_model = self.reliability_trainer.best_model
                self.reliability_trainer.best_model_name = 'gru'
                self.reliability_scaler = joblib.load(MODELS_DIR / "reliability_scaler.pkl")
            elif reliability_h5_path.exists():
                self.reliability_trainer.load_model(reliability_h5_path)
                self.reliability_trainer.best_model = self.reliability_trainer.best_model
                self.reliability_trainer.best_model_name = 'gru'
                self.reliability_scaler = joblib.load(MODELS_DIR / "reliability_scaler.pkl")
            else:
                raise FileNotFoundError("Reliability GRU model not found. Please train models first.")
            
            self.deception_trainer = RegressionModelTrainer()
            deception_model_path = MODELS_DIR / "deception_model.keras"
            deception_h5_path = MODELS_DIR / "deception_model.h5"
            if deception_model_path.exists():
                try:
                    self.deception_trainer.load_model(deception_model_path)
                except Exception:
                    if deception_h5_path.exists():
                        self.deception_trainer.load_model(deception_h5_path)
                    else:
                        raise
                self.deception_trainer.best_model = self.deception_trainer.best_model
                self.deception_trainer.best_model_name = 'gru'
                self.deception_scaler = joblib.load(MODELS_DIR / "deception_scaler.pkl")
            elif deception_h5_path.exists():
                self.deception_trainer.load_model(deception_h5_path)
                self.deception_trainer.best_model = self.deception_trainer.best_model
                self.deception_trainer.best_model_name = 'gru'
                self.deception_scaler = joblib.load(MODELS_DIR / "deception_scaler.pkl")
            else:
                raise FileNotFoundError("Deception GRU model not found. Please train models first.")
        
        # Step 3: Prepare TSSP inputs
        tssp_inputs = self.prepare_tssp_inputs(
            n_sources=opt_n_sources,
            n_tasks=opt_n_tasks
        )
        results['tssp_inputs'] = {
            'n_sources': len(tssp_inputs['sources']),
            'n_tasks': len(tssp_inputs['tasks'])
        }
        
        # Step 4: Solve TSSP
        success = self.solve_tssp(tssp_inputs, solver_name=solver_name)
        results['tssp'] = {'solved': success}
        
        # Step 5: Analyze results
        if success:
            analysis_results = self.analyze_results(
                save_outputs=True,
                include_advanced_metrics=True,
                sensitivity_variation=0.2,
                solver_name=solver_name
            )
            results['analysis'] = analysis_results
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        
        return results
