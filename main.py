"""
Main script to run the HUMINT ML-TSSP pipeline.

Usage:
    python main.py [--n-sources N] [--opt-sources N] [--opt-tasks N] [--solver NAME]
"""

import argparse
from pathlib import Path
from src.pipeline import MLTSSPPipeline
from src.utils.config import PROJECT_ROOT, DATASET_FILE


def main():
    parser = argparse.ArgumentParser(
        description='Run HUMINT ML-TSSP source performance evaluation pipeline'
    )
    parser.add_argument(
        '--n-sources',
        type=int,
        default=500,
        help='Number of sources in dataset (default: 500)'
    )
    parser.add_argument(
        '--opt-sources',
        type=int,
        default=500,
        help='Number of sources for optimization (default: 500)'
    )
    parser.add_argument(
        '--opt-tasks',
        type=int,
        default=20,
        help='Number of tasks for optimization (default: 20)'
    )
    parser.add_argument(
        '--solver',
        type=str,
        default='glpk',
        help='Optimization solver to use (default: glpk)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to existing dataset (default: generate new)'
    )
    parser.add_argument(
        '--skip-ml',
        action='store_true',
        help='Skip ML training (use existing models)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    data_path = Path(args.data_path) if args.data_path else DATASET_FILE
    pipeline = MLTSSPPipeline(data_path=data_path if data_path.exists() else None)
    
    # Run pipeline
    results = pipeline.run_full_pipeline(
        n_sources=args.n_sources,
        opt_n_sources=args.opt_sources,
        opt_n_tasks=args.opt_tasks,
        train_ml=not args.skip_ml,
        solver_name=args.solver
    )
    
    print("\nPipeline execution completed successfully!")
    return results


if __name__ == '__main__':
    main()
