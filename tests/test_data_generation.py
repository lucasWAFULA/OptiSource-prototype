"""
Basic tests for data generation module.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import generate_humint_dataset


def test_data_generation():
    """Test that data generation works correctly."""
    df = generate_humint_dataset(n_sources=100, random_seed=42)
    
    assert len(df) == 100
    assert 'source_id' in df.columns
    assert 'behavior_class' in df.columns
    assert 'reliability_score' in df.columns
    assert df['behavior_class'].isin(['cooperative', 'deceptive', 'coerced', 'uncertain']).all()


def test_data_columns():
    """Test that all expected columns are present."""
    df = generate_humint_dataset(n_sources=10, random_seed=42)
    
    expected_columns = [
        'source_id', 'task_success_rate', 'corroboration_score',
        'report_timeliness', 'handler_confidence', 'deception_score',
        'ci_flag', 'report_accuracy', 'report_frequency', 'access_level',
        'information_value', 'handling_cost_kes', 'threat_relevant_features',
        'reliability_score', 'behavior_class', 'scenario_probability'
    ]
    
    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"
