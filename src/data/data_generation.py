"""
Synthetic dataset generation for HUMINT ML-TSSP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import BEHAVIOR_CLASSES


def _assign_behavior_class(reliability: float, deception: float) -> str:
    """
    Simple heuristic mapping for synthetic behavior labels.
    Keeps class distribution plausible and deterministic.
    """
    if deception >= 0.70:
        return "deceptive"
    if reliability <= 0.35 and deception >= 0.45:
        return "coerced"
    if reliability <= 0.45:
        return "uncertain"
    return "cooperative"


def generate_humint_dataset(
    n_sources: int = 15000,
    random_seed: int = 42,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic HUMINT dataset with required columns.

    Parameters
    ----------
    n_sources : int
        Number of sources to generate.
    random_seed : int
        Random seed for reproducibility.
    output_path : Optional[str | Path]
        If provided, save CSV to this path.
    """
    rng = np.random.default_rng(random_seed)

    source_id = [f"SRC_{i+1:05d}" for i in range(n_sources)]
    task_success_rate = rng.uniform(0.2, 0.95, size=n_sources)
    corroboration_score = rng.uniform(0.1, 0.95, size=n_sources)
    report_timeliness = rng.uniform(0.1, 0.95, size=n_sources)
    handler_confidence = rng.uniform(0.2, 0.95, size=n_sources)
    deception_score = rng.uniform(0.0, 0.9, size=n_sources)
    ci_flag = rng.binomial(1, 0.2, size=n_sources)

    report_accuracy = rng.uniform(0.2, 0.95, size=n_sources)
    report_frequency = rng.uniform(0.1, 1.0, size=n_sources)
    access_level = rng.integers(1, 6, size=n_sources)
    information_value = rng.uniform(0.1, 1.0, size=n_sources)
    handling_cost_kes = rng.integers(500, 5000, size=n_sources)
    threat_relevant_features = rng.uniform(0.0, 1.0, size=n_sources)

    reliability_score = (
        0.45 * task_success_rate
        + 0.25 * corroboration_score
        + 0.20 * report_timeliness
        + 0.10 * handler_confidence
        - 0.15 * deception_score
    )
    reliability_score = np.clip(reliability_score, 0.0, 1.0)

    behavior_class = [
        _assign_behavior_class(rel, dec)
        for rel, dec in zip(reliability_score, deception_score)
    ]
    if set(behavior_class) - set(BEHAVIOR_CLASSES):
        # Fallback safety to allowed classes
        behavior_class = [
            cls if cls in BEHAVIOR_CLASSES else "uncertain" for cls in behavior_class
        ]

    scenario_probability = rng.uniform(0.0, 1.0, size=n_sources)

    df = pd.DataFrame(
        {
            "source_id": source_id,
            "task_success_rate": task_success_rate,
            "corroboration_score": corroboration_score,
            "report_timeliness": report_timeliness,
            "handler_confidence": handler_confidence,
            "deception_score": deception_score,
            "ci_flag": ci_flag,
            "report_accuracy": report_accuracy,
            "report_frequency": report_frequency,
            "access_level": access_level,
            "information_value": information_value,
            "handling_cost_kes": handling_cost_kes,
            "threat_relevant_features": threat_relevant_features,
            "reliability_score": reliability_score,
            "behavior_class": behavior_class,
            "scenario_probability": scenario_probability,
        }
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
