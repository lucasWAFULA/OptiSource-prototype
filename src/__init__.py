"""HUMINT ML-TSSP source performance evaluation system."""

# Avoid importing training pipeline at package import time.
# This keeps Streamlit Cloud from failing when src.data is not shipped.
try:
    from .pipeline import MLTSSPPipeline
    __all__ = ['MLTSSPPipeline']
except Exception:
    MLTSSPPipeline = None
    __all__ = []
