"""
Setup script for HUMINT ML-TSSP project.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="humint-ml-tssp",
    version="1.0.0",
    description="Hybrid ML-TSSP model for HUMINT source performance evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "xgboost>=1.7.0",
        "tensorflow>=2.10.0",
        "imbalanced-learn>=0.10.0",
        "pyomo>=6.4.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "joblib>=1.2.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
