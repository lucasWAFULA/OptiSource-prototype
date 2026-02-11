# Project: HUMINT Source Performance: ML-TSSP Model

This repository contains a hybrid Machine Learning - Two-Stage Stochastic Programming (ML-TSSP) system for evaluating and optimizing HUMINT (Human Intelligence) source performance and task assignments.

## How to Run on Streamlit Cloud

1. **Push this repository to GitHub** (if not already).
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)** and create a new app.
3. Set the main file to `streamlit_app.py`.
4. The app will use `requirements.txt` for dependencies and `.streamlit/config.toml` for configuration.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Environment variables

- **`DATABASE_URL`** (optional): Set to a PostgreSQL connection URI (e.g. Supabase) to use a shared database instead of local SQLite. All deployments using the same `DATABASE_URL` share optimization results and assignments across regions. See **[docs/POSTGRES_SUPABASE_SETUP.md](docs/POSTGRES_SUPABASE_SETUP.md)** for full setup (Supabase project, schema, and connection string).

## Repository Structure
- `dashboard.py`: Main Streamlit dashboard logic
- `streamlit_app.py`: Entry point for Streamlit Cloud
- `requirements.txt`: Python dependencies
- `.streamlit/config.toml`: Streamlit configuration
- `.github/workflows/deploy.yml`: GitHub Actions workflow
- `src/`: Source code (ML, optimization, analysis, utils)
- `models/`, `output/`, `data/`: Model files, results, and data (ignored in `.gitignore`)

## For More Information
See the main `README.md` for full documentation.
