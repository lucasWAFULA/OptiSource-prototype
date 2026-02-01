# Simple ML-TSSP Dashboard

A simplified FastAPI + Streamlit setup for running the ML-TSSP optimization model without the complexity of the full dashboard.

## Architecture

- **Backend** (`simple_backend.py`): FastAPI server that runs ML predictions and TSSP optimization
- **Frontend** (`simple_frontend.py`): Streamlit UI for configuration and results display

## How to Run

1. **Start the backend** (in one terminal):
   ```bash
   python simple_backend.py
   ```
   The server runs at http://127.0.0.1:8000

2. **Start the frontend** (in another terminal):
   ```bash
   streamlit run simple_frontend.py
   ```

3. In the Streamlit app:
   - Configure number of sources and recourse rules in the sidebar
   - Click **Run ML-TSSP Optimization**
   - View KPI cards, assignments table, and distribution charts

## Requirements

- ML models must be present in the `models/` directory (same as the full dashboard)
- Pyomo with a solver (glpk or cbc) for TSSP optimization
