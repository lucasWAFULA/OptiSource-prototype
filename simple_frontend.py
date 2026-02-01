"""
Simplified Streamlit frontend for ML-TSSP optimization.
Calls FastAPI backend and displays results.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ML-TSSP Simple Dashboard", layout="wide")

BACKEND_URL = st.sidebar.text_input(
    "Backend URL",
    value="http://127.0.0.1:8000",
    help="FastAPI backend URL (must be running)",
)

st.title("ML-TSSP Optimization Dashboard")
st.caption("Simplified dashboard – FastAPI backend + Streamlit frontend. Results from ML-TSSP model.")

# Sidebar: Configuration
st.sidebar.header("Configuration")
n_sources = st.sidebar.slider("Number of sources", min_value=5, max_value=80, value=20)
st.sidebar.caption("Demo mode: up to 80 sources. Backend generates synthetic data.")

st.sidebar.subheader("Recourse rules")
rel_disengage = st.sidebar.slider("Reliability disengage threshold", 0.0, 1.0, 0.35, 0.05)
rel_ci_flag = st.sidebar.slider("Reliability CI flag threshold", 0.0, 1.0, 0.50, 0.05)
dec_disengage = st.sidebar.slider("Deception disengage threshold", 0.0, 1.0, 0.75, 0.05)
dec_ci_flag = st.sidebar.slider("Deception CI flag threshold", 0.0, 1.0, 0.60, 0.05)

recourse_rules = {
    "rel_disengage": rel_disengage,
    "rel_ci_flag": rel_ci_flag,
    "dec_disengage": dec_disengage,
    "dec_ci_flag": dec_ci_flag,
}

# Health check
health_url = f"{BACKEND_URL.rstrip('/')}/health"
try:
    r = requests.get(health_url, timeout=3)
    health = r.json()
    if health.get("status") == "ok" and health.get("models_loaded"):
        st.sidebar.success("Backend connected. ML models loaded.")
    elif health.get("status") == "ok":
        st.sidebar.warning("Backend connected. ML models not loaded.")
    else:
        st.sidebar.error(f"Backend issue: {health.get('message', 'Unknown')}")
except requests.exceptions.RequestException as e:
    st.sidebar.error(f"Cannot reach backend at {BACKEND_URL}. Start it with: python simple_backend.py")

# Run optimization
if st.button("Run ML-TSSP Optimization", type="primary", use_container_width=True):
    optimize_url = f"{BACKEND_URL.rstrip('/')}/optimize"
    payload = {
        "n_sources": n_sources,
        "seed": 42,
        "recourse_rules": recourse_rules,
    }

    with st.spinner("Running ML-TSSP optimization..."):
        try:
            resp = requests.post(optimize_url, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()
            st.session_state["optimize_result"] = result
        except requests.exceptions.Timeout:
            st.error("Request timed out. Try fewer sources.")
        except requests.exceptions.RequestException as e:
            st.error(f"Backend error: {e}")
        except Exception as e:
            st.error(str(e))

# Display results
result = st.session_state.get("optimize_result")
if result:
    policies = result.get("policies", {})
    ml_policy = policies.get("ml_tssp", [])
    emv = result.get("emv", {})

    if not ml_policy:
        st.warning("No ML-TSSP results in response.")
    else:
        ml_emv = emv.get("ml_tssp", 0)
        uni_emv = emv.get("uniform", 1e-6)
        risk_reduction = ((uni_emv - ml_emv) / uni_emv * 100) if uni_emv > 0 else 0

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total sources", len(ml_policy))
        with col2:
            st.metric("ML-TSSP EMV", f"{ml_emv:.3f}")
        with col3:
            st.metric("Uniform EMV", f"{uni_emv:.3f}")
        with col4:
            st.metric("Risk reduction vs uniform", f"{risk_reduction:.1f}%")

        st.divider()

        # Assignments table
        df = pd.DataFrame(ml_policy)
        df = df[["source_id", "reliability", "deception", "action", "task", "expected_risk", "risk_bucket"]]
        st.subheader("Assignments")
        st.dataframe(df, use_container_width=True)

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            risk_counts = pd.Series([p["risk_bucket"] for p in ml_policy]).value_counts()
            fig_risk = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                labels={"x": "Risk bucket", "y": "Count"},
                title="Risk distribution",
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        with col_chart2:
            action_counts = pd.Series([p.get("action", "task") for p in ml_policy]).value_counts()
            fig_action = px.bar(
                x=action_counts.index,
                y=action_counts.values,
                labels={"x": "Action", "y": "Count"},
                title="Action distribution (ML-TSSP decisions)",
            )
            st.plotly_chart(fig_action, use_container_width=True)
else:
    st.info("Configure parameters and click **Run ML-TSSP Optimization**.")
