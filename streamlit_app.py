# Streamlit app entry point for cloud deployment
import streamlit as st
import dashboard

# The dashboard.py file contains the main Streamlit logic
# This file is a simple entry point for Streamlit Cloud

def main():
    # Optionally, you can add a welcome message or redirect to dashboard
    st.title("HUMINT Source Performance: ML-TSSP Dashboard")
    dashboard  # This will execute dashboard.py as a module

if __name__ == "__main__":
    main()
