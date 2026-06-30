"""About page."""

from __future__ import annotations

import streamlit as st

from prediction import StudentPerformancePredictor
from utils import css


st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")
st.markdown(f"<style>{css()}</style>", unsafe_allow_html=True)
engine = StudentPerformancePredictor()

st.title("About This Project")
st.markdown(
    """
    This portfolio project is a production-style academic intelligence system built with Python,
    Streamlit, scikit-learn, SQLite, Plotly, Altair, and optional SHAP explainability.

    It generates a realistic student dataset, trains multiple ML models, selects the best model,
    predicts final percentage, grade, pass/fail status, performance level, risk level, confidence,
    and produces personalized recommendations and downloadable reports.
    """
)
st.subheader("Model Metrics")
st.json(engine.bundle.metrics)
st.subheader("Architecture")
st.code(
    """
    Streamlit UI -> Prediction Service -> Trained ML Bundle
                -> SQLite History
                -> PDF/CSV/Excel Reports
    train_model.py -> Dataset generation -> Cleaning -> Encoding/Scaling -> Model comparison -> Artifacts
    """,
    language="text",
)
