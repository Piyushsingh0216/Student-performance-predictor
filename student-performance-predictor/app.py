"""Main Streamlit entrypoint for the AI Student Performance Predictor."""

from __future__ import annotations

import streamlit as st
import plotly.express as px

from database import StudentDatabase
from prediction import StudentPerformancePredictor
from utils import css, load_dataset, risk_badge


st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{css()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def dataset():
    return load_dataset()


@st.cache_resource(show_spinner="Loading AI model...")
def predictor():
    return StudentPerformancePredictor()


df = dataset()
engine = predictor()
db = StudentDatabase()

st.sidebar.markdown("<div class='brand'>AI Performance Lab</div>", unsafe_allow_html=True)
st.sidebar.caption("Production-grade academic intelligence")
theme = st.sidebar.toggle("Light mode", value=False)
if theme:
    st.markdown("<style>:root{color-scheme:light}.stApp{background:#f8fafc;color:#0f172a}</style>", unsafe_allow_html=True)

st.markdown("<section class='hero'><h1>AI-Powered Student Performance Predictor</h1><p>Predict grades, identify academic risk, explain outcomes, and generate professional student reports.</p></section>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", f"{len(df):,}")
col2.metric("Average Marks", f"{df['Final_Exam'].mean():.1f}%")
col3.metric("Average Attendance", f"{df['Attendance'].mean():.1f}%")
col4.metric("Model R2", engine.bundle.metrics.get("regression_r2", "N/A"))

col5, col6, col7, col8 = st.columns(4)
col5.metric("Top Performer", df.loc[df["Final_Exam"].idxmax(), "Name"])
col6.metric("Study Hours", f"{df['Study_Hours'].mean():.1f}/day")
col7.metric("Assignments", f"{df['Assignments'].mean():.1f}%")
col8.metric("Risk Accuracy", engine.bundle.metrics.get("risk_accuracy", "N/A"))

st.markdown("### Performance Dashboard")
filters = st.columns(4)
selected_class = filters[0].multiselect("Class", sorted(df["Class"].unique()), default=sorted(df["Class"].unique()))
selected_section = filters[1].multiselect("Section", sorted(df["Section"].unique()), default=sorted(df["Section"].unique()))
selected_risk = filters[2].multiselect("Risk Level", sorted(df["Risk_Level"].unique()), default=sorted(df["Risk_Level"].unique()))
selected_grade = filters[3].multiselect("Grade", sorted(df["Final_Grade"].unique()), default=sorted(df["Final_Grade"].unique()))

view = df[
    df["Class"].isin(selected_class)
    & df["Section"].isin(selected_section)
    & df["Risk_Level"].isin(selected_risk)
    & df["Final_Grade"].isin(selected_grade)
]

left, right = st.columns((1.15, 1))
with left:
    st.plotly_chart(px.histogram(view, x="Final_Grade", color="Risk_Level", title="Grade Distribution", template="plotly_dark"), use_container_width=True)
    st.plotly_chart(px.scatter(view, x="Attendance", y="Final_Exam", color="Risk_Level", size="Study_Hours", hover_data=["Name", "Class"], title="Attendance vs Final Marks", template="plotly_dark"), use_container_width=True)
with right:
    risk_counts = view["Risk_Level"].value_counts().reset_index()
    risk_counts.columns = ["Risk_Level", "Count"]
    st.plotly_chart(px.pie(risk_counts, names="Risk_Level", values="Count", hole=0.55, title="Risk Distribution", template="plotly_dark"), use_container_width=True)
    importance = engine.feature_importance().head(12)
    st.plotly_chart(px.bar(importance, x="Importance", y="Feature", orientation="h", title="Feature Importance", template="plotly_dark"), use_container_width=True)

st.markdown("### Academic Operations")
tab_students, tab_history = st.tabs(["Students", "Prediction History"])
with tab_students:
    query = st.text_input("Search students by ID, name, class, section, risk, or grade")
    table = view.copy()
    if query:
        q = query.lower()
        table = table[table.astype(str).apply(lambda row: row.str.lower().str.contains(q).any(), axis=1)]
    st.dataframe(table.head(500), use_container_width=True, hide_index=True)
with tab_history:
    history = db.prediction_history()
    if history.empty:
        st.info("Prediction history will appear here after using the Prediction page.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)

st.markdown("### Risk Snapshot")
cols = st.columns(4)
for index, risk in enumerate(["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]):
    count = int((df["Risk_Level"] == risk).sum())
    cols[index].markdown(f"<div class='glass-card'><h3>{risk_badge(risk)}</h3><strong>{count:,}</strong><span>students</span></div>", unsafe_allow_html=True)
