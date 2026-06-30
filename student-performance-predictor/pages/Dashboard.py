"""Detailed dashboard page."""

from __future__ import annotations

import altair as alt
import plotly.express as px
import streamlit as st

from prediction import StudentPerformancePredictor
from utils import css, load_dataset


st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
st.markdown(f"<style>{css()}</style>", unsafe_allow_html=True)
df = load_dataset()
engine = StudentPerformancePredictor()

st.title("Performance Dashboard")
metrics = st.columns(4)
metrics[0].metric("Average Project", f"{df['Project_Score'].mean():.1f}%")
metrics[1].metric("Average Practical", f"{df['Practical_Score'].mean():.1f}%")
metrics[2].metric("Average Risk", df["Risk_Level"].mode().iloc[0])
metrics[3].metric("Best Model", engine.bundle.model_name)

left, right = st.columns(2)
with left:
    st.plotly_chart(px.imshow(df.select_dtypes("number").corr(), title="Correlation Heatmap", template="plotly_dark"), use_container_width=True)
    chart = alt.Chart(df.sample(min(1000, len(df)), random_state=42)).mark_circle(size=55, opacity=0.65).encode(
        x="Study_Hours",
        y="Final_Exam",
        color="Risk_Level",
        tooltip=["Name", "Attendance", "Final_Grade"],
    ).properties(title="Study Hours and Performance")
    st.altair_chart(chart, use_container_width=True)
with right:
    st.plotly_chart(px.box(df, x="Class", y="Final_Exam", color="Class", title="Class-wise Performance", template="plotly_dark"), use_container_width=True)
    st.plotly_chart(px.histogram(df, x="Attendance", nbins=30, color="Risk_Level", title="Attendance Distribution", template="plotly_dark"), use_container_width=True)

st.subheader("Rankings")
ranked = df.sort_values("Final_Exam", ascending=False)
tabs = st.tabs(["Top Performers", "Weak Students", "Department Statistics"])
tabs[0].dataframe(ranked[["Student_ID", "Name", "Class", "Section", "Final_Exam", "Final_Grade"]].head(25), use_container_width=True, hide_index=True)
tabs[1].dataframe(ranked[["Student_ID", "Name", "Class", "Section", "Final_Exam", "Risk_Level"]].tail(25), use_container_width=True, hide_index=True)
tabs[2].dataframe(df.groupby(["Class", "Section"]).agg(Students=("Student_ID", "count"), Marks=("Final_Exam", "mean"), Attendance=("Attendance", "mean")).round(2), use_container_width=True)
