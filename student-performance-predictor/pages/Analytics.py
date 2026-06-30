"""Student analytics and explainable AI page."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from prediction import StudentPerformancePredictor
from utils import css, generate_recommendations, load_dataset


st.set_page_config(page_title="Analytics", page_icon="🧠", layout="wide")
st.markdown(f"<style>{css()}</style>", unsafe_allow_html=True)
df = load_dataset()
engine = StudentPerformancePredictor()

st.title("Student Analytics and Explainable AI")
student_id = st.selectbox("Select Student", df["Student_ID"].head(1000))
student = df[df["Student_ID"] == student_id].iloc[0]

profile, charts = st.columns((0.8, 1.2))
with profile:
    st.subheader(student["Name"])
    st.write(f"Class {student['Class']} - Section {student['Section']}")
    st.metric("Final Marks", f"{student['Final_Exam']}%")
    st.metric("Grade", student["Final_Grade"])
    st.metric("Risk", student["Risk_Level"])
    st.subheader("Improvement Suggestions")
    for rec in generate_recommendations(student.to_dict()):
        st.info(rec)

with charts:
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=[
            student["Attendance"],
            student["Assignments"],
            student["Internal_Marks"],
            student["Project_Score"],
            student["Practical_Score"],
            student["Participation"],
        ],
        theta=["Attendance", "Assignments", "Internal", "Project", "Practical", "Participation"],
        fill="toself",
        name=student["Name"],
    ))
    radar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True, range=[0, 100])), title="Subject-wise Radar")
    st.plotly_chart(radar, use_container_width=True)

    trend = df[(df["Class"] == student["Class"]) & (df["Section"] == student["Section"])].sample(min(60, len(df)), random_state=7).copy()
    trend["Sequence"] = range(1, len(trend) + 1)
    st.plotly_chart(px.line(trend, x="Sequence", y=["Attendance", "Final_Exam"], title="Peer Trend Forecast Proxy", template="plotly_dark"), use_container_width=True)

st.subheader("Explainable AI")
importance = engine.feature_importance().head(15)
st.plotly_chart(px.bar(importance, x="Importance", y="Feature", orientation="h", title="Global Feature Importance", template="plotly_dark"), use_container_width=True)
st.caption("Install SHAP to enable native SHAP summary and waterfall plots. The app uses model feature importance as a robust fallback.")
