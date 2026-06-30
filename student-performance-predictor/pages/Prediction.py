"""Interactive prediction page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database import StudentDatabase
from prediction import StudentPerformancePredictor
from utils import css, generate_pdf_report, risk_badge


st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")
st.markdown(f"<style>{css()}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def predictor():
    return StudentPerformancePredictor()


engine = predictor()
db = StudentDatabase()
st.title("Student Performance Prediction")

with st.form("prediction_form"):
    left, mid, right = st.columns(3)
    student_id = left.text_input("Student ID", "STU-DEMO-001")
    name = mid.text_input("Name", "Demo Student")
    gender = right.selectbox("Gender", ["Female", "Male", "Other"])
    age = left.slider("Age", 11, 20, 16)
    student_class = mid.slider("Class", 6, 12, 10)
    section = right.selectbox("Section", list("ABCD"))
    attendance = left.slider("Attendance", 0, 100, 82)
    study_hours = mid.slider("Study Hours", 0.0, 10.0, 3.5, 0.25)
    previous = right.slider("Previous Marks", 0, 100, 72)
    assignments = left.slider("Assignments", 0, 100, 78)
    project = mid.slider("Project Score", 0, 100, 80)
    practical = right.slider("Practical Score", 0, 100, 76)
    internal = left.slider("Internal Marks", 0, 100, 74)
    participation = mid.slider("Participation", 0, 100, 70)
    behavior = right.slider("Behavior", 0, 100, 78)
    sleep = left.slider("Sleep Hours", 3.0, 10.0, 7.0, 0.25)
    stress = mid.slider("Stress Level", 1, 10, 5)
    exam_prep = right.slider("Exam Preparation", 0, 100, 70)
    internet = left.selectbox("Internet Access", ["Yes", "No"])
    parent_ed = mid.selectbox("Parent Education", ["High School", "Diploma", "Graduate", "Postgraduate", "Doctorate"])
    income = right.number_input("Family Income", min_value=0, max_value=500000, value=65000, step=5000)
    extra = left.selectbox("Extra Curricular", ["Yes", "No"])
    submitted = st.form_submit_button("Predict Performance", use_container_width=True)

if submitted:
    payload = {
        "Student_ID": student_id,
        "Name": name,
        "Gender": gender,
        "Age": age,
        "Class": student_class,
        "Section": section,
        "Attendance": attendance,
        "Study_Hours": study_hours,
        "Assignments": assignments,
        "Previous_Marks": previous,
        "Internal_Marks": internal,
        "Project_Score": project,
        "Practical_Score": practical,
        "Behavior_Score": behavior,
        "Participation": participation,
        "Internet_Access": internet,
        "Family_Income": income,
        "Parent_Education": parent_ed,
        "Sleep_Hours": sleep,
        "Extra_Curricular": extra,
        "Stress_Level": stress,
        "Exam_Preparation": exam_prep,
    }
    with st.spinner("Analyzing academic signals..."):
        prediction = engine.predict(payload)
        db.upsert_student(payload)
        db.store_prediction(student_id, prediction)
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Predicted Percentage", f"{prediction['Final Percentage']}%")
    b.metric("Expected Grade", prediction["Expected Grade"])
    c.metric("Pass / Fail", prediction["Pass / Fail"])
    d.markdown(risk_badge(prediction["Risk Level"]), unsafe_allow_html=True)
    e, f, g = st.columns(3)
    e.metric("Performance", prediction["Performance Level"])
    f.metric("Confidence", f"{prediction['Confidence Score']}%")
    g.metric("Probability", f"{prediction['Probability Score']}%")
    st.markdown("</div>", unsafe_allow_html=True)
    st.subheader("Personalized Recommendations")
    for item in prediction["Recommendations"]:
        st.success(item)
    report_path = generate_pdf_report(payload, prediction)
    db.store_report(student_id, str(report_path))
    st.download_button("Download PDF Report", report_path.read_bytes(), file_name=report_path.name, mime="application/pdf")

st.divider()
st.subheader("CSV Upload and Batch Prediction")
uploaded = st.file_uploader("Upload student CSV", type=["csv"])
if uploaded:
    data = pd.read_csv(uploaded)
    try:
        scored = engine.predict_batch(data)
        st.dataframe(scored, use_container_width=True, hide_index=True)
        st.download_button("Export Predictions CSV", scored.to_csv(index=False).encode("utf-8"), "student_predictions.csv", "text/csv")
        buffer = None
        try:
            import io
            buffer = io.BytesIO()
            scored.to_excel(buffer, index=False)
            st.download_button("Export Predictions Excel", buffer.getvalue(), "student_predictions.xlsx")
        except Exception:
            st.caption("Install openpyxl to enable Excel export.")
    except Exception as exc:
        st.error(f"Could not process the file: {exc}")
