"""Reports, exports, and admin operations."""

from __future__ import annotations

import streamlit as st

from database import StudentDatabase
from utils import css, load_dataset


st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")
st.markdown(f"<style>{css()}</style>", unsafe_allow_html=True)
db = StudentDatabase()
df = load_dataset()

st.title("Reports and Admin Center")
st.subheader("Dataset Export")
st.download_button("Download Sample Dataset CSV", df.to_csv(index=False).encode("utf-8"), "student_data.csv", "text/csv")

history = db.prediction_history()
st.subheader("Prediction History")
if history.empty:
    st.info("No prediction history stored yet.")
else:
    risk = st.multiselect("Filter by risk", sorted(history["Risk Level"].dropna().unique()), default=sorted(history["Risk Level"].dropna().unique()))
    view = history[history["Risk Level"].isin(risk)] if risk else history
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button("Export History CSV", view.to_csv(index=False).encode("utf-8"), "prediction_history.csv", "text/csv")

st.subheader("Student Records")
students = db.students_df()
if students.empty:
    st.caption("Students saved from the Prediction page will appear here.")
else:
    query = st.text_input("Search saved students")
    if query:
        q = query.lower()
        students = students[students.astype(str).apply(lambda row: row.str.lower().str.contains(q).any(), axis=1)]
    st.dataframe(students, use_container_width=True, hide_index=True)
    delete_id = st.text_input("Delete Student ID")
    if st.button("Delete Student") and delete_id:
        db.delete_student(delete_id)
        st.success(f"Deleted {delete_id}")
        st.rerun()
