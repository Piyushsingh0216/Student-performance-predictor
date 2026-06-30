"""Reusable utility functions for data, UI, reports, and logging."""

from __future__ import annotations

import base64
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ASSET_DIR,
    DATASET_PATH,
    GRADE_BANDS,
    LOG_PATH,
    MIN_DATASET_ROWS,
    REPORT_DIR,
    RISK_COLORS,
    ensure_directories,
)


FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
    "Ayaan", "Krishna", "Ishaan", "Ananya", "Diya", "Ira", "Myra",
    "Aadhya", "Avni", "Saanvi", "Kiara", "Riya", "Meera",
]
LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Singh", "Patel", "Rao", "Mehta",
    "Nair", "Iyer", "Khan", "Das", "Mishra", "Joshi", "Bose",
]


def setup_logging() -> logging.Logger:
    """Configure a rotating-style file logger without external dependencies."""
    ensure_directories()
    logger = logging.getLogger("student_performance")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
        )
        file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


LOGGER = setup_logging()


def grade_from_score(score: float) -> str:
    """Convert a percentage score into a grade band."""
    score = float(np.clip(score, 0, 100))
    for threshold, grade in GRADE_BANDS:
        if score >= threshold:
            return grade
    return "F"


def performance_from_score(score: float) -> str:
    """Return a human-friendly performance level."""
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Average"
    return "Needs Improvement"


def risk_from_signals(score: float, attendance: float, stress: float) -> str:
    """Compute academic risk from marks, attendance, and stress."""
    risk_points = 0
    if score < 40:
        risk_points += 3
    elif score < 55:
        risk_points += 2
    elif score < 70:
        risk_points += 1
    if attendance < 60:
        risk_points += 3
    elif attendance < 75:
        risk_points += 2
    elif attendance < 85:
        risk_points += 1
    if stress >= 8:
        risk_points += 2
    elif stress >= 6:
        risk_points += 1
    if risk_points >= 6:
        return "Critical Risk"
    if risk_points >= 4:
        return "High Risk"
    if risk_points >= 2:
        return "Medium Risk"
    return "Low Risk"


def attendance_category(attendance: float) -> str:
    """Categorize attendance for dashboard badges."""
    if attendance >= 90:
        return "Excellent"
    if attendance >= 75:
        return "Satisfactory"
    if attendance >= 60:
        return "Warning"
    return "Critical"


def generate_recommendations(payload: dict[str, Any]) -> list[str]:
    """Generate dynamic recommendations from prediction inputs and outputs."""
    recs: list[str] = []
    if float(payload.get("Attendance", 100)) < 75:
        recs.append("Raise attendance above 75% to avoid academic risk flags.")
    if float(payload.get("Study_Hours", 0)) < 3:
        recs.append("Increase focused study time to at least 3-4 hours per day.")
    if float(payload.get("Assignments", 100)) < 70:
        recs.append("Complete assignments earlier and revise feedback before exams.")
    if float(payload.get("Stress_Level", 0)) >= 7:
        recs.append("Use shorter study sprints, breaks, and counselling support to reduce stress.")
    if float(payload.get("Sleep_Hours", 8)) < 6:
        recs.append("Improve sleep to 7-8 hours for better memory and exam performance.")
    if float(payload.get("Previous_Marks", 100)) < 60:
        recs.append("Practice previous-year papers and build a topic-wise error log.")
    if float(payload.get("Participation", 100)) < 60:
        recs.append("Participate more in class discussions and doubt-clearing sessions.")
    if not recs:
        recs.extend([
            "Maintain the current learning routine and attempt timed mock tests weekly.",
            "Use advanced practice sets to convert strong performance into top ranking.",
        ])
    return recs


def generate_synthetic_dataset(rows: int = MIN_DATASET_ROWS, path: Path = DATASET_PATH) -> pd.DataFrame:
    """Create a realistic synthetic student dataset."""
    ensure_directories()
    rng = np.random.default_rng(42)
    classes = rng.integers(6, 13, rows)
    gender = rng.choice(["Female", "Male", "Other"], rows, p=[0.48, 0.49, 0.03])
    attendance = np.clip(rng.normal(82, 12, rows), 35, 100)
    study_hours = np.clip(rng.normal(3.7, 1.6, rows), 0.5, 9.5)
    assignments = np.clip(rng.normal(76, 15, rows), 20, 100)
    previous = np.clip(rng.normal(70, 16, rows), 18, 99)
    internal = np.clip(previous * 0.55 + assignments * 0.25 + rng.normal(12, 7, rows), 15, 100)
    project = np.clip(rng.normal(74, 15, rows) + study_hours * 2, 20, 100)
    practical = np.clip(rng.normal(76, 14, rows) + project * 0.08, 20, 100)
    behavior = np.clip(rng.normal(78, 12, rows), 25, 100)
    participation = np.clip(rng.normal(70, 18, rows), 15, 100)
    sleep = np.clip(rng.normal(7.0, 1.1, rows), 3.5, 10)
    stress = np.clip(rng.normal(5.1, 2.0, rows), 1, 10)
    exam_prep = np.clip(rng.normal(67, 18, rows) + study_hours * 3 - stress * 1.8, 5, 100)
    internet = rng.choice(["Yes", "No"], rows, p=[0.84, 0.16])
    extra = rng.choice(["Yes", "No"], rows, p=[0.56, 0.44])
    parent_ed = rng.choice(
        ["High School", "Diploma", "Graduate", "Postgraduate", "Doctorate"],
        rows,
        p=[0.25, 0.18, 0.35, 0.18, 0.04],
    )
    income = np.clip(rng.lognormal(mean=10.8, sigma=0.45, size=rows), 12000, 220000)
    income_factor = np.interp(income, (income.min(), income.max()), (0, 4))
    internet_boost = np.where(internet == "Yes", 2.0, -2.5)
    extra_boost = np.where(extra == "Yes", 1.3, -0.4)
    noise = rng.normal(0, 1.2, rows)
    final = (
        previous * 0.24
        + internal * 0.18
        + assignments * 0.10
        + attendance * 0.12
        + project * 0.08
        + practical * 0.08
        + behavior * 0.04
        + participation * 0.04
        + exam_prep * 0.13
        + study_hours * 1.8
        + sleep * 0.9
        - stress * 1.5
        + internet_boost
        + extra_boost
        + income_factor
        - 29.0
        + noise
    )
    final = np.clip(final, 0, 100).round(2)
    names = [
        f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        for _ in range(rows)
    ]
    df = pd.DataFrame({
        "Student_ID": [f"STU{100000 + i}" for i in range(rows)],
        "Name": names,
        "Gender": gender,
        "Age": np.clip(classes + rng.integers(5, 8, rows), 11, 19),
        "Class": classes,
        "Section": rng.choice(list("ABCD"), rows),
        "Attendance": attendance.round(2),
        "Study_Hours": study_hours.round(2),
        "Assignments": assignments.round(2),
        "Previous_Marks": previous.round(2),
        "Internal_Marks": internal.round(2),
        "Project_Score": project.round(2),
        "Practical_Score": practical.round(2),
        "Behavior_Score": behavior.round(2),
        "Participation": participation.round(2),
        "Internet_Access": internet,
        "Family_Income": income.round(0),
        "Parent_Education": parent_ed,
        "Sleep_Hours": sleep.round(2),
        "Extra_Curricular": extra,
        "Stress_Level": stress.round(2),
        "Exam_Preparation": exam_prep.round(2),
        "Final_Exam": final,
    })
    df["Final_Grade"] = df["Final_Exam"].apply(grade_from_score)
    df["Performance"] = df["Final_Exam"].apply(performance_from_score)
    df["Risk_Level"] = [
        risk_from_signals(score, att, st)
        for score, att, st in zip(df["Final_Exam"], df["Attendance"], df["Stress_Level"])
    ]
    df["Pass_Fail"] = np.where(df["Final_Exam"] >= 40, "Pass", "Fail")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Generated synthetic dataset with %s rows at %s", len(df), path)
    return df


def load_dataset() -> pd.DataFrame:
    """Load the dataset, creating it when absent or undersized."""
    if not DATASET_PATH.exists():
        return generate_synthetic_dataset()
    df = pd.read_csv(DATASET_PATH)
    if len(df) < MIN_DATASET_ROWS:
        return generate_synthetic_dataset()
    return df


def clean_student_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Basic validation, imputation, and clipping for student data."""
    cleaned = df.copy()
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        lower, upper = cleaned[col].quantile([0.01, 0.99])
        cleaned[col] = cleaned[col].clip(lower, upper)
    for col in cleaned.select_dtypes(exclude=[np.number]).columns:
        cleaned[col] = cleaned[col].fillna(cleaned[col].mode().iloc[0] if not cleaned[col].mode().empty else "Unknown")
    return cleaned


def css() -> str:
    """Read the custom stylesheet."""
    css_path = Path(__file__).resolve().parent / "css" / "style.css"
    return css_path.read_text(encoding="utf-8") if css_path.exists() else ""


def image_to_base64(path: Path) -> str:
    """Encode an image for Streamlit HTML blocks."""
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def risk_badge(label: str) -> str:
    """Return a colored HTML badge for risk labels."""
    color = RISK_COLORS.get(label, "#64748b")
    return f"<span class='risk-badge' style='background:{color}'>{label}</span>"


def generate_pdf_report(student: dict[str, Any], prediction: dict[str, Any]) -> Path:
    """Generate a professional PDF report, with a text fallback if ReportLab is absent."""
    ensure_directories()
    filename = f"report_{student.get('Student_ID', 'student')}_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    output = REPORT_DIR / filename
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        elements: list[Any] = [
            Paragraph("AI Student Performance Report", styles["Title"]),
            Paragraph(datetime.now().strftime("Generated on %d %B %Y, %I:%M %p"), styles["Normal"]),
            Spacer(1, 16),
        ]
        rows = [["Field", "Value"]]
        for key in ["Student_ID", "Name", "Class", "Section"]:
            rows.append([key.replace("_", " "), str(student.get(key, "N/A"))])
        for key in ["Final Percentage", "Expected Grade", "Pass / Fail", "Risk Level", "Confidence Score"]:
            rows.append([key, str(prediction.get(key, "N/A"))])
        table = Table(rows, colWidths=[180, 320])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 18))
        elements.append(Paragraph("Recommendations", styles["Heading2"]))
        for rec in prediction.get("Recommendations", []):
            elements.append(Paragraph(f"- {rec}", styles["Normal"]))
        doc.build(elements)
    except Exception as exc:  # pragma: no cover - fallback for missing optional dependency
        LOGGER.exception("PDF generation fallback used: %s", exc)
        output.write_text(
            "AI Student Performance Report\n\n"
            f"Student: {student}\n\nPrediction: {prediction}\n",
            encoding="utf-8",
        )
    LOGGER.info("Generated report at %s", output)
    return output


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value into a finite float."""
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default
