"""Central configuration for the Student Performance Predictor."""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
ASSET_DIR = BASE_DIR / "assets"
CSS_DIR = BASE_DIR / "css"
LOG_DIR = BASE_DIR / "logs"

DATASET_PATH = DATASET_DIR / "student_data.csv"
DATABASE_PATH = BASE_DIR / "student_performance.db"
MODEL_PATH = MODEL_DIR / "best_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
LOG_PATH = LOG_DIR / "app.log"

RANDOM_STATE = 42
MIN_DATASET_ROWS = 5000
TEST_SIZE = 0.2

FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Class",
    "Attendance",
    "Study_Hours",
    "Assignments",
    "Previous_Marks",
    "Internal_Marks",
    "Project_Score",
    "Practical_Score",
    "Behavior_Score",
    "Participation",
    "Internet_Access",
    "Family_Income",
    "Parent_Education",
    "Sleep_Hours",
    "Extra_Curricular",
    "Stress_Level",
    "Exam_Preparation",
]

NUMERIC_FEATURES = [
    "Age",
    "Class",
    "Attendance",
    "Study_Hours",
    "Assignments",
    "Previous_Marks",
    "Internal_Marks",
    "Project_Score",
    "Practical_Score",
    "Behavior_Score",
    "Participation",
    "Family_Income",
    "Sleep_Hours",
    "Stress_Level",
    "Exam_Preparation",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Internet_Access",
    "Parent_Education",
    "Extra_Curricular",
]

TARGET_REGRESSION = "Final_Exam"
TARGET_PASS = "Pass_Fail"
TARGET_RISK = "Risk_Level"

GRADE_BANDS = [
    (90, "A+"),
    (80, "A"),
    (70, "B+"),
    (60, "B"),
    (50, "C"),
    (40, "D"),
    (0, "F"),
]

RISK_COLORS = {
    "Low Risk": "#16a34a",
    "Medium Risk": "#eab308",
    "High Risk": "#f97316",
    "Critical Risk": "#dc2626",
}

PERFORMANCE_COLORS = {
    "Excellent": "#10b981",
    "Good": "#38bdf8",
    "Average": "#f59e0b",
    "Needs Improvement": "#ef4444",
}


def ensure_directories() -> None:
    """Create all runtime directories used by the project."""
    for path in [DATASET_DIR, MODEL_DIR, REPORT_DIR, ASSET_DIR, CSS_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
