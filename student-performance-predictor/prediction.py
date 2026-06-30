"""Prediction service used by the Streamlit UI and batch scoring."""

from __future__ import annotations

from typing import Any

import importlib.util
import joblib
import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, MODEL_PATH
from train_model import ModelBundle, train
from utils import (
    attendance_category,
    generate_recommendations,
    grade_from_score,
    performance_from_score,
    risk_from_signals,
    safe_float,
    LOGGER,
)


class StudentPerformancePredictor:
    """High-level inference API with validation and explainability helpers."""

    def __init__(self) -> None:
        self.bundle = self._load_or_train()

    def _load_or_train(self) -> ModelBundle:
        if not MODEL_PATH.exists():
            LOGGER.info("Model artifact missing; starting training.")
            return train()
        return joblib.load(MODEL_PATH)

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize one student record."""
        normalized = dict(payload)
        limits = {
            "Age": (10, 22),
            "Class": (1, 12),
            "Attendance": (0, 100),
            "Study_Hours": (0, 12),
            "Assignments": (0, 100),
            "Previous_Marks": (0, 100),
            "Internal_Marks": (0, 100),
            "Project_Score": (0, 100),
            "Practical_Score": (0, 100),
            "Behavior_Score": (0, 100),
            "Participation": (0, 100),
            "Family_Income": (0, 500000),
            "Sleep_Hours": (0, 12),
            "Stress_Level": (1, 10),
            "Exam_Preparation": (0, 100),
        }
        for col, (low, high) in limits.items():
            normalized[col] = float(np.clip(safe_float(normalized.get(col), low), low, high))
        defaults = {
            "Gender": "Female",
            "Internet_Access": "Yes",
            "Parent_Education": "Graduate",
            "Extra_Curricular": "Yes",
        }
        for col, default in defaults.items():
            normalized[col] = normalized.get(col) or default
        return normalized

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Predict marks, grade, pass/fail, risk, and recommendations."""
        clean = self.validate(payload)
        frame = pd.DataFrame([clean])[FEATURE_COLUMNS]
        percentage = float(np.clip(self.bundle.regressor.predict(frame)[0], 0, 100))
        pass_label = str(self.bundle.pass_classifier.predict(frame)[0])
        risk_label = str(self.bundle.risk_classifier.predict(frame)[0])
        if hasattr(self.bundle.pass_classifier.named_steps["model"], "predict_proba"):
            pass_proba = self.bundle.pass_classifier.predict_proba(frame)[0]
            pass_conf = float(np.max(pass_proba) * 100)
        else:
            pass_conf = float(max(percentage, 100 - percentage))
        if hasattr(self.bundle.risk_classifier.named_steps["model"], "predict_proba"):
            risk_conf = float(np.max(self.bundle.risk_classifier.predict_proba(frame)[0]) * 100)
        else:
            risk_conf = 82.0
        risk_rule = risk_from_signals(percentage, clean["Attendance"], clean["Stress_Level"])
        if risk_rule in {"High Risk", "Critical Risk"} and clean["Attendance"] < 75:
            risk_label = risk_rule
        result = {
            "Final Percentage": round(percentage, 2),
            "Expected Grade": grade_from_score(percentage),
            "Pass / Fail": pass_label if percentage >= 38 else "Fail",
            "Performance Level": performance_from_score(percentage),
            "Risk Level": risk_label,
            "Probability Score": round(pass_conf, 2),
            "Confidence Score": round((pass_conf * 0.55 + risk_conf * 0.45), 2),
            "Attendance Category": attendance_category(clean["Attendance"]),
            "Attendance Impact": round((clean["Attendance"] - 75) * 0.18, 2),
            "Recommendations": generate_recommendations({**clean, "Predicted_Percentage": percentage}),
        }
        LOGGER.info("Prediction generated: %s", result)
        return result

    def predict_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Score an uploaded CSV and return appended predictions."""
        records = []
        for _, row in frame.iterrows():
            payload = {col: row.get(col) for col in FEATURE_COLUMNS}
            pred = self.predict(payload)
            records.append(pred)
        return pd.concat([frame.reset_index(drop=True), pd.DataFrame(records)], axis=1)

    def feature_importance(self) -> pd.DataFrame:
        """Return model feature importance aggregated to readable names."""
        pipe = self.bundle.regressor
        model = pipe.named_steps["model"]
        pre = pipe.named_steps["preprocessor"]
        if not hasattr(model, "feature_importances_"):
            return pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": np.ones(len(FEATURE_COLUMNS)) / len(FEATURE_COLUMNS)})
        names = pre.get_feature_names_out()
        values = model.feature_importances_
        grouped: dict[str, float] = {}
        for name, value in zip(names, values):
            clean = name.split("__", 1)[-1].split("_", 1)[0]
            grouped[clean] = grouped.get(clean, 0.0) + float(value)
        return pd.DataFrame(
            [{"Feature": key, "Importance": value} for key, value in grouped.items()]
        ).sort_values("Importance", ascending=False)

    def shap_values(self, sample: pd.DataFrame) -> Any:
        """Compute SHAP values when the optional package is installed."""
        if not importlib.util.find_spec("shap"):
            return None
        import shap
        transformed = self.bundle.regressor.named_steps["preprocessor"].transform(sample[FEATURE_COLUMNS])
        model = self.bundle.regressor.named_steps["model"]
        explainer = shap.Explainer(model)
        return explainer(transformed)
