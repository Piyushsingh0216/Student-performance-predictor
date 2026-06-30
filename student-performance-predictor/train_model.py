"""Train and persist the ML pipeline for student performance prediction."""

from __future__ import annotations

import importlib.util
import pickle
import sys
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from config import (
    CATEGORICAL_FEATURES,
    DATASET_PATH,
    ENCODER_PATH,
    FEATURE_COLUMNS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    SCALER_PATH,
    TARGET_PASS,
    TARGET_REGRESSION,
    TARGET_RISK,
    TEST_SIZE,
    ensure_directories,
)
from utils import clean_student_frame, generate_synthetic_dataset, load_dataset, LOGGER


if __name__ == "__main__":
    sys.modules["train_model"] = sys.modules[__name__]


@dataclass
class ModelBundle:
    """Persisted model bundle used by prediction.py."""

    regressor: Pipeline
    pass_classifier: Pipeline
    risk_classifier: Pipeline
    metrics: dict[str, Any]
    feature_columns: list[str]
    model_name: str


ModelBundle.__module__ = "train_model"


def _optional_regressors() -> dict[str, Any]:
    models: dict[str, Any] = {
        "Random Forest": RandomForestRegressor(n_estimators=180, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesRegressor(n_estimators=220, random_state=RANDOM_STATE, n_jobs=-1),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=12),
    }
    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=220, max_depth=4, learning_rate=0.06, random_state=RANDOM_STATE)
    if importlib.util.find_spec("lightgbm"):
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(n_estimators=260, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1)
    if importlib.util.find_spec("catboost"):
        from catboost import CatBoostRegressor
        models["CatBoost"] = CatBoostRegressor(iterations=280, learning_rate=0.05, depth=6, random_seed=RANDOM_STATE, verbose=False)
    return models


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _pipeline(model: Any) -> Pipeline:
    return Pipeline([("preprocessor", _preprocessor()), ("model", model)])


def train() -> ModelBundle:
    """Run data preparation, model comparison, tuning, and persistence."""
    ensure_directories()
    df = load_dataset() if DATASET_PATH.exists() else generate_synthetic_dataset()
    fail_rate = float((df[TARGET_PASS] == "Fail").mean()) if TARGET_PASS in df else 0.0
    if df[TARGET_PASS].nunique() < 2 or df[TARGET_RISK].nunique() < 3 or fail_rate < 0.04:
        df = generate_synthetic_dataset()
    df = clean_student_frame(df)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_REGRESSION]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    comparison: dict[str, dict[str, float]] = {}
    best_name = ""
    best_score = -np.inf
    best_model: Pipeline | None = None
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, estimator in _optional_regressors().items():
        pipe = _pipeline(estimator)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        comparison[name] = {"r2": round(float(r2), 4), "mae": round(float(mae), 4)}
        LOGGER.info("%s evaluation: r2=%s mae=%s", name, r2, mae)
        if r2 > best_score:
            best_name, best_score, best_model = name, r2, pipe

    tuned = GridSearchCV(
        _pipeline(ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        {
            "model__n_estimators": [160, 240],
            "model__max_depth": [None, 18],
            "model__min_samples_leaf": [1, 2],
        },
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )
    tuned.fit(X_train, y_train)
    tuned_preds = tuned.predict(X_test)
    tuned_r2 = r2_score(y_test, tuned_preds)
    comparison["Tuned Extra Trees"] = {
        "r2": round(float(tuned_r2), 4),
        "mae": round(float(mean_absolute_error(y_test, tuned_preds)), 4),
    }
    if tuned_r2 > best_score:
        best_name, best_score, best_model = "Tuned Extra Trees", tuned.best_estimator_

    assert best_model is not None

    pass_candidates = {
        "Extra Trees": ExtraTreesClassifier(n_estimators=220, random_state=RANDOM_STATE, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=220, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
    }
    pass_scores: dict[str, float] = {}
    pass_clf: Pipeline | None = None
    best_pass_name = ""
    best_pass_acc = -np.inf
    for clf_name, clf in pass_candidates.items():
        candidate = Pipeline([("preprocessor", _preprocessor()), ("model", clf)])
        candidate.fit(X_train, df.loc[X_train.index, TARGET_PASS])
        acc = accuracy_score(df.loc[X_test.index, TARGET_PASS], candidate.predict(X_test))
        pass_scores[clf_name] = round(float(acc), 4)
        if acc > best_pass_acc:
            best_pass_name, best_pass_acc, pass_clf = clf_name, acc, candidate

    risk_candidates = {
        "Extra Trees": ExtraTreesClassifier(n_estimators=260, random_state=RANDOM_STATE, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=260, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    risk_scores: dict[str, float] = {}
    risk_clf: Pipeline | None = None
    best_risk_name = ""
    best_risk_acc = -np.inf
    for clf_name, clf in risk_candidates.items():
        candidate = Pipeline([("preprocessor", _preprocessor()), ("model", clf)])
        candidate.fit(X_train, df.loc[X_train.index, TARGET_RISK])
        acc = accuracy_score(df.loc[X_test.index, TARGET_RISK], candidate.predict(X_test))
        risk_scores[clf_name] = round(float(acc), 4)
        if acc > best_risk_acc:
            best_risk_name, best_risk_acc, risk_clf = clf_name, acc, candidate

    assert pass_clf is not None and risk_clf is not None
    pass_pred = pass_clf.predict(X_test)
    risk_pred = risk_clf.predict(X_test)

    metrics = {
        "best_regressor": best_name,
        "regression_r2": round(float(best_score), 4),
        "comparison": comparison,
        "best_pass_classifier": best_pass_name,
        "best_risk_classifier": best_risk_name,
        "pass_model_comparison": pass_scores,
        "risk_model_comparison": risk_scores,
        "pass_accuracy": round(float(accuracy_score(df.loc[X_test.index, TARGET_PASS], pass_pred)), 4),
        "risk_accuracy": round(float(accuracy_score(df.loc[X_test.index, TARGET_RISK], risk_pred)), 4),
        "pass_report": classification_report(df.loc[X_test.index, TARGET_PASS], pass_pred, output_dict=True),
    }

    bundle = ModelBundle(
        regressor=best_model,
        pass_classifier=pass_clf,
        risk_classifier=risk_clf,
        metrics=metrics,
        feature_columns=FEATURE_COLUMNS,
        model_name=best_name,
    )
    joblib.dump(bundle, MODEL_PATH)
    preprocessor = best_model.named_steps["preprocessor"]
    joblib.dump(preprocessor.named_transformers_["num"], SCALER_PATH)
    joblib.dump(preprocessor.named_transformers_["cat"], ENCODER_PATH)
    with open(MODEL_PATH.with_suffix(".pickle"), "wb") as fh:
        pickle.dump(bundle, fh)
    LOGGER.info("Saved model bundle at %s with metrics %s", MODEL_PATH, metrics)
    return bundle


if __name__ == "__main__":
    trained = train()
    print(f"Best model: {trained.model_name}")
    print(trained.metrics)
