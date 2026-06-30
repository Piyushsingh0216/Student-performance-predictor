# AI-Powered Student Performance Predictor

A production-style machine learning portfolio project that predicts student performance, detects academic risk, explains model behavior, stores prediction history, and generates professional reports.

## Features

- Final percentage, grade, pass/fail, performance level, risk level, probability, and confidence prediction
- Realistic synthetic dataset generation with 5,000+ records
- Complete ML workflow: cleaning, outlier handling, encoding, scaling, train/test split, cross validation, hyperparameter tuning, model comparison, and artifact persistence
- Model comparison across Random Forest, Gradient Boosting, Extra Trees, Decision Tree, and optional XGBoost, LightGBM, CatBoost
- Streamlit multipage UI with custom CSS, glassmorphism cards, responsive dashboards, and dark/light mode
- Dashboard metrics, filters, ranking, risk distribution, grade distribution, heatmaps, attendance analysis, and feature importance
- CSV upload, batch prediction, CSV/Excel export
- SQLite persistence for students, predictions, reports, and action history
- PDF report generation with recommendations
- Explainable AI via SHAP when installed, with feature importance fallback
- Dynamic recommendations engine, attendance warnings, study planning cues, ranking, weak-student detection, and history views

## Project Structure

```text
student-performance-predictor/
├── app.py
├── train_model.py
├── prediction.py
├── database.py
├── utils.py
├── config.py
├── models/
├── dataset/
├── reports/
├── assets/
├── pages/
├── css/
├── requirements.txt
├── README.md
└── screenshots/
```

## Installation

```bash
cd student-performance-predictor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

If Windows says `streamlit` is not recognized, use the `python -m streamlit` command above. It works even when the user-level Python Scripts directory is not on PATH.

## Dataset

If `dataset/student_data.csv` is missing or has fewer than 5,000 rows, the project automatically creates a realistic synthetic dataset with demographics, attendance, study habits, assessment scores, behavioral signals, final grade, performance category, risk level, and pass/fail status.

## Machine Learning Workflow

1. Generate or load dataset
2. Clean missing values and clip outliers
3. Encode categorical features and scale numeric features
4. Split train/test data
5. Compare multiple regressors
6. Tune Extra Trees with grid search and cross validation
7. Train pass/fail and risk classifiers
8. Save `best_model.pkl`, `scaler.pkl`, and `encoder.pkl`

The synthetic target is designed from realistic academic drivers, so tree ensembles typically achieve high predictive accuracy on local training.

## Application Pages

- `app.py`: executive dashboard and admin overview
- `pages/Prediction.py`: individual and batch prediction
- `pages/Dashboard.py`: deeper analytics, correlations, rankings
- `pages/Analytics.py`: individual profile, radar chart, explainability
- `pages/Reports.py`: exports and database records
- `pages/About.py`: architecture and model metrics

## Architecture

```text
Streamlit UI
  -> Prediction Service
  -> Trained ML Bundle
  -> SQLite Database
  -> PDF/CSV/Excel Reports

train_model.py
  -> Dataset Generation
  -> Cleaning
  -> Encoding and Scaling
  -> Model Comparison
  -> Persisted Artifacts
```

## Screenshots

Add screenshots to the `screenshots/` folder after running the app locally.

## Future Scope

- Authentication and role-based access
- Real school ERP integration
- Longitudinal semester-level training data
- Notification emails for high-risk students
- Cloud deployment with scheduled retraining

## License

MIT License. Use freely for learning, portfolio, and placement demonstrations.
