"""SQLite persistence layer for students, predictions, reports, and actions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATABASE_PATH, ensure_directories
from utils import LOGGER


class StudentDatabase:
    """Small production-style repository around SQLite."""

    def __init__(self, db_path: Path = DATABASE_PATH) -> None:
        self.db_path = db_path
        ensure_directories()
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT,
                    class INTEGER,
                    section TEXT,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    prediction TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    report_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )
        LOGGER.info("Database initialized at %s", self.db_path)

    def upsert_student(self, student: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO students(student_id, name, class, section, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(student_id) DO UPDATE SET
                    name=excluded.name,
                    class=excluded.class,
                    section=excluded.section,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (
                    student.get("Student_ID"),
                    student.get("Name"),
                    student.get("Class"),
                    student.get("Section"),
                    json.dumps(student, default=str),
                    datetime.utcnow().isoformat(),
                ),
            )
        self.log_action("upsert_student", str(student.get("Student_ID")))

    def delete_student(self, student_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        self.log_action("delete_student", student_id)

    def store_prediction(self, student_id: str, prediction: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO predictions(student_id, prediction, created_at) VALUES (?, ?, ?)",
                (student_id, json.dumps(prediction, default=str), datetime.utcnow().isoformat()),
            )
        self.log_action("prediction", student_id)

    def store_report(self, student_id: str, report_path: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO reports(student_id, report_path, created_at) VALUES (?, ?, ?)",
                (student_id, report_path, datetime.utcnow().isoformat()),
            )
        self.log_action("report", report_path)

    def log_action(self, action: str, detail: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO history(action, detail, created_at) VALUES (?, ?, ?)",
                (action, detail, datetime.utcnow().isoformat()),
            )

    def students_df(self) -> pd.DataFrame:
        with self.connect() as conn:
            rows = conn.execute("SELECT payload FROM students ORDER BY updated_at DESC").fetchall()
        data = [json.loads(row["payload"]) for row in rows]
        return pd.DataFrame(data)

    def prediction_history(self) -> pd.DataFrame:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT student_id, prediction, created_at FROM predictions ORDER BY created_at DESC"
            ).fetchall()
        records = []
        for row in rows:
            item = json.loads(row["prediction"])
            item["Student_ID"] = row["student_id"]
            item["Timestamp"] = row["created_at"]
            records.append(item)
        return pd.DataFrame(records)
