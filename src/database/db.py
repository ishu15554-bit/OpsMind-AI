
# SQLite database connection and prediction logging

import sqlite3
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Database file location
DB_PATH = BASE_DIR / "data" / "processed" / "opsmind.db"


# Create database and prediction history table
def init_db():

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                input_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()


# Save one prediction
def save_prediction(model_name, input_text, prediction):

    with sqlite3.connect(DB_PATH) as conn:

        conn.execute("""
            INSERT INTO prediction_history
            (model_name, input_text, prediction)
            VALUES (?, ?, ?)
        """, (model_name, input_text, str(prediction)))

        conn.commit()


# Fetch recent prediction history
def get_prediction_history(limit=100):

    with sqlite3.connect(DB_PATH) as conn:

        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT id, model_name, input_text, prediction, created_at
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]


# Initialize database when module is imported
init_db()
