"""TrueBot Flask application."""

from __future__ import annotations

import logging
import sqlite3

from flask import Flask, jsonify, render_template, request

from config import APP_CONFIG, DB_PATH, MODEL_PATH, VECTORIZER_PATH
from modules.predictor import predict_label

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
app.config.update(APP_CONFIG)


def get_db_connection() -> sqlite3.Connection:
    """Return sqlite connection with row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS manual_inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            prediction TEXT,
            confidence REAL,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


@app.route("/")
def home():
    """Render landing page."""
    return render_template("home.html")


@app.route("/about")
def about():
    """Render about page."""
    return render_template("about.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    """Render detection page."""
    if request.method == "POST":
        text = request.form.get("news_text", "")
        if not text.strip():
            return render_template(
                "detect.html", error="Please enter news text for detection."
            )
        prediction = predict_label(text, MODEL_PATH, VECTORIZER_PATH)
        save_manual_input(text, prediction)
        return render_template("result.html", text=text, prediction=prediction)
    return render_template("detect.html")


@app.route("/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint."""
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    if not text.strip():
        return jsonify({"error": "Text is required"}), 400
    prediction = predict_label(text, MODEL_PATH, VECTORIZER_PATH)
    save_manual_input(text, prediction)
    return jsonify({"text": text, **prediction})


def save_manual_input(text: str, prediction: dict) -> None:
    """Persist manual detection to DB."""
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO manual_inputs(text, prediction, confidence) VALUES (?, ?, ?)",
        (text, prediction["label"], prediction["confidence"]),
    )
    conn.commit()
    conn.close()


@app.context_processor
def inject_globals():
    """Shared template context."""
    return {"app_name": "TrueBot"}


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)


