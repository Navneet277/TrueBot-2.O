"""Central configuration for TrueBot application."""

from pathlib import Path
from datetime import timedelta

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "news.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
DB_PATH = BASE_DIR / "database" / "truebot.db"
LOG_PATH = BASE_DIR / "truebot.log"

APP_CONFIG = {
    "SECRET_KEY": "replace-this-with-env-secret",
    "PERMANENT_SESSION_LIFETIME": timedelta(hours=12),
    "JSON_SORT_KEYS": False,
}


