"""
Prediction helpers for TrueBot.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict

from joblib import load

from config import DATA_PATH, MODEL_DIR
from modules.preprocessing import preprocess_text
from modules.train_model import main as train_main

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_artifacts(model_path: Path, vectorizer_path: Path):
    """Load classifier and vectorizer."""
    if not model_path.exists() or not vectorizer_path.exists():
        LOGGER.info("Model artifacts missing. Training new model...")
        train_main(DATA_PATH, MODEL_DIR)
    classifier = load(model_path)
    vectorizer = load(vectorizer_path)
    return classifier, vectorizer


def predict_label(text: str, model_path: Path, vectorizer_path: Path) -> Dict[str, str]:
    """
    Predict whether text is fake or real.
    """
    classifier, vectorizer = load_artifacts(model_path, vectorizer_path)
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])
    proba = classifier.predict_proba(vectorized)[0]
    label = "Real" if proba[1] >= 0.5 else "Fake"
    confidence = max(proba) * 100
    LOGGER.debug("Prediction label=%s confidence=%.2f", label, confidence)
    return {"label": label, "confidence": round(confidence, 2)}

