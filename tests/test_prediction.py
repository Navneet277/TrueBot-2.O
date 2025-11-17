"""Tests for predictor module."""

from pathlib import Path

import pytest

from config import DATA_PATH, MODEL_DIR, MODEL_PATH, VECTORIZER_PATH
from modules import predictor
from modules.train_model import main as train_main


@pytest.fixture(scope="session", autouse=True)
def ensure_model():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        train_main(DATA_PATH, MODEL_DIR)


def test_predict_label_returns_confidence(ensure_model):
    sample_text = "Government confirms the budget has been approved."
    result = predictor.predict_label(sample_text, MODEL_PATH, VECTORIZER_PATH)
    assert result["label"] in {"Real", "Fake"}
    assert 0 <= result["confidence"] <= 100


