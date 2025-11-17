"""
Training script for TrueBot models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from modules.preprocessing import build_vectorizer, preprocess_corpus

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_model")


def load_dataset(path: Path) -> pd.DataFrame:
    """Read dataset CSV."""
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")
    return df.dropna(subset=["text", "label"])


def build_models() -> Dict[str, Pipeline]:
    """Instantiate candidate pipelines."""
    vectorizer = build_vectorizer()
    models = {
        "log_reg": Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("tfidf", build_vectorizer()),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = Pipeline(
            [
                ("tfidf", build_vectorizer()),
                (
                    "clf",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        max_depth=6,
                        n_estimators=200,
                        learning_rate=0.1,
                        subsample=0.8,
                    ),
                ),
            ]
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("XGBoost unavailable: %s", exc)
    return models


def train_and_evaluate(df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, float]]:
    """Train candidate models and return best."""
    X = preprocess_corpus(df["text"].tolist())
    y = (df["label"].str.lower() == "real").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scores: Dict[str, float] = {}
    reports: Dict[str, str] = {}
    best_model: Pipeline | None = None
    best_score = -np.inf

    for name, pipeline in build_models().items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        scores[name] = acc
        reports[name] = classification_report(y_test, preds, zero_division=0)
        LOGGER.info("Model %s achieved accuracy %.3f", name, acc)
        if acc > best_score:
            best_score = acc
            best_model = pipeline

    if not best_model:
        raise RuntimeError("No model was successfully trained")

    LOGGER.info("Best model: %s (%.3f)", best_model, best_score)
    LOGGER.info("Reports: \\n%s", json.dumps(reports, indent=2))
    return best_model, scores


def persist_model(pipeline: Pipeline, model_dir: Path) -> None:
    """Save model and vectorizer to disk."""
    from joblib import dump

    model_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["clf"]
    dump(classifier, model_dir / "model.pkl")
    dump(vectorizer, model_dir / "vectorizer.pkl")


def main(dataset_path: Path, model_dir: Path) -> Dict[str, float]:
    """Driver function."""
    df = load_dataset(dataset_path)
    pipeline, scores = train_and_evaluate(df)
    persist_model(pipeline, model_dir)
    return scores


if __name__ == "__main__":
    from config import DATA_PATH, MODEL_DIR

    metrics = main(DATA_PATH, MODEL_DIR)
    print(json.dumps(metrics, indent=2))


