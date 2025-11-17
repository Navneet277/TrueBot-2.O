"""
Text preprocessing utilities for TrueBot.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

LOGGER = logging.getLogger(__name__)
STOP_WORDS: List[str] = []
LEMMATIZER = WordNetLemmatizer()


def ensure_nltk_resources() -> None:
    """Download nltk datasets if missing."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


def init_resources() -> None:
    """Initialize stopwords list."""
    ensure_nltk_resources()
    global STOP_WORDS
    STOP_WORDS = stopwords.words("english")


def clean_text(text: str) -> str:
    """
    Perform basic cleanup: lowercase, remove urls, punctuation, digits.
    """
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def lemmatize_tokens(tokens: Iterable[str]) -> List[str]:
    """Lemmatize tokens with fallback."""
    lemmas = []
    for token in tokens:
        lemma = LEMMATIZER.lemmatize(token)
        lemmas.append(lemma)
    return lemmas


def preprocess_text(text: str) -> str:
    """Full pipeline for a single text."""
    if not STOP_WORDS:
        init_resources()
    cleaned = clean_text(text)
    tokens = cleaned.split()
    tokens = [token for token in tokens if token not in STOP_WORDS]
    lemmas = lemmatize_tokens(tokens)
    return " ".join(lemmas)


def preprocess_corpus(corpus: Iterable[str]) -> List[str]:
    """Apply preprocess_text to list."""
    return [preprocess_text(doc) for doc in corpus]


def build_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Create TF-IDF vectorizer preconfigured."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )


