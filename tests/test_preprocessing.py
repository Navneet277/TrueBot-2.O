"""Unit tests for preprocessing utilities."""

from modules import preprocessing


def test_clean_text_lowercase_and_strip():
    text = "Hello WORLD!!! Visit https://example.com"
    cleaned = preprocessing.clean_text(text)
    assert "hello world" in cleaned
    assert "http" not in cleaned


def test_preprocess_text_removes_stopwords():
    preprocessing.init_resources()
    text = "This is a simple sentence for testing."
    processed = preprocessing.preprocess_text(text)
    assert "this" not in processed
    assert "sentence" in processed


