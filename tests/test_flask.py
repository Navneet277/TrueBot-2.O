"""Integration tests for Flask routes."""

import json

import pytest

from app import app, init_db


@pytest.fixture(scope="module")
def test_client(tmp_path_factory):
    init_db()
    app.config.update({"TESTING": True})
    with app.test_client() as client:
        yield client


def test_home_status_code(test_client):
    response = test_client.get("/")
    assert response.status_code == 200


def test_predict_endpoint_requires_text(test_client):
    response = test_client.post("/predict", json={})
    assert response.status_code == 400


def test_predict_endpoint_success(test_client, monkeypatch):
    def fake_predict(*_args, **_kwargs):
        return {"label": "Real", "confidence": 99.0}

    monkeypatch.setattr("modules.predictor.predict_label", fake_predict)
    response = test_client.post("/predict", json={"text": "Sample"})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["label"] == "Real"


