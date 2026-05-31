"""API tests for Tamil Sentiment API."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


MOCK_RESULT = {
    "text": "romba nalla da",
    "sentiment": "positive",
    "confidence": 0.92,
    "probabilities": {"negative": 0.03, "neutral": 0.05, "positive": 0.92},
    "inference_time_ms": 45.2,
    "detected_language": "tanglish",
}

MOCK_METADATA = {
    "model_name": "ai4bharat/indic-bert",
    "test_accuracy": 0.918,
    "test_f1": 0.923,
    "best_val_f1": 0.931,
    "id_to_label": {"0": "negative", "1": "neutral", "2": "positive"},
    "label_map": {"positive": 2, "neutral": 1, "negative": 0},
}


@pytest.fixture
def client():
    with patch("api.main._model", MagicMock()), \
         patch("api.main._tokenizer", MagicMock()), \
         patch("api.main._metadata", MOCK_METADATA), \
         patch("api.main.predict_single", return_value={
             "sentiment": "positive",
             "confidence": 0.92,
             "probabilities": {"negative": 0.03, "neutral": 0.05, "positive": 0.92},
             "inference_time_ms": 45.2,
         }):
        from api.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_predict_english(client):
    resp = client.post("/predict", json={"text": "This product is amazing!"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sentiment"] in ["positive", "neutral", "negative"]
    assert 0 <= data["confidence"] <= 1


def test_predict_tamil(client):
    resp = client.post("/predict", json={"text": "இந்த பொருள் நல்லா இருக்கு"})
    assert resp.status_code == 200
    assert resp.json()["detected_language"] == "tamil"


def test_predict_tanglish(client):
    resp = client.post("/predict", json={"text": "romba nalla da!"})
    assert resp.status_code == 200


def test_predict_batch(client):
    resp = client.post("/predict/batch", json={
        "texts": ["Good product", "மோசமான தரம்", "okay only"]
    })
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_empty_text(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_metrics(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "test_f1" in data
    assert "test_accuracy" in data
