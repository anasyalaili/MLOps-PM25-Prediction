"""Tes API inferensi dan seed registry pada store MLflow sementara."""

import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

import seed_registry
import serve
from train import FEATURES

NAME = "pm25-karhutla-h6"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Store MLflow kosong (SQLite) dan reset model di API."""
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    serve.holder.model = None
    serve.holder.info = {}
    serve.holder.error = None
    yield tmp_path
    serve.holder.model = None


@pytest.fixture
def seeded(store):
    """Store yang sudah berisi model seed di Production."""
    assert seed_registry.main(["--rows", "120"]) == 0
    return store


def record(pm25=120.0):
    """Satu baris fitur lengkap."""
    row = {name: 0.5 for name in FEATURES}
    row["pm2_5"] = pm25
    return row


def test_make_synthetic_data_has_all_features():
    data, delta = seed_registry.make_synthetic_data(30, 1)
    assert list(data.columns) == FEATURES
    assert len(delta) == 30


def test_seed_registers_production_and_is_idempotent(store, capsys):
    assert seed_registry.main(["--rows", "120"]) == 0
    client = MlflowClient()
    assert seed_registry.has_production(client, NAME)
    assert seed_registry.main(["--rows", "120"]) == 0
    assert "dilewati" in capsys.readouterr().out
    assert len(client.search_model_versions(f"name='{NAME}'")) == 1


def test_health_without_model_reports_not_loaded(store):
    with TestClient(serve.app) as api:
        body = api.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False
        reply = api.post("/predict", json={"records": [record()]})
        assert reply.status_code == 503


def test_model_info_and_predict(seeded):
    with TestClient(serve.app) as api:
        info = api.get("/model-info").json()
        assert info["stage"] == "Production"
        assert info["target_mode"] == "delta"
        assert info["features"] == FEATURES
        reply = api.post("/predict", json={"records": [record(), record(40)]})
        assert reply.status_code == 200
        values = reply.json()["predictions_h6"]
        assert len(values) == 2
        assert all(v >= 0 for v in values)


def test_predict_rejects_missing_feature_and_empty(seeded):
    with TestClient(serve.app) as api:
        partial = record()
        del partial["pm10"]
        bad = api.post("/predict", json={"records": [partial]})
        assert bad.status_code == 422
        assert "pm10" in bad.json()["detail"]
        assert api.post("/predict", json={"records": []}).status_code == 422


def test_reload_picks_up_model(store):
    with TestClient(serve.app) as api:
        assert api.post("/reload").status_code == 503
        seed_registry.main(["--rows", "120"])
        assert api.post("/reload").status_code == 200
        assert api.get("/health").json()["model_loaded"] is True
