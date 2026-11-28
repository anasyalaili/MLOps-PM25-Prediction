"""Tes pemantauan: /metrics, pencatatan request, dan PSI."""

import numpy as np
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

import metrics_app
from package_model import drift_baseline

BASELINE = drift_baseline(np.random.default_rng(0).uniform(0, 100, 5000))


def fake_model_app():
    """Tiruan server MLflow: prediksi = input + 10."""
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {}

    @app.post("/invocations")
    async def invocations(request: Request):
        rows = metrics_app.input_rows(await request.json())
        if not rows or "pm2_5" not in rows[0]:
            raise HTTPException(status_code=400)
        return {"predictions": [row["pm2_5"] + 10 for row in rows]}

    return app


@pytest.fixture
def setup():
    registry = CollectorRegistry()
    monitor = metrics_app.Monitor(
        {
            "baseline": {"pm2_5": BASELINE},
            "offline_metrics": {"cv_rmse_mean": 315.3},
        },
        registry=registry,
    )
    app = metrics_app.instrument(fake_model_app(), monitor)
    return TestClient(app), registry


def send(client, values):
    rows = [{"pm2_5": float(v)} for v in values]
    return client.post("/invocations", json={"dataframe_records": rows})


def test_metrics_endpoint_and_passthrough(setup):
    client, _ = setup
    assert client.get("/ping").status_code == 200
    assert send(client, [50]).json() == {"predictions": [60.0]}
    text = client.get("/metrics").text
    assert 'pm25_requests_total{status="200"} 1.0' in text
    assert 'pm25_model_offline_metric{metric="cv_rmse_mean"} 315.3' in text


def test_counts_predictions_and_errors(setup):
    client, registry = setup
    send(client, [100, 400])
    assert client.post("/invocations", json={}).status_code == 400
    value = registry.get_sample_value
    assert value("pm25_prediction_ugm3_count") == 2
    assert value("pm25_prediction_ugm3_sum") == 520
    assert value("pm25_requests_total", {"status": "400"}) == 1
    assert value("pm25_request_duration_seconds_count") == 2


def test_psi_appears_after_enough_samples(setup):
    client, registry = setup
    labels = {"feature": "pm2_5"}
    send(client, np.linspace(0, 100, 50))
    assert registry.get_sample_value("pm25_drift_psi", labels) is None
    send(client, np.linspace(0, 100, 450))
    assert registry.get_sample_value("pm25_drift_psi", labels) < 0.05
    send(client, np.full(500, 400.0))
    assert registry.get_sample_value("pm25_drift_psi", labels) > 0.2


def test_dataframe_split_input():
    body = {"dataframe_split": {"columns": ["pm2_5"], "data": [[1], [2]]}}
    assert metrics_app.input_rows(body) == [{"pm2_5": 1}, {"pm2_5": 2}]


def test_load_monitoring_missing_file(tmp_path):
    assert metrics_app.load_monitoring(str(tmp_path)) == {}


def test_resource_metrics(setup, tmp_path):
    _, registry = setup
    assert registry.get_sample_value("pm25_process_cpu_seconds_total") > 0
    current = tmp_path / "memory.current"
    current.write_text("2048\n")
    missing = str(tmp_path / "no")
    assert metrics_app.memory_bytes([missing, str(current)]) == 2048
    assert metrics_app.memory_bytes([missing]) is None
