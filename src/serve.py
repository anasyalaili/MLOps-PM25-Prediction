r"""API inferensi PM2.5 H+6 (FastAPI) yang memuat model dari MLflow.

Model diambil dari Model Registry MLflow (default: stage Production),
sehingga API tidak perlu dibangun ulang saat versi model berganti, cukup
panggil POST /reload.

Variabel lingkungan:
    MLFLOW_TRACKING_URI  alamat server MLflow (default http://localhost:5000)
    MODEL_NAME           nama model terdaftar (default pm25-karhutla-h6)
    MODEL_STAGE          stage yang dilayani (default Production)

Endpoint:
    GET  /health      status API dan apakah model sudah termuat.
    GET  /model-info  versi, stage, fitur, dan target-mode model aktif.
    POST /predict     {"records": [{fitur: nilai, ...}, ...]} -> prediksi
                      PM2.5 H+6 dalam ug/m3.
    POST /reload      muat ulang model dari registry.

Jalankan lokal:
    uvicorn serve:app --app-dir src --port 8000
"""

import os
import threading
import warnings
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel

from train import CURRENT_COLUMN, restore_level

DEFAULT_MODEL_NAME = "pm25-karhutla-h6"
DEFAULT_STAGE = "Production"


class PredictRequest(BaseModel):
    """Satu atau lebih baris fitur."""

    records: list[dict[str, float]]


class ModelHolder:
    """Menyimpan model aktif; aman dipanggil dari banyak request."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.info = {}
        self.error = None

    def load(self):
        """Muat model stage aktif dari registry; simpan galat bila gagal."""
        name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        stage = os.getenv("MODEL_STAGE", DEFAULT_STAGE)
        with self.lock:
            try:
                client = MlflowClient()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    latest = client.get_latest_versions(name, [stage])
                if not latest:
                    raise LookupError(
                        f"Belum ada versi '{name}' di stage '{stage}'."
                    )
                version = latest[0]
                run = client.get_run(version.run_id)
                model = mlflow.pyfunc.load_model(
                    f"models:/{name}/{version.version}"
                )
                self.model = model
                self.info = {
                    "model_name": name,
                    "version": int(version.version),
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "features": run.data.params["features"].split(","),
                    "target_mode": run.data.tags.get("target_mode", "level"),
                    "data_version": run.data.tags.get("data_version", ""),
                }
                self.error = None
            except Exception as exc:
                self.model = None
                self.info = {}
                self.error = f"{type(exc).__name__}: {exc}"
            return self.model is not None


holder = ModelHolder()


@asynccontextmanager
async def lifespan(_app):
    """Coba muat model saat start; kegagalan tidak menghentikan API."""
    holder.load()
    yield


app = FastAPI(
    title="API Prediksi PM2.5 H+6", version="1.0.0", lifespan=lifespan
)


def require_model():
    """Pastikan model tersedia, coba muat ulang bila belum."""
    if holder.model is None and not holder.load():
        raise HTTPException(status_code=503, detail=holder.error)
    return holder


@app.get("/health")
def health():
    """Status API; model_loaded false berarti registry belum siap."""
    return {
        "status": "ok",
        "model_loaded": holder.model is not None,
        "detail": holder.error,
    }


@app.get("/model-info")
def model_info():
    """Metadata model yang sedang dilayani."""
    return require_model().info


@app.post("/reload")
def reload_model():
    """Muat ulang model dari registry (mis. setelah promosi versi baru)."""
    if not holder.load():
        raise HTTPException(status_code=503, detail=holder.error)
    return holder.info


@app.post("/predict")
def predict(request: PredictRequest):
    """Prediksi PM2.5 H+6 (ug/m3) untuk setiap baris fitur."""
    current = require_model()
    if not request.records:
        raise HTTPException(status_code=422, detail="records kosong.")
    features = current.info["features"]
    frame = pd.DataFrame(request.records)
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"Fitur wajib belum ada: {missing}"
        )
    frame = frame[features]
    raw = current.model.predict(frame)
    level = restore_level(raw, frame, current.info["target_mode"])
    return {
        "model_version": current.info["version"],
        "target_mode": current.info["target_mode"],
        "base_column": CURRENT_COLUMN,
        "predictions_h6": [round(float(v), 1) for v in level],
    }
