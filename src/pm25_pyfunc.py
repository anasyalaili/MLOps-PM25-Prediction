"""Model pyfunc (models-from-code): keluaran langsung PM2.5 H+6 (ug/m3).

Model dasar memprediksi target yang sudah ditransformasi (mis. selisih
PM2.5). Pembungkus ini mengembalikannya ke skala asli, sehingga
`mlflow models serve` dan image hasil `build-docker` tidak butuh kode lain.

Berkas ini sengaja berdiri sendiri (tidak mengimpor train.py) karena ikut
disalin ke dalam artefak model. Logikanya harus sama dengan
train.restore_level, kesamaannya dijaga tests/test_package_model.py.
"""

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import set_model


class PM25Forecaster(mlflow.pyfunc.PythonModel):
    """Bungkus model dasar dan kembalikan prediksi pada skala PM2.5."""

    def load_context(self, context):
        self.model = mlflow.sklearn.load_model(context.artifacts["base"])
        self.features = list(context.model_config["features"])
        self.target_mode = context.model_config["target_mode"]

    def predict(self, context, model_input, params=None):
        frame = model_input[self.features]
        prediction = np.asarray(self.model.predict(frame), dtype=float)
        current = frame["pm2_5"].to_numpy(dtype=float)
        if self.target_mode == "delta":
            prediction = prediction + current
        elif self.target_mode == "log-ratio":
            prediction = np.expm1(prediction + np.log1p(current))
        return np.clip(prediction, 0.0, None)


set_model(PM25Forecaster())
