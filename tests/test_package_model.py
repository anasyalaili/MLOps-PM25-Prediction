"""Tes pengemasan model serving pada store MLflow sementara."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

import package_model
import seed_registry
from train import FEATURES, restore_level

NAME = "pm25-karhutla-h6"


@pytest.fixture(scope="module")
def packaged(tmp_path_factory):
    """Store berisi model dasar (delta) yang sudah dikemas."""
    folder = tmp_path_factory.mktemp("serving")
    uri = f"sqlite:///{folder}/mlflow.db"
    patch = pytest.MonkeyPatch()
    patch.chdir(folder)
    patch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    seed_registry.main(["--rows", "120"])
    example = folder / "request.json"
    package_model.main(["--tracking-uri", uri, "--example", str(example)])
    yield {"uri": uri, "example": example}
    patch.undo()


def test_wrapper_becomes_only_production_version(packaged):
    client = MlflowClient()
    stages = {
        v.version: v.current_stage
        for v in client.search_model_versions(f"name='{NAME}'")
    }
    assert {str(k): v for k, v in stages.items()} == {
        "1": "Archived",
        "2": "Production",
    }


def test_output_matches_restore_level(packaged):
    data, _ = seed_registry.make_synthetic_data(20, 7)
    base = mlflow.sklearn.load_model(f"models:/{NAME}/1")
    expected = restore_level(base.predict(data[FEATURES]), data, "delta")
    served = mlflow.pyfunc.load_model(f"models:/{NAME}/2")
    np.testing.assert_allclose(served.predict(data[FEATURES]), expected)


def test_example_request_is_valid(packaged):
    body = json.loads(packaged["example"].read_text())
    frame = pd.DataFrame(body["dataframe_records"])
    assert list(frame.columns) == FEATURES
    served = mlflow.pyfunc.load_model(f"models:/{NAME}/2")
    assert served.predict(frame)[0] >= 0


def test_packaging_again_uses_base_version(packaged):
    client = MlflowClient()
    assert str(package_model.find_base_version(client, NAME).version) == "1"
    with pytest.raises(ValueError):
        package_model.find_base_version(client, NAME, 2)


def test_example_row_picks_most_specific_prefix():
    row = package_model.example_row(["pm2_5_diff_1h", "pm2_5_lag_1h"])
    assert row == {"pm2_5_diff_1h": 5.0, "pm2_5_lag_1h": 120.0}
