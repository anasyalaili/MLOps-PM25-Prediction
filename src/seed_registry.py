r"""Seed Model Registry MLflow dengan model demo untuk orkestrasi Docker.

Data asli proyek ada di remote DVC (butuh kredensial), jadi kontainer seed
melatih ExtraTrees kecil pada data SINTETIS dengan 34 fitur yang sama
seperti champion, lalu mendaftarkannya dan mempromosikannya ke Production.
Model ini hanya untuk membuktikan alur layanan (registry -> API), metriknya
tidak bermakna dan tidak boleh dibandingkan dengan hasil di README.

Idempoten: bila stage Production sudah terisi, skrip tidak melakukan apa-apa
(sehingga data di volume tidak digandakan saat compose dijalankan ulang).

Contoh:
    MLFLOW_TRACKING_URI=http://localhost:5000 python src/seed_registry.py
"""

import argparse
import time
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import ExtraTreesRegressor

from train import CURRENT_COLUMN, FEATURES, TRUSTED_SKOPS_TYPES

DEFAULT_MODEL_NAME = "pm25-karhutla-h6"
DEFAULT_EXPERIMENT = "pm25-karhutla-h6-docker-seed"
WAIT_SECONDS = 60


def parse_arguments(argv=None):
    """Baca argumen command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def make_synthetic_data(rows, seed):
    """Bangun data sintetis: kolom FEATURES dan target delta H+6."""
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        rng.uniform(0.0, 1.0, size=(rows, len(FEATURES))), columns=FEATURES
    )
    data[CURRENT_COLUMN] = rng.uniform(10.0, 300.0, size=rows)
    delta = (
        0.2 * (data["hotspot_count_100km"] - 0.5) * 100.0
        - 0.1 * data[CURRENT_COLUMN]
        + rng.normal(0.0, 5.0, size=rows)
    )
    return data, delta


def wait_for_server(client):
    """Tunggu server MLflow menjawab (jaga-jaga selain healthcheck)."""
    deadline = time.time() + WAIT_SECONDS
    while True:
        try:
            client.search_experiments(max_results=1)
            return
        except Exception:
            if time.time() > deadline:
                raise
            time.sleep(2)


def has_production(client, model_name):
    """True bila stage Production sudah punya versi."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bool(client.get_latest_versions(model_name, ["Production"]))
    except mlflow.exceptions.MlflowException:
        return False


def seed(args):
    """Latih, daftarkan, dan promosikan model demo."""
    client = MlflowClient()
    wait_for_server(client)
    if has_production(client, args.model_name):
        print("Production sudah terisi; seed dilewati.")
        return 0

    data, delta = make_synthetic_data(args.rows, args.seed)
    model = ExtraTreesRegressor(
        n_estimators=50, max_depth=8, random_state=args.seed
    ).fit(data[FEATURES], delta)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="seed-sintetis") as run:
        mlflow.log_params(
            {
                "model": "extra",
                "features": ",".join(FEATURES),
                "n_features": len(FEATURES),
                "target_mode": "delta",
                "n_train": args.rows,
            }
        )
        mlflow.set_tags(
            {
                "target_mode": "delta",
                "data_version": "sintetis",
                "keterangan": "model demo orkestrasi, bukan champion",
            }
        )
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=list(TRUSTED_SKOPS_TYPES),
            registered_model_name=args.model_name,
        )
        run_id = run.info.run_id

    versions = client.search_model_versions(f"run_id='{run_id}'")
    version = str(max(int(v.version) for v in versions))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client.transition_model_version_stage(
            args.model_name, version, "Production"
        )
    client.set_registered_model_alias(args.model_name, "production", version)
    print(f"Seed selesai: {args.model_name} v{version} -> Production.")
    return 0


def main(argv=None):
    """Titik masuk CLI."""
    return seed(parse_arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())
