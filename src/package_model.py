r"""Kemas model registry menjadi model siap-serving, lalu jadikan Production.

Mengambil satu versi model dasar dari Model Registry, membungkusnya dengan
pm25_pyfunc.PM25Forecaster (keluaran langsung ug/m3), mendaftarkannya sebagai
versi baru pada nama model yang sama, dan mempromosikannya ke Production.

Contoh:
    python src/package_model.py --tracking-uri http://localhost:5000
    python src/package_model.py --tracking-uri http://localhost:5000 \
        --version 1 --example examples/predict_request.json
"""

import argparse
import json
import os
import warnings

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

DEFAULT_MODEL_NAME = "pm25-karhutla-h6"
DEFAULT_EXPERIMENT = "pm25-karhutla-h6-serving"
WRAPPER_TAG = "serving_wrapper"
CODE_PATH = os.path.join(os.path.dirname(__file__), "pm25_pyfunc.py")

EXAMPLE_VALUES = {
    "temperature": 28.0,
    "relative_humidity": 75.0,
    "wind_speed": 6.0,
    "precip": 0.0,
    "cloud_cover": 40.0,
    "pm2_5_diff": 5.0,
    "pm2_5": 120.0,
    "pm10": 150.0,
    "european_aqi": 90.0,
    "hotspot_nearest": 35.0,
    "hotspot_trend": 0.0,
    "hotspot": 3.0,
    "hour_sin": 0.0,
    "hour_cos": -1.0,
    "hour_of_day": 12.0,
}


def parse_arguments(argv=None):
    """Baca argumen command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--version",
        type=int,
        help="Versi model dasar. Default: versi terbaru yang belum dikemas.",
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--example", help="Tulis contoh request /invocations ke berkas ini."
    )
    return parser.parse_args(argv)


def example_row(features):
    """Satu baris contoh dengan nilai wajar untuk setiap fitur."""
    row = {}
    for name in features:
        matches = [key for key in EXAMPLE_VALUES if name.startswith(key)]
        key = max(matches, key=len) if matches else None
        row[name] = EXAMPLE_VALUES.get(key, 1.0)
    return row


def find_base_version(client, name, number=None):
    """Versi yang diminta, atau versi terbaru yang bukan pembungkus."""
    if number is not None:
        version = client.get_model_version(name, str(number))
        if WRAPPER_TAG in client.get_run(version.run_id).data.tags:
            raise ValueError(f"v{number} sudah berupa model serving.")
        return version
    versions = client.search_model_versions(f"name='{name}'")
    for version in sorted(versions, key=lambda v: -int(v.version)):
        if WRAPPER_TAG not in client.get_run(version.run_id).data.tags:
            return version
    raise ValueError(f"Tidak ada model dasar bernama '{name}'.")


def package(args):
    """Bungkus, daftarkan, dan promosikan model."""
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    base = find_base_version(client, args.model_name, args.version)
    run = client.get_run(base.run_id)
    features = run.data.params["features"].split(",")
    target_mode = run.data.tags.get("target_mode", "level")
    data_version = run.data.tags.get("data_version", "")

    example = pd.DataFrame([example_row(features)])
    requirements = ["mlflow", "scikit-learn", "numpy", "pandas", "skops"]
    if run.data.params.get("model") == "xgb":
        requirements.append("xgboost")
    pinned = [f"{pkg}=={installed_version(pkg)}" for pkg in requirements]

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"serving-v{base.version}") as wrapper:
        mlflow.log_params(
            {"features": ",".join(features), "base_version": base.version}
        )

        mlflow.set_tags(
            {
                WRAPPER_TAG: "true",
                "target_mode": "level",
                "base_target_mode": target_mode,
                "base_run_id": base.run_id,
                "data_version": data_version,
            }
        )
        mlflow.pyfunc.log_model(
            name="model",
            python_model=CODE_PATH,
            artifacts={"base": f"models:/{args.model_name}/{base.version}"},
            model_config={"features": features, "target_mode": target_mode},
            signature=infer_signature(example, pd.Series([0.0]).to_numpy()),
            input_example=example,
            pip_requirements=pinned,
            registered_model_name=args.model_name,
        )
        run_id = wrapper.info.run_id

    found = client.search_model_versions(f"run_id='{run_id}'")
    version = str(max(int(v.version) for v in found))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client.transition_model_version_stage(
            args.model_name,
            version,
            "Production",
            archive_existing_versions=True,
        )
    client.set_registered_model_alias(args.model_name, "production", version)

    if args.example:
        body = {"dataframe_records": example.to_dict(orient="records")}
        with open(args.example, "w", encoding="utf-8") as handle:
            json.dump(body, handle, indent=2)
            handle.write("\n")
    print(
        f"v{base.version} (target {target_mode}, data {data_version}) "
        f"dikemas menjadi {args.model_name} v{version} -> Production."
    )
    return int(version)


def installed_version(package_name):
    """Versi paket yang terpasang, agar image sama dengan lingkungan ini."""
    from importlib.metadata import version

    return version(package_name)


def main(argv=None):
    """Titik masuk CLI."""
    package(parse_arguments(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
