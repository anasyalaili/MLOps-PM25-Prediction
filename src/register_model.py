r"""Alur MLflow Model Registry untuk model prediksi PM2.5 H+6.

Subperintah:
    list      tampilkan semua versi model beserta stage dan alias.
    register  daftarkan artefak model dari sebuah run sebagai versi baru.
    promote   ubah stage (atau alias) sebuah versi model.
    metadata  tulis metadata silsilah satu versi model ke JSON (untuk DVC).
    verify    muat model lewat mlflow.pyfunc dan jalankan prediksi uji.
    active    cetak ringkasan model aktif (Production) dalam tabel
              Markdown untuk dokumentasi README.

Contoh (dijalankan dari root repositori):
    python src/register_model.py list
    python src/register_model.py register --run-name 03-rf-300-d10-champion
    python src/register_model.py promote --version 1 --stage Staging
    python src/register_model.py promote --version 1 --stage Production
    python src/register_model.py metadata --version 1
    python src/register_model.py verify --stage Production
    python src/register_model.py active

Registrasi versi baru dari hasil training tidak membutuhkan skrip ini:
    python src/train.py ... --register-model
membuat versi baru pada model yang sama.

Catatan stage: fitur stage (None, Staging, Production, Archived) sudah
ditandai deprecated oleh MLflow dan digantikan alias.
"""

import argparse
import json
import os
import sys
import warnings

import mlflow
import pandas as pd
from mlflow import MlflowClient

from train import (
    DEFAULT_DATA_PATH,
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_URI,
    TARGET_COLUMN,
    TIME_COLUMN,
    load_dataset,
    restore_level,
)

DEFAULT_MODEL_NAME = "pm25-karhutla-h6"
DEFAULT_METADATA_DIR = os.path.join("models", "registry")
STAGES = ("None", "Staging", "Production", "Archived")
LINEAGE_TAGS = (
    "data_version",
    "git_commit",
    "data_raw_dvc_md5",
    "data_processed_dvc_md5",
    "model_fit_on",
)
METADATA_METRICS = (
    "cv_rmse_mean",
    "cv_rmse_std",
    "cv_rmse_log_mean",
    "cv_seasonal_rmse_mean",
    "val_rmse",
    "test_rmse",
    "test_rmse_log",
)


def parse_arguments(argv=None):
    """Baca argumen command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Daftar versi model.")

    reg = sub.add_parser("register", help="Daftarkan model dari sebuah run.")
    reg.add_argument("--run-id", help="ID run MLflow.")
    reg.add_argument("--run-name", help="Nama run (alternatif --run-id).")
    reg.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    reg.add_argument("--description", help="Deskripsi versi model.")

    pro = sub.add_parser("promote", help="Ubah stage sebuah versi.")
    pro.add_argument("--version", type=int, required=True)
    pro.add_argument("--stage", choices=STAGES, required=True)
    pro.add_argument(
        "--archive-existing",
        action="store_true",
        help="Arsipkan versi lain yang sedang berada di stage tujuan.",
    )
    pro.add_argument(
        "--alias",
        action="store_true",
        help="Tetapkan juga alias (nama stage huruf kecil) ke versi ini.",
    )

    meta = sub.add_parser("metadata", help="Tulis metadata silsilah ke JSON.")
    meta.add_argument("--version", type=int, required=True)
    meta.add_argument("--output-dir", default=DEFAULT_METADATA_DIR)

    ver = sub.add_parser("verify", help="Uji inferensi lewat pyfunc.")
    ver.add_argument("--version", type=int, help="Nomor versi.")
    ver.add_argument("--stage", choices=STAGES, help="Stage model.")
    ver.add_argument("--alias", help="Alias model, mis. production.")
    ver.add_argument("--data", default=DEFAULT_DATA_PATH)
    ver.add_argument("--rows", type=int, default=5)

    act = sub.add_parser(
        "active", help="Ringkasan model aktif dalam tabel Markdown."
    )
    act.add_argument("--version", type=int, help="Nomor versi.")
    act.add_argument("--stage", choices=STAGES, help="Default: Production.")
    act.add_argument("--alias", help="Alias model, mis. production.")
    return parser.parse_args(argv)


def find_run_id(client, experiment_name, run_id, run_name):
    """Tentukan ID run dari --run-id atau --run-name."""
    if run_id:
        return run_id
    if not run_name:
        raise ValueError("Isi --run-id atau --run-name.")
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' tidak ditemukan.")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["start_time DESC"],
    )
    if not runs:
        raise ValueError(
            f"Run '{run_name}' tidak ditemukan di '{experiment_name}'."
        )
    if len(runs) > 1:
        print(
            f"Ada {len(runs)} run bernama '{run_name}', dipakai yang terbaru."
        )
    return runs[0].info.run_id


def version_row(version):
    """Ringkas satu versi model menjadi satu baris teks."""
    aliases = ",".join(version.aliases) or "-"
    return (
        f"v{version.version}  stage={version.current_stage:<10} "
        f"alias={aliases:<12} run={version.run_id[:8]}  "
        f"{version.description or ''}"
    )


def command_list(client, args):
    """Tampilkan semua versi model."""
    versions = client.search_model_versions(f"name = '{args.model_name}'")
    if not versions:
        print(f"Belum ada versi untuk model '{args.model_name}'.")
        return 0
    for found in sorted(versions, key=lambda v: int(v.version)):
        # search_model_versions tidak mengisi alias; ambil per versi.
        version = client.get_model_version(args.model_name, found.version)
        run = client.get_run(version.run_id)
        data_version = run.data.tags.get("data_version", "unknown")
        print(f"{version_row(version)}  data={data_version}")
    return 0


def command_register(client, args):
    """Daftarkan artefak model dari sebuah run sebagai versi baru."""
    run_id = find_run_id(client, args.experiment, args.run_id, args.run_name)
    version = mlflow.register_model(f"runs:/{run_id}/model", args.model_name)
    if args.description:
        client.update_model_version(
            args.model_name, version.version, args.description
        )
    print(
        f"Terdaftar: {args.model_name} versi {version.version} (run {run_id})"
    )
    return 0


def command_promote(client, args):
    """Ubah stage (dan opsional alias) sebuah versi."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client.transition_model_version_stage(
            args.model_name,
            str(args.version),
            args.stage,
            archive_existing_versions=args.archive_existing,
        )
    if args.alias and args.stage in ("Staging", "Production"):
        client.set_registered_model_alias(
            args.model_name, args.stage.lower(), str(args.version)
        )
    current = client.get_model_version(args.model_name, str(args.version))
    print(f"v{args.version} sekarang: stage={current.current_stage}")
    return 0


def build_metadata(client, model_name, version_number):
    """Susun metadata silsilah untuk satu versi model."""
    version = client.get_model_version(model_name, str(version_number))
    run = client.get_run(version.run_id)
    tags, params, metrics = run.data.tags, run.data.params, run.data.metrics
    return {
        "model_name": model_name,
        "version": int(version.version),
        "stage": version.current_stage,
        "aliases": list(version.aliases),
        "run_id": version.run_id,
        "run_name": run.info.run_name,
        "experiment_id": run.info.experiment_id,
        "model_source": version.source,
        "target_mode": tags.get("target_mode", "level"),
        "lineage": {name: tags.get(name) for name in LINEAGE_TAGS},
        "model_params": {
            key: params.get(key)
            for key in (
                "model",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "max_features",
                "log_target",
            )
            if key in params
        },
        "features": params.get("features", "").split(","),
        "metrics": {
            key: metrics.get(key) for key in METADATA_METRICS if key in metrics
        },
    }


def command_metadata(client, args):
    """Tulis metadata satu versi ke JSON agar bisa dilacak DVC."""
    metadata = build_metadata(client, args.model_name, args.version)
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(
        args.output_dir, f"{args.model_name}_v{args.version}.json"
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"Metadata ditulis: {path}")
    print("Lacak dengan DVC: dvc add " + args.output_dir)
    return 0


def resolve_model_uri(client, args):
    """Susun URI model dari --version, --stage, atau --alias."""
    chosen = [args.version is not None, bool(args.stage), bool(args.alias)]
    if sum(chosen) != 1:
        raise ValueError("Pilih satu: --version, --stage, atau --alias.")
    if args.version is not None:
        return f"models:/{args.model_name}/{args.version}", args.version
    if args.alias:
        found = client.get_model_version_by_alias(args.model_name, args.alias)
        return f"models:/{args.model_name}@{args.alias}", int(found.version)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        latest = client.get_latest_versions(args.model_name, [args.stage])
    if not latest:
        raise ValueError(f"Tidak ada versi di stage '{args.stage}'.")
    return f"models:/{args.model_name}/{args.stage}", int(latest[0].version)


def command_verify(client, args):
    """Muat model lewat pyfunc dan jalankan prediksi pada data uji."""
    uri, version_number = resolve_model_uri(client, args)
    version = client.get_model_version(args.model_name, str(version_number))
    run = client.get_run(version.run_id)
    features = run.data.params.get("features", "").split(",")
    target_mode = run.data.tags.get("target_mode", "level")

    df = load_dataset(args.data)
    missing = [name for name in features if name not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset di '{args.data}' tidak punya kolom {missing} yang "
            f"dibutuhkan versi {version_number}."
        )
    sample = df.tail(args.rows).reset_index(drop=True)

    print(
        f"Memuat {uri} (versi {version_number}, stage "
        f"{version.current_stage}, {len(features)} fitur)"
    )
    model = mlflow.pyfunc.load_model(uri)
    raw = model.predict(sample[features])
    level = restore_level(raw, sample, target_mode)

    result = pd.DataFrame(
        {
            TIME_COLUMN: sample[TIME_COLUMN],
            "prediksi_h6": level.round(1),
            "aktual_h6": sample[TARGET_COLUMN].round(1),
        }
    )
    print(result.to_string(index=False))
    if target_mode != "level":
        print(
            f"Catatan: model ini dilatih dengan target-mode '{target_mode}'. "
            "Keluaran pyfunc berupa nilai transformasi; tabel di atas sudah "
            "dikembalikan ke skala PM2.5."
        )
    print("VERIFIKASI BERHASIL: model dapat dimuat dan memberi prediksi.")
    return 0


def format_metric(metrics, name, spread=None):
    """Teks satu metrik (opsional dengan simpangan baku), '-' jika kosong."""
    value = metrics.get(name)
    if value is None:
        return "-"
    text = f"{value:.1f}"
    if spread and metrics.get(spread) is not None:
        text += f" ± {metrics[spread]:.1f}"
    return text


def command_active(client, args):
    """Cetak ringkasan model aktif sebagai tabel Markdown."""
    if args.version is None and not args.stage and not args.alias:
        args.stage = "Production"
    uri, version_number = resolve_model_uri(client, args)
    meta = build_metadata(client, args.model_name, version_number)
    params = ", ".join(
        f"{key}={value}" for key, value in meta["model_params"].items()
    )
    metrics = meta["metrics"]
    rows = [
        ("Model", meta["model_name"]),
        ("Versi aktif", f"{meta['version']} (stage {meta['stage']})"),
        ("URI inferensi", f"`{uri}`"),
        ("Run asal", f"{meta['run_name']} (`{meta['run_id'][:8]}`)"),
        ("Versi data", meta["lineage"]["data_version"] or "-"),
        ("Git commit", f"`{(meta['lineage']['git_commit'] or '-')[:10]}`"),
        ("Jumlah fitur", len(meta["features"])),
        ("Target mode", meta["target_mode"]),
        ("Parameter", params or "-"),
        (
            "CV RMSE (± std)",
            format_metric(metrics, "cv_rmse_mean", "cv_rmse_std"),
        ),
        (
            "CV RMSE baseline musiman",
            format_metric(metrics, "cv_seasonal_rmse_mean"),
        ),
        ("Test RMSE", format_metric(metrics, "test_rmse")),
    ]
    print("| Item | Nilai |")
    print("|---|---|")
    for label, value in rows:
        print(f"| {label} | {value} |")
    return 0


COMMANDS = {
    "list": command_list,
    "register": command_register,
    "promote": command_promote,
    "metadata": command_metadata,
    "verify": command_verify,
    "active": command_active,
}


def main(argv=None):
    """Titik masuk skrip; return 0 jika sukses."""
    args = parse_arguments(argv)
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    try:
        return COMMANDS[args.command](client, args)
    except (ValueError, FileNotFoundError, mlflow.MlflowException) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
