r"""Training model prediksi PM2.5 H+6 dengan pelacakan MLflow.

Membaca dataset hasil feature engineering (data/processed/), membagi
data secara kronologis (train / validasi / test), melatih model regresi,
lalu mencatat parameter, metrik, dan model ke MLflow.

Satu eksekusi skrip = satu run MLflow. Jalankan beberapa kali dengan
hyperparameter berbeda untuk membandingkan hasilnya di dashboard MLflow.

Contoh run:
    python src/train.py --n-estimators 100 --max-depth 5 \
        --data-version data-v3
    python src/train.py --n-estimators 300 --max-depth 10 --log-target \
        --data-version data-v3
    python src/train.py --n-estimators 300 --max-depth 10 \
        --drop-features european_aqi --data-version data-v3
    python src/train.py --n-estimators 300 --max-depth 10 --no-hotspot \
        --data-version data-v3
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import (GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.inspection import permutation_importance

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT = "pm25-karhutla-h6"
DEFAULT_DATA_PATH = os.path.join("data", "processed")
RAW_DVC_FILE = os.path.join("data", "raw.dvc")
PROCESSED_DVC_FILE = os.path.join("data", "processed.dvc")
TARGET_COLUMN = "target_pm2_5_h6"
TIME_COLUMN = "timestamp"
CLASS_THRESHOLDS = (150.0, 500.0)
SPIKE_RATIO = 1.5
MIN_ROWS_PER_SPLIT = 24
HOTSPOT_FEATURES = ["hotspot_count_100km", "hotspot_count_log"]
TRUSTED_SKOPS_TYPES = ["sklearn.tree._tree.Tree"]
FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
    "pm2_5",
    "pm10",
    "european_aqi",
    "pm2_5_lag_1h",
    "pm2_5_lag_6h",
    "hotspot_count_100km",
    "hotspot_count_log",
    "hour_of_day",
    "is_dry_season",
]


def parse_arguments(argv=None):
    """Baca argumen command line."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=DEFAULT_DATA_PATH)
    p.add_argument("--model", choices=["rf", "gbr", "linear", "xgb"],
                   default="rf")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--val-days", type=int, default=7)
    p.add_argument("--test-days", type=int, default=7)
    p.add_argument("--gap-hours", type=int, default=6)
    p.add_argument(
        "--log-target",
        action="store_true",
        help="Latih pada log1p(target), prediksi skala asli.",
    )
    p.add_argument("--cv-folds", type=int, default=0,
                   help="Jumlah fold walk-forward (0 = nonaktif).")
    p.add_argument("--cv-fold-days", type=int, default=7)
    p.add_argument(
        "--no-hotspot",
        action="store_true",
        help="Ablation: buang kedua fitur hotspot.",
    )
    p.add_argument(
        "--drop-features",
        nargs="*",
        default=[],
        help="Fitur tambahan yang dibuang, mis. european_aqi.",
    )
    p.add_argument(
        "--cutoff",
        default=None,
        help="Buang baris dengan timestamp setelah nilai ini.",
    )
    p.add_argument(
        "--data-version", default=None, help="Label versi data, mis. data-v3."
    )
    p.add_argument("--run-name", default=None)
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    p.add_argument("--register-model", action="store_true")
    return p.parse_args(argv)


def find_data_files(path):
    """Cari file parquet (diutamakan) atau csv di bawah path."""
    if os.path.isfile(path):
        return [path]
    for ext in ("parquet", "csv"):
        files = sorted(
            glob.glob(os.path.join(path, "**", f"*.{ext}"), recursive=True)
        )
        if files:
            return files
    return []


def load_dataset(path):
    """Gabungkan semua partisi, urut waktu, tanpa timestamp ganda."""
    files = find_data_files(path)
    if not files:
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di '{path}'. Jalankan "
            "src/build_features.py dulu."
        )
    frames = [
        pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN])
    df = df.drop_duplicates(TIME_COLUMN, keep="last")
    return df.sort_values(TIME_COLUMN).reset_index(drop=True)


def resolve_features(df, no_hotspot=False, drop=()):
    """Daftar fitur final, error jelas jika ada kolom hilang."""
    missing = [c for c in FEATURES + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ada di dataset: {missing}")
    removed = set(drop) | (set(HOTSPOT_FEATURES) if no_hotspot else set())
    return [c for c in FEATURES if c not in removed]


def prepare_training_frame(df, features, cutoff=None):
    """Terapkan cutoff dan buang baris dengan NaN."""
    if cutoff:
        df = df[df[TIME_COLUMN] <= pd.Timestamp(cutoff)]
    return df.dropna(subset=features + [TARGET_COLUMN]).reset_index(drop=True)


def chronological_split(df, val_days, test_days, gap_hours):
    """Split train/val/test berdasarkan waktu dengan jeda anti-bocor."""
    end = df[TIME_COLUMN].max()
    gap = pd.Timedelta(hours=gap_hours)
    test_start = end - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
    val_end = test_start - gap
    val_start = val_end - pd.Timedelta(days=val_days)
    train_end = val_start - gap
    t = df[TIME_COLUMN]
    train = df[t <= train_end]
    val = df[(t > val_start) & (t <= val_end)]
    test = df[t >= test_start]
    for name, part in (("train", train), ("val", val), ("test", test)):
        if len(part) < MIN_ROWS_PER_SPLIT:
            raise ValueError(
                f"Split {name} hanya {len(part)} baris (min "
                f"{MIN_ROWS_PER_SPLIT}). Tambah data (backfill) atau "
                "kecilkan --val-days/--test-days."
            )
    return train, val, test


def build_model(args):
    """Buat model, opsional dibungkus transformasi log pada target."""
    if args.model == "xgb":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth or 6,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
        )
    elif args.model == "gbr":
        model = GradientBoostingRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth or 3,
            learning_rate=args.learning_rate,
            random_state=args.random_state)
    elif args.model == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=-1,
        )
    if args.log_target:
        model = TransformedTargetRegressor(
            regressor=model, func=np.log1p, inverse_func=np.expm1
        )
    return model


def class_scores(true_cls, pred_cls, name):
    """Precision, recall, dan support untuk satu definisi kelas."""
    return {
        f"{name}_precision": float(
            precision_score(true_cls, pred_cls, zero_division=0)
        ),
        f"{name}_recall": float(
            recall_score(true_cls, pred_cls, zero_division=0)
        ),
        f"{name}_support": int(true_cls.sum()),
    }


def compute_metrics(y_true, y_pred, prefix, current=None):
    """Regresi + kelas ambang (>150, >500) + kelas lonjakan.

    Lonjakan: PM2.5 jam t+6 lebih dari SPIKE_RATIO kali PM2.5 saat ini.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}_r2": float(r2_score(y_true, y_pred)),
    }
    for limit in CLASS_THRESHOLDS:
        out.update(
            class_scores(
                (y_true > limit).astype(int),
                (y_pred > limit).astype(int),
                f"{prefix}_gt{int(limit)}",
            )
        )
    if current is not None:
        base = SPIKE_RATIO * np.asarray(current)
        out.update(
            class_scores(
                (y_true > base).astype(int),
                (y_pred > base).astype(int),
                f"{prefix}_spike",
            )
        )
    return out


def compute_baseline_metrics(df, prefix):
    """Baseline persistensi: prediksi H+6 = PM2.5 saat ini."""
    return compute_metrics(df[TARGET_COLUMN], df["pm2_5"], prefix, df["pm2_5"])

def summarize_folds(rows, prefix):
    """Rata-rata dan std tiap metrik antar fold (kunci {prefix}_*)."""
    summary = {}
    for key in rows[0]:
        if key.endswith("_support") or key.endswith("_n_rows"):
            continue
        values = np.array([row[key] for row in rows])
        name = key[len("fold_"):]
        summary[f"{prefix}_{name}_mean"] = float(values.mean())
        summary[f"{prefix}_{name}_std"] = float(values.std())
    return summary


def walk_forward_scores(df, features, args):
    """Evaluasi rolling-origin: tiap fold dilatih dari masa lalu saja.

    Fold terakhir = test-days terakhir, fold sebelumnya bergeser mundur.
    Baseline musiman dihitung di fold yang sama (kunci cv_seasonal_*).
    """
    end = df[TIME_COLUMN].max()
    gap = pd.Timedelta(hours=args.gap_hours + 1)
    span = pd.Timedelta(days=args.cv_fold_days)
    model_rows, seasonal_rows = [], []
    for k in range(args.cv_folds, 0, -1):
        test_end = end - (k - 1) * span
        test_start = test_end - span + pd.Timedelta(hours=1)
        t = df[TIME_COLUMN]
        train = df[t <= test_start - gap]
        test = df[(t >= test_start) & (t <= test_end)]
        if len(train) < MIN_ROWS_PER_SPLIT or len(test) < MIN_ROWS_PER_SPLIT:
            continue
        model = build_model(args)
        model.fit(train[features], train[TARGET_COLUMN])
        model_rows.append(compute_metrics(
            test[TARGET_COLUMN], model.predict(test[features]), "fold",
            test["pm2_5"]))
        seasonal_rows.append(
            compute_seasonal_baseline_metrics(df, test, "fold"))
    if not model_rows:
        raise ValueError("Tidak ada fold walk-forward yang valid.")
    summary = {"cv_folds_used": float(len(model_rows))}
    summary.update(summarize_folds(model_rows, "cv"))
    summary.update(summarize_folds(seasonal_rows, "cv_seasonal"))
    return summary

def compute_feature_importance(model, val, features, random_state):
    """Tabel pentingnya fitur.

    permutation_*: kenaikan RMSE di data validasi saat satu fitur diacak
    (satuan ug/m3). impurity: bawaan pohon, bila tersedia.
    """
    perm = permutation_importance(
        model, val[features], val[TARGET_COLUMN], n_repeats=5,
        random_state=random_state, scoring="neg_root_mean_squared_error")
    table = pd.DataFrame({
        "feature": features,
        "permutation_mean": perm.importances_mean,
        "permutation_std": perm.importances_std,
    })
    inner = getattr(model, "regressor_", model)
    if hasattr(inner, "feature_importances_"):
        table["impurity"] = inner.feature_importances_
    return table.sort_values(
        "permutation_mean", ascending=False).reset_index(drop=True)


def log_feature_importance(mlflow, table):
    """Simpan tabel (CSV) dan grafik batang ke MLflow."""
    mlflow.log_text(table.to_csv(index=False), "feature_importance.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib tidak terpasang, grafik dilewati.")
        return
    ordered = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ordered["feature"], ordered["permutation_mean"],
            xerr=ordered["permutation_std"])
    ax.set_xlabel("Kenaikan RMSE val saat fitur diacak")
    ax.set_title("Permutation importance")
    fig.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)

def compute_seasonal_baseline_metrics(full, part, prefix):
    """Baseline musiman: prediksi H+6 = PM2.5 pada jam yang sama kemarin.

    Target jam t+6 dibandingkan dengan pm2_5 di t+6-24 = t-18 jam, yang
    sudah diketahui saat prediksi dibuat (tanpa kebocoran).
    """
    lookup = full.set_index(TIME_COLUMN)["pm2_5"]
    pred = lookup.reindex(part[TIME_COLUMN] - pd.Timedelta(hours=18))
    pred = pred.to_numpy()
    keep = ~np.isnan(pred)
    out = compute_metrics(part[TARGET_COLUMN].to_numpy()[keep], pred[keep],
                          prefix, part["pm2_5"].to_numpy()[keep])
    out[f"{prefix}_n_rows"] = int(keep.sum())
    return out

def _git_output(*cmd):
    """Jalankan perintah git, kembalikan teks atau 'unknown'."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_dvc_md5(path):
    """Ambil md5 dari file .dvc, 'unknown' jika tidak ada."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if "md5:" in line:
                    return line.split("md5:")[1].strip()
    return "unknown"


def collect_data_lineage(data_version=None):
    """Tag silsilah data: commit, label versi, md5 pointer DVC."""
    if not data_version:
        data_version = _git_output(
            "git", "describe", "--tags", "--match", "data-v*", "--exact-match"
        )
    return {
        "git_commit": _git_output("git", "rev-parse", "HEAD"),
        "data_version": data_version,
        "data_raw_dvc_md5": read_dvc_md5(RAW_DVC_FILE),
        "data_processed_dvc_md5": read_dvc_md5(PROCESSED_DVC_FILE),
    }


def log_model_artifact(mlflow, model, args):
    """Log model; kompatibel MLflow 2.x (artifact_path) dan 3.x (name)."""
    import mlflow.sklearn
    kwargs = {"skops_trusted_types": TRUSTED_SKOPS_TYPES}
    if args.register_model:
        kwargs["registered_model_name"] = "pm25-karhutla-h6"
    try:
        mlflow.sklearn.log_model(model, name="model", **kwargs)
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path="model", **kwargs)


def train_and_log(args):
    """Latih model, hitung metrik, catat ke MLflow."""
    import mlflow

    df = load_dataset(args.data)
    features = resolve_features(df, args.no_hotspot, args.drop_features)
    df = prepare_training_frame(df, features, args.cutoff)
    train, val, test = chronological_split(
        df, args.val_days, args.test_days, args.gap_hours
    )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    name = args.run_name or (
        f"{args.model}-n{args.n_estimators}-d{args.max_depth}"
        + ("-nohotspot" if args.no_hotspot else "")
        + ("-log" if args.log_target else "")
    )
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "model": args.model,
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "random_state": args.random_state,
                "use_hotspot": not args.no_hotspot,
                "n_features": len(features),
                "features": ",".join(features),
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "gap_hours": args.gap_hours,
                "log_target": args.log_target,
            }
        )
        lineage = collect_data_lineage(args.data_version)
        mlflow.set_tags(lineage)
        mlflow.log_param("data_version", lineage["data_version"])
        model = build_model(args)
        model.fit(train[features], train[TARGET_COLUMN])
        metrics = {}
        for prefix, part in (("val", val), ("test", test)):
            metrics.update(
                compute_metrics(
                    part[TARGET_COLUMN],
                    model.predict(part[features]),
                    prefix,
                    part["pm2_5"],
                )
            )
        metrics.update(compute_baseline_metrics(test, "baseline_test"))
        metrics.update(compute_seasonal_baseline_metrics(
            df, val, "seasonal_val"))
        metrics.update(compute_seasonal_baseline_metrics(
            df, test, "seasonal_test"))
        if args.cv_folds:
            metrics.update(walk_forward_scores(df, features, args))        
        mlflow.log_metrics(metrics)
        importance = compute_feature_importance(
            model, val, features, args.random_state)
        log_feature_importance(mlflow, importance)
        print(importance.to_string(index=False))
        log_model_artifact(mlflow, model, args)
    return metrics


def main(argv=None):
    """Titik masuk skrip; return 0 jika sukses."""
    args = parse_arguments(argv)
    try:
        metrics = train_and_log(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
