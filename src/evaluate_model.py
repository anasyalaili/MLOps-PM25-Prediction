r"""Evaluasi dan validasi model baru sebelum didaftarkan ke registry (gate CI).

Membaca run training CI, lalu memutuskan lolos atau tidak berdasarkan ambang
di config/model_thresholds.toml:

    1. CV RMSE harus lebih rendah dari baseline musiman (selisih minimal
       dapat diatur) dan model harus memenangkan sebagian besar fold.
    2. Champion-challenger (jika sudah ada model Production): model baru
       dan model Production dievaluasi ulang pada holdout 7 hari terakhir
       yang sama; RMSE model baru harus lebih rendah dan precision kelas
       Unhealthy (>150 ug/m3) tidak boleh turun.

Target awal proyek (RMSE, MAE, recall) hanya dilaporkan sebagai informasi.

Hasil ditulis ke JSON (--report), dicetak sebagai tabel Markdown, dan
dicatat ke run (tag gate_passed, gate_summary). Variabel GITHUB_OUTPUT diisi
dengan passed=true/false bila tersedia. Exit code 0 berarti evaluasi
berhasil dijalankan (baik lolos maupun tidak); 1 berarti terjadi error.

Contoh:
    python src/evaluate_model.py --tracking-uri sqlite:///mlflow-ci.db \
        --experiment pm25-karhutla-h6-ci --run-name ci-123-abcdef1
"""

import argparse
import json
import os
import sys
import tomllib
import warnings
from dataclasses import asdict, dataclass

import mlflow
import pandas as pd
from mlflow import MlflowClient

from train import (
    CLASS_THRESHOLDS,
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    TIME_COLUMN,
    compute_metrics,
    load_dataset,
    prepare_training_frame,
    restore_level,
)

DEFAULT_THRESHOLDS = os.path.join("config", "model_thresholds.toml")
DEFAULT_MODEL_NAME = "pm25-karhutla-h6"
UNHEALTHY_LIMIT = int(CLASS_THRESHOLDS[0])
# Selisih RMSE di bawah ini (ug/m3) dianggap sama (derau floating-point).
RMSE_EPSILON = 1e-6


@dataclass
class Check:
    """Satu pemeriksaan gate."""

    name: str
    passed: bool
    detail: str
    blocking: bool = True


def parse_arguments(argv=None):
    """Baca argumen command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--report", default="evaluation.json")
    return parser.parse_args(argv)


def load_thresholds(path):
    """Baca ambang dari berkas TOML."""
    with open(path, "rb") as handle:
        config = tomllib.load(handle)
    return config["blocking"], config.get("targets", {})


def find_run(client, experiment_name, run_name):
    """Cari run training berdasarkan nama."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' tidak ditemukan.")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["start_time DESC"],
    )
    if not runs:
        raise ValueError(f"Run '{run_name}' tidak ditemukan.")
    return runs[0]


def baseline_checks(run, blocking):
    """Periksa model baru terhadap baseline musiman (dari metrik CV)."""
    metrics = run.data.metrics
    needed = ("cv_rmse_mean", "cv_seasonal_rmse_mean", "cv_folds_used")
    missing = [key for key in needed if key not in metrics]
    if missing:
        detail = f"Metrik CV tidak ada: {missing} (jalankan dengan --cv-folds)"
        return [Check("Metrik CV tersedia", False, detail)]
    model_rmse = metrics["cv_rmse_mean"]
    seasonal = metrics["cv_seasonal_rmse_mean"]
    improvement = (seasonal - model_rmse) / seasonal
    required = blocking["min_cv_improvement_vs_seasonal"]
    folds = metrics["cv_folds_used"]
    wins = metrics.get("cv_rmse_wins_vs_seasonal", 0.0)
    win_ratio = wins / folds if folds else 0.0
    ratio_needed = blocking["min_fold_win_ratio"]
    return [
        Check(
            "CV RMSE lebih baik dari baseline musiman",
            improvement > required,
            f"model {model_rmse:.1f} vs baseline {seasonal:.1f} "
            f"(selisih {improvement:.1%}, minimal {required:.1%})",
        ),
        Check(
            "Menang di mayoritas fold vs baseline musiman",
            win_ratio >= ratio_needed,
            f"{wins:.0f} dari {folds:.0f} fold "
            f"(minimal {ratio_needed:.0%})",
        ),
    ]


def target_checks(run, targets):
    """Target awal proyek (informatif, tidak memblokir)."""
    metrics = run.data.metrics
    checks = []
    for label, key, limit, lower_is_better in (
        ("RMSE (test)", "test_rmse", targets.get("rmse"), True),
        ("MAE (test)", "test_mae", targets.get("mae"), True),
        (
            "Recall kelas >150 (test)",
            f"test_gt{UNHEALTHY_LIMIT}_recall",
            targets.get("recall_unhealthy"),
            False,
        ),
    ):
        if limit is None or key not in metrics:
            continue
        value = metrics[key]
        passed = value <= limit if lower_is_better else value >= limit
        sign = "<=" if lower_is_better else ">="
        checks.append(
            Check(
                f"Target awal: {label}",
                passed,
                f"{value:.3f} (target {sign} {limit})",
                blocking=False,
            )
        )
    return checks


def run_features(run):
    """Daftar fitur dan target-mode sebuah run."""
    features = run.data.params["features"].split(",")
    return features, run.data.tags.get("target_mode", "level")


def candidate_model_uri(run):
    """URI model dari run (logged model MLflow 3, atau runs:/ cadangan)."""
    models = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run.info.run_id}'",
        output_format="list",
    )
    if models:
        return f"models:/{models[0].model_id}"
    return f"runs:/{run.info.run_id}/model"


def production_version(client, model_name):
    """Versi model Production (alias lebih dulu, lalu stage), atau None."""
    try:
        return client.get_model_version_by_alias(model_name, "production")
    except mlflow.MlflowException:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            latest = client.get_latest_versions(model_name, ["Production"])
        except mlflow.MlflowException:
            return None
    return latest[0] if latest else None


def holdout_frame(data_path, feature_sets, test_days):
    """Baris holdout (test_days terakhir) yang lengkap untuk semua fitur."""
    union = sorted({name for features in feature_sets for name in features})
    frame = prepare_training_frame(load_dataset(data_path), union)
    end = frame[TIME_COLUMN].max()
    start = end - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
    return frame[frame[TIME_COLUMN] >= start].reset_index(drop=True)


def holdout_metrics(model_uri, features, mode, holdout):
    """Metrik sebuah model pada holdout, pada skala PM2.5."""
    model = mlflow.pyfunc.load_model(model_uri)
    prediction = restore_level(model.predict(holdout[features]), holdout, mode)
    result = compute_metrics(
        holdout[TARGET_COLUMN], prediction, "h", holdout["pm2_5"]
    )
    return {
        "rmse": result["h_rmse"],
        "mae": result["h_mae"],
        f"gt{UNHEALTHY_LIMIT}_precision": result[
            f"h_gt{UNHEALTHY_LIMIT}_precision"
        ],
        f"gt{UNHEALTHY_LIMIT}_recall": result[f"h_gt{UNHEALTHY_LIMIT}_recall"],
        "rows": len(holdout),
    }


def champion_checks(candidate, production, blocking):
    """Bandingkan metrik holdout model baru dengan model Production."""
    key = f"gt{UNHEALTHY_LIMIT}_precision"
    checks = []
    if blocking["require_lower_rmse_than_production"]:
        margin = blocking.get("min_rmse_improvement_vs_production", 0.0)
        gain = production["rmse"] - candidate["rmse"]
        needed = max(production["rmse"] * margin, RMSE_EPSILON)
        relative = gain / production["rmse"] if production["rmse"] else 0.0
        checks.append(
            Check(
                "RMSE holdout lebih rendah dari Production",
                gain > needed,
                f"baru {candidate['rmse']:.2f} vs Production "
                f"{production['rmse']:.2f} (selisih {relative:.1%}, "
                f"minimal {margin:.1%})",
            )
        )
    drop = production[key] - candidate[key]
    limit = blocking["max_precision_drop_vs_production"]
    checks.append(
        Check(
            f"Precision kelas >{UNHEALTHY_LIMIT} tidak turun",
            drop <= limit + 1e-12,
            f"baru {candidate[key]:.3f} vs Production {production[key]:.3f} "
            f"(turun {max(drop, 0):.3f}, batas {limit:.3f})",
        )
    )
    return checks


def render_markdown(report):
    """Tabel Markdown untuk ringkasan job."""
    verdict = "LOLOS" if report["passed"] else "TIDAK LOLOS"
    lines = [
        "### Evaluasi model",
        f"- Run: `{report['run_name']}`",
        f"- Keputusan: **{verdict}**",
    ]
    if report["production"]:
        lines.append(f"- Pembanding: Production versi {report['production']}")
    else:
        lines.append("- Pembanding: belum ada model Production")
    lines += ["", "| Pemeriksaan | Hasil | Rincian |", "|---|---|---|"]
    for check in report["checks"]:
        if check["blocking"]:
            mark = "lolos" if check["passed"] else "GAGAL"
        else:
            mark = "tercapai" if check["passed"] else "belum (informasi)"
        lines.append(f"| {check['name']} | {mark} | {check['detail']} |")
    return "\n".join(lines)


def evaluate(args):
    """Jalankan semua pemeriksaan dan susun laporan."""
    blocking, targets = load_thresholds(args.thresholds)
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    run = find_run(client, args.experiment, args.run_name)

    checks = baseline_checks(run, blocking) + target_checks(run, targets)
    reference = production_version(client, args.model_name)
    holdout_result = {"candidate": None, "production": None}
    if reference is not None:
        ref_run = client.get_run(reference.run_id)
        cand_features, cand_mode = run_features(run)
        ref_features, ref_mode = run_features(ref_run)
        holdout = holdout_frame(
            args.data, [cand_features, ref_features], args.test_days
        )
        candidate = holdout_metrics(
            candidate_model_uri(run), cand_features, cand_mode, holdout
        )
        production = holdout_metrics(
            f"models:/{args.model_name}/{reference.version}",
            ref_features,
            ref_mode,
            holdout,
        )
        holdout_result = {"candidate": candidate, "production": production}
        checks += champion_checks(candidate, production, blocking)

    passed = all(check.passed for check in checks if check.blocking)
    return {
        "passed": passed,
        "run_name": args.run_name,
        "run_id": run.info.run_id,
        "production": reference.version if reference else None,
        "holdout": holdout_result,
        "checks": [asdict(check) for check in checks],
    }


def record_result(report, tracking_uri):
    """Catat keputusan gate ke run (tag) agar terlacak di MLflow."""
    client = MlflowClient(tracking_uri)
    failed = [
        c["name"]
        for c in report["checks"]
        if c["blocking"] and not c["passed"]
    ]
    client.set_tag(report["run_id"], "gate_passed", str(report["passed"]))
    client.set_tag(
        report["run_id"],
        "gate_summary",
        "lolos" if report["passed"] else "gagal: " + "; ".join(failed),
    )
    candidate = report["holdout"]["candidate"]
    if candidate:
        client.log_metric(report["run_id"], "holdout_rmse", candidate["rmse"])


def main(argv=None):
    """Titik masuk skrip."""
    args = parse_arguments(argv)
    try:
        report = evaluate(args)
        record_result(report, args.tracking_uri)
    except (
        ValueError,
        KeyError,
        FileNotFoundError,
        mlflow.MlflowException,
    ) as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return 1
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    markdown = render_markdown(report)
    print(markdown)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")
    output = os.getenv("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"passed={str(report['passed']).lower()}\n")
    if not report["passed"]:
        print("::warning::Model tidak lolos validasi, tidak akan didaftarkan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
