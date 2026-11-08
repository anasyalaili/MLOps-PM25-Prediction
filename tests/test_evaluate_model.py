"""Tes evaluate_model: logika gate dan alur evaluasi pada store sementara."""

import json
import os
from types import SimpleNamespace

import pytest
from mlflow import MlflowClient

import evaluate_model as ev
import register_model as rm
import train

EXPERIMENT = "pm25-karhutla-h6-ci"
STRICT = """
[blocking]
min_cv_improvement_vs_seasonal = 0.99
min_fold_win_ratio = 0.75
require_lower_rmse_than_production = true
max_precision_drop_vs_production = 0.0
[targets]
rmse = 15.0
mae = 10.0
recall_unhealthy = 0.85
"""
RELAXED = """
[blocking]
min_cv_improvement_vs_seasonal = -10.0
min_fold_win_ratio = 0.0
require_lower_rmse_than_production = false
max_precision_drop_vs_production = 1.0
[targets]
rmse = 15.0
mae = 10.0
recall_unhealthy = 0.85
"""
BLOCKING = {
    "min_cv_improvement_vs_seasonal": 0.0,
    "min_fold_win_ratio": 0.75,
    "require_lower_rmse_than_production": True,
    "max_precision_drop_vs_production": 0.0,
}


def fake_run(**metrics):
    return SimpleNamespace(data=SimpleNamespace(metrics=metrics))


def cv_run(model, seasonal, wins=4.0, folds=4.0):
    return fake_run(
        cv_rmse_mean=model,
        cv_seasonal_rmse_mean=seasonal,
        cv_rmse_wins_vs_seasonal=wins,
        cv_folds_used=folds,
    )


class TestBaselineChecks:
    def test_better_than_baseline_passes(self):
        checks = ev.baseline_checks(cv_run(300.0, 354.0), BLOCKING)
        assert all(check.passed for check in checks)

    def test_worse_than_baseline_fails(self):
        checks = ev.baseline_checks(cv_run(400.0, 354.0), BLOCKING)
        assert not checks[0].passed

    def test_margin_is_enforced(self):
        strict = dict(BLOCKING, min_cv_improvement_vs_seasonal=0.2)
        checks = ev.baseline_checks(cv_run(300.0, 354.0), strict)
        assert not checks[0].passed  # hanya sekitar 15% lebih baik

    def test_needs_enough_fold_wins(self):
        checks = ev.baseline_checks(cv_run(300.0, 354.0, wins=2.0), BLOCKING)
        assert checks[0].passed and not checks[1].passed

    def test_missing_cv_metrics_fail_clearly(self):
        checks = ev.baseline_checks(fake_run(test_rmse=1.0), BLOCKING)
        assert len(checks) == 1 and not checks[0].passed
        assert "cv-folds" in checks[0].detail


class TestTargets:
    def test_targets_are_informational(self):
        run = fake_run(test_rmse=37.5, test_mae=20.0, test_gt150_recall=0.9)
        checks = ev.target_checks(
            run, {"rmse": 15.0, "mae": 10.0, "recall_unhealthy": 0.85}
        )
        assert [c.blocking for c in checks] == [False, False, False]
        assert [c.passed for c in checks] == [False, False, True]


class TestChampionChecks:
    GOOD = {"rmse": 30.0, "gt150_precision": 0.95}
    BAD = {"rmse": 50.0, "gt150_precision": 0.90}

    def test_better_model_passes(self):
        checks = ev.champion_checks(self.GOOD, self.BAD, BLOCKING)
        assert all(check.passed for check in checks)

    def test_higher_rmse_fails(self):
        checks = ev.champion_checks(self.BAD, self.GOOD, BLOCKING)
        assert not checks[0].passed

    def test_precision_drop_fails_even_if_rmse_improves(self):
        lower_precision = {"rmse": 20.0, "gt150_precision": 0.80}
        checks = ev.champion_checks(lower_precision, self.GOOD, BLOCKING)
        assert checks[0].passed and not checks[1].passed

    def test_small_precision_drop_allowed_by_tolerance(self):
        tolerant = dict(BLOCKING, max_precision_drop_vs_production=0.2)
        slightly = {"rmse": 20.0, "gt150_precision": 0.90}
        checks = ev.champion_checks(slightly, self.GOOD, tolerant)
        assert checks[1].passed

    def test_identical_model_is_not_lower(self):
        same = {"rmse": 30.0, "gt150_precision": 0.95}
        checks = ev.champion_checks(dict(same), dict(same), BLOCKING)
        assert not checks[0].passed

    def test_floating_point_noise_is_not_an_improvement(self):
        noisy = {"rmse": 30.0 - 1e-13, "gt150_precision": 0.95}
        checks = ev.champion_checks(noisy, dict(self.GOOD), BLOCKING)
        assert not checks[0].passed

    def test_improvement_margin_is_enforced(self):
        strict = dict(BLOCKING, min_rmse_improvement_vs_production=0.5)
        # 30 vs 50 = 40% lebih baik, kurang dari 50% yang diminta.
        checks = ev.champion_checks(self.GOOD, self.BAD, strict)
        assert not checks[0].passed
        lenient = dict(BLOCKING, min_rmse_improvement_vs_production=0.3)
        assert ev.champion_checks(self.GOOD, self.BAD, lenient)[0].passed

    def test_rmse_rule_can_be_disabled(self):
        relaxed = dict(BLOCKING, require_lower_rmse_than_production=False)
        checks = ev.champion_checks(self.BAD, self.GOOD, relaxed)
        assert len(checks) == 1


@pytest.fixture(scope="module")
def ci_store(processed_dir, tmp_path_factory):
    """Store CI sementara: satu model Production dan satu kandidat."""
    folder = tmp_path_factory.mktemp("ci")
    uri = f"sqlite:///{folder}/mlflow-ci.db"
    previous = os.getcwd()
    os.chdir(folder)
    try:
        for name, depth, register in (
            ("prod", "3", True),
            ("cand", "6", False),
        ):
            args = [
                "--data", processed_dir,
                "--tracking-uri", uri,
                "--experiment", EXPERIMENT,
                "--run-name", name,
                "--data-version", "test",
                "--n-estimators", "8",
                "--max-depth", depth,
                "--cv-folds", "2",
            ]  # fmt: skip
            if register:
                args.append("--register-model")
            assert train.main(args) == 0
        yield {"uri": uri, "folder": folder, "data": processed_dir}
    finally:
        os.chdir(previous)


def write_thresholds(folder, text, name="thresholds.toml"):
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def evaluate_cli(store, thresholds, run="cand", extra=()):
    report = store["folder"] / f"report-{run}.json"
    code = ev.main(
        [
            "--tracking-uri", store["uri"],
            "--experiment", EXPERIMENT,
            "--run-name", run,
            "--data", store["data"],
            "--thresholds", thresholds,
            "--report", str(report),
            *extra,
        ]
    )  # fmt: skip
    return code, report


class TestEvaluation:
    def test_without_production_only_baseline_gate_applies(
        self, ci_store, capsys
    ):
        thresholds = write_thresholds(ci_store["folder"], RELAXED)
        code, report = evaluate_cli(ci_store, thresholds)
        data = json.loads(report.read_text(encoding="utf-8"))
        assert code == 0
        assert data["production"] is None
        assert data["holdout"]["production"] is None
        assert data["passed"] is True
        assert "belum ada model Production" in capsys.readouterr().out

    def test_impossible_threshold_fails_gate_but_not_job(self, ci_store):
        thresholds = write_thresholds(
            ci_store["folder"], STRICT, "strict.toml"
        )
        code, report = evaluate_cli(ci_store, thresholds)
        data = json.loads(report.read_text(encoding="utf-8"))
        assert (
            code == 0
        )  # evaluasi berhasil dijalankan, hasilnya "tidak lolos"
        assert data["passed"] is False
        assert any(not c["passed"] and c["blocking"] for c in data["checks"])

    def test_gate_result_is_tagged_on_run(self, ci_store):
        client = MlflowClient(ci_store["uri"])
        experiment = client.get_experiment_by_name(EXPERIMENT)
        run = client.search_runs(
            [experiment.experiment_id],
            filter_string="attributes.run_name = 'cand'",
        )[0]
        assert run.data.tags["gate_passed"] in ("True", "False")
        assert run.data.tags["gate_summary"]

    def test_champion_comparison_with_production(self, ci_store):
        rm.main(
            [
                "--tracking-uri", ci_store["uri"],
                "promote", "--version", "1", "--stage", "Production",
            ]
        )  # fmt: skip
        thresholds = write_thresholds(ci_store["folder"], RELAXED)
        code, report = evaluate_cli(ci_store, thresholds)
        data = json.loads(report.read_text(encoding="utf-8"))
        assert code == 0
        assert data["production"] == 1
        candidate = data["holdout"]["candidate"]
        production = data["holdout"]["production"]
        assert candidate["rows"] == production["rows"] > 0
        names = [c["name"] for c in data["checks"]]
        assert any("Precision kelas" in n for n in names)

    def test_strict_champion_rules_can_block(self, ci_store):
        text = RELAXED.replace(
            "max_precision_drop_vs_production = 1.0",
            "max_precision_drop_vs_production = -1.0",
        )
        thresholds = write_thresholds(ci_store["folder"], text, "block.toml")
        _, report = evaluate_cli(ci_store, thresholds)
        data = json.loads(report.read_text(encoding="utf-8"))
        assert data["passed"] is False

    def test_github_output_is_written(self, ci_store, monkeypatch):
        output = ci_store["folder"] / "gh_output.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        thresholds = write_thresholds(ci_store["folder"], RELAXED)
        evaluate_cli(ci_store, thresholds)
        assert output.read_text(encoding="utf-8").strip() in (
            "passed=true",
            "passed=false",
        )

    def test_unknown_run_returns_error(self, ci_store, capsys):
        thresholds = write_thresholds(ci_store["folder"], RELAXED)
        code, _ = evaluate_cli(ci_store, thresholds, run="tidak-ada")
        assert code == 1
        assert "tidak ditemukan" in capsys.readouterr().out


class TestReport:
    def test_markdown_contains_verdict_and_checks(self):
        report = {
            "passed": False,
            "run_name": "ci-1",
            "production": None,
            "checks": [
                {
                    "name": "A",
                    "passed": False,
                    "detail": "x",
                    "blocking": True,
                },
                {
                    "name": "B",
                    "passed": False,
                    "detail": "y",
                    "blocking": False,
                },
            ],
        }
        text = ev.render_markdown(report)
        assert "TIDAK LOLOS" in text
        assert "| A | GAGAL | x |" in text
        assert "belum (informasi)" in text
