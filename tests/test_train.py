"""Tes train: fitur, split kronologis, target, metrik, dan run MLflow."""

import mlflow
import numpy as np
import pandas as pd
import pytest

import train


class TestTarget:
    @pytest.mark.parametrize("mode", train.TARGET_MODES)
    def test_roundtrip_returns_original_level(self, mode):
        frame = pd.DataFrame(
            {
                "pm2_5": [10.0, 200.0, 800.0],
                "target_pm2_5_h6": [30.0, 150.0, 1200.0],
            }
        )
        model_target = train.to_model_target(frame, mode)
        restored = train.restore_level(model_target, frame, mode)
        assert np.allclose(restored, frame["target_pm2_5_h6"])

    def test_delta_target_is_difference(self):
        frame = pd.DataFrame({"pm2_5": [10.0], "target_pm2_5_h6": [25.0]})
        assert train.to_model_target(frame, "delta")[0] == 15.0

    def test_restored_level_never_negative(self):
        frame = pd.DataFrame({"pm2_5": [10.0], "target_pm2_5_h6": [0.0]})
        assert train.restore_level([-50.0], frame, "delta")[0] == 0.0


class TestFeatures:
    def test_default_features_exist_in_dataset(self, processed_frame):
        features = train.resolve_features(processed_frame)
        assert len(features) == 34
        assert set(features) <= set(processed_frame.columns)
        assert "is_dry_season" not in features
        assert "hotspot_count_log" not in features

    def test_no_hotspot_removes_all_hotspot_features(self, processed_frame):
        features = train.resolve_features(processed_frame, no_hotspot=True)
        assert not set(features) & set(train.HOTSPOT_FEATURES)
        assert len(features) == 28

    def test_optional_features_can_be_added(self, processed_frame):
        features = train.resolve_features(
            processed_frame, extra=["is_dry_season", "hotspot_count_log"]
        )
        assert "is_dry_season" in features
        assert len(features) == 36

    def test_unknown_optional_feature_rejected(self, processed_frame):
        with pytest.raises(ValueError, match="opsional"):
            train.resolve_features(processed_frame, extra=["nonsense"])

    def test_unknown_drop_rejected(self, processed_frame):
        with pytest.raises(ValueError, match="bukan fitur"):
            train.resolve_features(processed_frame, drop=["nonsense"])

    def test_delta_mode_requires_current_pm25(self, processed_frame):
        with pytest.raises(ValueError, match="pm2_5"):
            train.resolve_features(
                processed_frame, drop=["pm2_5"], target_mode="delta"
            )

    def test_old_schema_gives_clear_error(self, processed_frame):
        old = processed_frame[["timestamp", "pm2_5", "target_pm2_5_h6"]]
        with pytest.raises(ValueError, match="skema lama"):
            train.resolve_features(old)

    def test_parse_max_features(self):
        assert train.parse_max_features("0.5") == 0.5
        assert train.parse_max_features("3") == 3
        assert train.parse_max_features("sqrt") == "sqrt"
        assert train.parse_max_features("none") is None


class TestSplit:
    def test_chronological_without_overlap(self, processed_frame):
        features = train.resolve_features(processed_frame)
        frame = train.prepare_training_frame(processed_frame, features)
        tr, va, te = train.chronological_split(frame, 7, 7, 6)
        assert tr["timestamp"].max() < va["timestamp"].min()
        assert va["timestamp"].max() < te["timestamp"].min()
        gap = pd.Timedelta(hours=6)
        assert va["timestamp"].min() - tr["timestamp"].max() > gap
        assert te["timestamp"].min() - va["timestamp"].max() >= gap
        assert te["timestamp"].max() == frame["timestamp"].max()

    def test_too_little_data_raises(self, processed_frame):
        features = train.resolve_features(processed_frame)
        frame = train.prepare_training_frame(processed_frame, features)
        with pytest.raises(ValueError, match="hanya"):
            train.chronological_split(frame.head(100), 7, 7, 6)

    def test_cutoff_filters_rows(self, processed_frame):
        features = train.resolve_features(processed_frame)
        cut = "2026-09-01 00:00"
        frame = train.prepare_training_frame(processed_frame, features, cut)
        assert frame["timestamp"].max() <= pd.Timestamp(cut)


class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([50.0, 200.0, 300.0, 600.0])
        out = train.compute_metrics(y, y, "t", current=np.array([40.0] * 4))
        assert out["t_rmse"] == 0.0
        assert out["t_mae"] == 0.0
        assert out["t_rmse_log"] == 0.0
        assert out["t_gt150_precision"] == 1.0
        assert out["t_gt250_recall"] == 1.0

    def test_expected_keys(self):
        y = np.array([50.0, 200.0, 300.0, 600.0])
        out = train.compute_metrics(y, y, "t", current=y)
        for name in ("gt150", "gt250", "gt500", "spike"):
            assert f"t_{name}_precision" in out
            assert f"t_{name}_recall" in out

    def test_rmse_value(self):
        out = train.compute_metrics([0.0, 0.0], [3.0, 4.0], "t")
        assert out["t_rmse"] == pytest.approx(np.sqrt(12.5))

    def test_seasonal_baseline_is_perfect_for_daily_cycle(self):
        times = pd.date_range("2026-08-01", periods=24 * 10, freq="h")
        cycle = 100 + 50 * np.sin(times.hour.to_numpy() / 24 * 2 * np.pi)
        full = pd.DataFrame({"timestamp": times, "pm2_5": cycle})
        full["target_pm2_5_h6"] = full["pm2_5"].shift(-6)
        part = full.dropna().iloc[24 * 3 : 24 * 6]
        out = train.compute_seasonal_baseline_metrics(full, part, "s")
        assert out["s_rmse"] == pytest.approx(0.0, abs=1e-9)
        assert out["s_n_rows"] == len(part)

    def test_persistence_baseline_uses_current_value(self):
        part = pd.DataFrame(
            {"pm2_5": [100.0, 100.0], "target_pm2_5_h6": [110.0, 90.0]}
        )
        out = train.compute_baseline_metrics(part, "b")
        assert out["b_rmse"] == pytest.approx(10.0)


class TestModels:
    @pytest.mark.parametrize("name", ["rf", "extra", "gbr", "linear", "xgb"])
    def test_every_model_fits_and_predicts(self, name):
        args = train.parse_arguments(
            ["--model", name, "--n-estimators", "5", "--max-depth", "3"]
        )
        rng = np.random.default_rng(0)
        x = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
        y = x["a"] * 2 + 1
        model = train.build_model(args)
        model.fit(x, y)
        assert model.predict(x).shape == (60,)


class TestTrainRun:
    def test_smoke_run_logs_to_mlflow(
        self, processed_dir, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        uri = f"sqlite:///{tmp_path}/mlflow.db"
        code = train.main(
            [
                "--data", processed_dir,
                "--tracking-uri", uri,
                "--experiment", "smoke",
                "--run-name", "smoke-delta",
                "--data-version", "test",
                "--n-estimators", "5",
                "--max-depth", "4",
                "--cv-folds", "2",
                "--target-mode", "delta",
            ]
        )  # fmt: skip
        assert code == 0
        mlflow.set_tracking_uri(uri)
        runs = mlflow.search_runs(experiment_names=["smoke"])
        assert len(runs) == 1
        run = runs.iloc[0]
        assert run["tags.target_mode"] == "delta"
        assert run["tags.data_version"] == "test"
        assert run["params.n_features"] == "34"
        for metric in (
            "metrics.test_rmse",
            "metrics.cv_rmse_mean",
            "metrics.cv_seasonal_rmse_mean",
            "metrics.cv_rmse_wins_vs_seasonal",
        ):
            assert not pd.isna(run[metric])

    def test_old_schema_dataset_returns_error(self, tmp_path, capsys):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-08-01", periods=48, freq="h"),
                "pm2_5": 1.0,
                "target_pm2_5_h6": 1.0,
            }
        )
        path = tmp_path / "old.parquet"
        frame.to_parquet(path)
        code = train.main(
            [
                "--data",
                str(path),
                "--tracking-uri",
                f"sqlite:///{tmp_path}/m.db",
            ]
        )
        assert code == 1
        assert "ERROR" in capsys.readouterr().out
