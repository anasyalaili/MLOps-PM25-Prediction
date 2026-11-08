"""Tes build_features: skema v4, hitungan hotspot, dan tanpa kebocoran."""

import numpy as np
import pandas as pd
import pytest

import build_features as bf
from conftest import CUTOFF


class TestGeometry:
    def test_haversine_one_degree_latitude(self):
        distance = bf.haversine_km(
            0.0, 100.0, np.array([1.0]), np.array([100.0])
        )
        assert distance[0] == pytest.approx(111.2, abs=0.2)

    def test_haversine_zero_distance(self):
        distance = bf.haversine_km(
            -2.21, 113.92, np.array([-2.21]), np.array([113.92])
        )
        assert distance[0] == pytest.approx(0.0, abs=1e-9)


class TestWindows:
    def test_window_bounds_and_sum(self):
        times = np.array(
            ["2026-08-01T00:00", "2026-08-01T05:00", "2026-08-02T01:00"],
            dtype="datetime64[ns]",
        )
        hours = pd.Series(
            pd.to_datetime(["2026-08-01 06:00", "2026-08-02 02:00"])
        )
        bounds = bf.window_bounds(times, hours, 24)
        counts = bf.windowed_sum(np.ones(3), bounds)
        # (t-24h, t]: jam 06:00 mencakup 2 titik, 02:00 hari berikutnya
        # mencakup titik 05:00 dan 01:00 (titik 00:00 jatuh di batas bawah).
        assert list(counts) == [2.0, 2.0]

    def test_nearest_distance_defaults_when_empty(self):
        bounds = (np.array([0]), np.array([0]))
        nearest = bf.nearest_distance(np.array([10.0]), bounds)
        assert nearest[0] == bf.NO_HOTSPOT_DISTANCE_KM


class TestSchema:
    def test_columns_match_output_schema(self, processed_frame):
        assert list(processed_frame.columns) == bf.OUTPUT_COLUMNS
        assert len(bf.OUTPUT_COLUMNS) == 39

    def test_no_missing_values(self, processed_frame):
        assert processed_frame.isna().sum().sum() == 0

    def test_integer_columns(self, processed_frame):
        for column in bf.HOTSPOT_INT_COLUMNS + ["european_aqi"]:
            assert pd.api.types.is_integer_dtype(processed_frame[column])

    def test_timestamps_sorted_and_unique(self, processed_frame):
        stamps = processed_frame["timestamp"]
        assert stamps.is_monotonic_increasing
        assert stamps.is_unique

    def test_cutoff_respected(self, processed_frame):
        assert processed_frame["timestamp"].max() <= CUTOFF

    def test_label_is_pm25_six_hours_ahead(
        self, processed_frame, synthetic_weather
    ):
        pm25 = synthetic_weather.assign(
            time=pd.to_datetime(synthetic_weather["time"])
        ).set_index("time")["pm2_5"]
        row = processed_frame.iloc[100]
        future = row["timestamp"] + pd.Timedelta(hours=6)
        assert row["target_pm2_5_h6"] == pytest.approx(pm25[future])

    def test_lags_are_past_values(self, processed_frame, synthetic_weather):
        pm25 = synthetic_weather.assign(
            time=pd.to_datetime(synthetic_weather["time"])
        ).set_index("time")["pm2_5"]
        row = processed_frame.iloc[100]
        for hours in (1, 6, 12, 18, 24):
            past = row["timestamp"] - pd.Timedelta(hours=hours)
            assert row[f"pm2_5_lag_{hours}h"] == pytest.approx(pm25[past])

    def test_cyclic_hour_features(self, processed_frame):
        hours = processed_frame["hour_of_day"]
        assert np.allclose(
            processed_frame["hour_sin"], np.sin(2 * np.pi * hours / 24)
        )
        assert np.allclose(
            processed_frame["hour_cos"], np.cos(2 * np.pi * hours / 24)
        )


class TestHotspotFeatures:
    def test_counts_match_brute_force(self, processed_frame, synthetic_firms):
        local = bf.firms_local_datetime(synthetic_firms)
        dist = bf.haversine_km(
            bf.LATITUDE,
            bf.LONGITUDE,
            synthetic_firms["latitude"].to_numpy(),
            synthetic_firms["longitude"].to_numpy(),
        )
        sample = processed_frame["timestamp"].sample(8, random_state=3)
        for stamp in sample:
            window = (local > stamp - pd.Timedelta(hours=24)) & (
                local <= stamp
            )
            previous = (local > stamp - pd.Timedelta(hours=48)) & (
                local <= stamp - pd.Timedelta(hours=24)
            )
            row = processed_frame[processed_frame["timestamp"] == stamp].iloc[
                0
            ]
            for radius in bf.RADII_KM:
                expected = (window & (dist <= radius)).sum()
                assert row[f"hotspot_count_{radius}km"] == expected
            main = bf.MAIN_RADIUS_KM
            assert row["hotspot_trend_100km"] == (
                (window & (dist <= main)).sum()
                - (previous & (dist <= main)).sum()
            )
            assert row["hotspot_frp_sum_100km"] == pytest.approx(
                synthetic_firms["frp"][window & (dist <= main)].sum()
            )
            nearest = dist[window].min() if window.any() else 400.0
            assert row["hotspot_nearest_km"] == pytest.approx(
                min(nearest, 400.0)
            )

    def test_hotspot_log_matches_count(self, processed_frame):
        assert np.allclose(
            processed_frame["hotspot_count_log"],
            np.log1p(processed_frame["hotspot_count_100km"]),
        )


class TestNoLeakage:
    def test_features_ignore_data_after_time_t(
        self, synthetic_weather, synthetic_firms, processed_frame
    ):
        """Mengubah data setelah T tidak mengubah fitur pada waktu <= T."""
        cut = pd.Timestamp("2026-08-25 12:00")
        weather = synthetic_weather.copy()
        firms = synthetic_firms.copy()
        later = pd.to_datetime(weather["time"]) > cut
        weather.loc[later, ["pm2_5", "temperature_2m", "precipitation"]] *= 3
        shifted = bf.firms_local_datetime(firms) > cut
        firms.loc[shifted, "acq_date"] = "2026-09-01"
        frame, _, _ = bf.build_features(weather, firms, CUTOFF)
        changed = bf.finalize(frame)

        columns = [
            c
            for c in bf.OUTPUT_COLUMNS
            if c not in ("timestamp", "location", "target_pm2_5_h6")
            and not c.endswith("_h6")
        ]
        before = processed_frame[processed_frame["timestamp"] <= cut]
        after = changed[changed["timestamp"] <= cut]
        common = set(before["timestamp"]) & set(after["timestamp"])
        assert len(common) > 100
        left = before[before["timestamp"].isin(common)].set_index("timestamp")
        right = after[after["timestamp"].isin(common)].set_index("timestamp")
        pd.testing.assert_frame_equal(left[columns], right[columns])


class TestPartitions:
    def test_partitions_written_per_month(self, processed_dir):
        import glob

        files = glob.glob(f"{processed_dir}/**/data.parquet", recursive=True)
        assert len(files) >= 2

    def test_roundtrip_preserves_rows(self, processed_dir, processed_frame):
        import glob

        frames = [
            pd.read_parquet(f)
            for f in glob.glob(
                f"{processed_dir}/**/data.parquet", recursive=True
            )
        ]
        assert sum(len(f) for f in frames) == len(processed_frame)
