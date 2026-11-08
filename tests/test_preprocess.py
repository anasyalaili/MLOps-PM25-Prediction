"""Tes preprocess: urutan file, deduplikasi, validasi, konversi waktu."""

import json

import pandas as pd
import pytest

import preprocess as pp


def write_hourly(path, pm25):
    """Tulis file JSON Open-Meteo mini dengan satu jam yang sama."""
    payload = {"hourly": {"time": ["2026-08-01T00:00"], "pm2_5": [pm25]}}
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestFileOrdering:
    def test_historical_files_sorted_before_live(self, tmp_path, monkeypatch):
        folder = tmp_path / "weather"
        folder.mkdir()
        names = [
            "weather_20260301_000000.json",
            "weather_hist_20230601_20231130_20261101_000000.json",
            "weather_20260101_000000.json",
        ]
        for name in names:
            (folder / name).write_text("{}", encoding="utf-8")
        monkeypatch.setattr(pp, "RAW_DIR", str(tmp_path))
        ordered = [
            p.split("/")[-1] for p in pp.list_raw_files("weather", "json")
        ]
        assert ordered == [
            "weather_hist_20230601_20231130_20261101_000000.json",
            "weather_20260101_000000.json",
            "weather_20260301_000000.json",
        ]

    def test_live_data_wins_over_historical(self, tmp_path, monkeypatch):
        folder = tmp_path / "weather"
        folder.mkdir()
        # File historis dibuat belakangan (timestamp lebih baru) tetapi
        # nilai data live harus tetap menang.
        write_hourly(
            folder / "weather_hist_20230601_20231130_20261101_000000.json",
            999.0,
        )
        write_hourly(folder / "weather_20200101_000000.json", 1.0)
        monkeypatch.setattr(pp, "RAW_DIR", str(tmp_path))
        files = pp.list_raw_files("weather", "json")
        combined = pp.load_hourly_json_files(files)
        assert len(combined) == 1
        assert combined["pm2_5"].iloc[0] == 1.0


class TestCleaning:
    def test_drop_missing_values(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1, 2, 3]})
        assert len(pp.drop_missing_values(df)) == 2

    def test_out_of_range_values_removed(self):
        df = pd.DataFrame(
            {
                "relative_humidity_2m": [50, 120, 60, 70],
                "cloud_cover": [10, 20, 30, 40],
                "pm2_5": [10.0, 10.0, -1.0, 10.0],
                "pm10": [20.0, 20.0, 20.0, 20.0],
                "wind_speed_10m": [3.0, 3.0, 3.0, 3.0],
                "precipitation": [0.0, 0.0, 0.0, 0.0],
            }
        )
        cleaned = pp.remove_out_of_range_values(df)
        assert list(cleaned.index) == [0, 3]

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("n", True),
            ("h", True),
            ("l", False),
            ("nominal", True),
            ("low", False),
            ("80", True),
            ("70", False),
            ("xyz", False),
        ],
    )
    def test_confidence_filter(self, value, expected):
        assert pp.is_confident_detection(value) is expected

    def test_filter_firms_confidence(self):
        df = pd.DataFrame({"confidence": ["n", "l", "h", "30", "90"]})
        kept = pp.filter_firms_confidence(df)
        assert list(kept["confidence"]) == ["n", "h", "90"]


class TestFirmsTime:
    def test_utc_converted_to_jakarta(self):
        df = pd.DataFrame({"acq_date": ["2026-08-01"], "acq_time": ["30"]})
        out = pp.add_firms_jakarta_time(df)
        # 00:30 UTC = 07:30 WIB
        assert out["acq_datetime_jkt"].iloc[0].hour == 7
        assert out["acq_datetime_jkt"].iloc[0].minute == 30
        assert out["acq_hour_jkt"].iloc[0].minute == 0

    def test_rollover_to_next_day(self):
        df = pd.DataFrame({"acq_date": ["2026-08-01"], "acq_time": ["1830"]})
        out = pp.add_firms_jakarta_time(df)
        assert out["acq_datetime_jkt"].iloc[0].day == 2
        assert out["acq_datetime_jkt"].iloc[0].hour == 1

    def test_invalid_times_dropped(self):
        df = pd.DataFrame(
            {
                "acq_date": ["2026-08-01", "bukan-tanggal"],
                "acq_time": ["30", "30"],
            }
        )
        assert len(pp.add_firms_jakarta_time(df)) == 1
