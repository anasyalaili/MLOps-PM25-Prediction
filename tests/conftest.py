"""Fixture bersama untuk tes: data sintetis kecil, tanpa jaringan."""

import numpy as np
import pandas as pd
import pytest

import build_features as bf

START = "2026-08-01"
DAYS = 60
CUTOFF = pd.Timestamp("2026-09-29 00:00")


@pytest.fixture(scope="session")
def synthetic_weather():
    """Data cuaca + kualitas udara per jam (format interim)."""
    rng = np.random.default_rng(0)
    times = pd.date_range(START, periods=24 * DAYS, freq="h")
    hours = times.hour.to_numpy()
    n = len(times)
    daily_level = np.repeat(rng.uniform(80, 400, DAYS), 24)
    return pd.DataFrame(
        {
            "time": times.astype(str),
            "temperature_2m": 25 + 5 * np.sin(hours / 24 * 2 * np.pi),
            "relative_humidity_2m": rng.uniform(50, 95, n),
            "wind_speed_10m": rng.uniform(1, 15, n),
            "precipitation": rng.choice([0.0, 0.0, 0.0, 1.5], n),
            "cloud_cover": rng.uniform(0, 100, n),
            "pm2_5": (
                daily_level * (1 + 0.4 * np.sin(hours / 24 * 2 * np.pi))
                + rng.normal(0, 10, n)
            ).clip(5),
            "pm10": rng.uniform(100, 400, n),
            "european_aqi": rng.integers(100, 500, n),
        }
    )


@pytest.fixture(scope="session")
def synthetic_firms():
    """Titik api acak di sekitar kota target, mulai 2 hari sebelum cuaca."""
    rng = np.random.default_rng(1)
    n = 1500
    begin = pd.Timestamp(START) - pd.Timedelta(days=2)
    stamps = begin + pd.to_timedelta(rng.integers(0, (DAYS + 2) * 24, n), "h")
    return pd.DataFrame(
        {
            "latitude": bf.LATITUDE + rng.uniform(-1.5, 1.5, n),
            "longitude": bf.LONGITUDE + rng.uniform(-1.5, 1.5, n),
            "acq_date": stamps.strftime("%Y-%m-%d"),
            "acq_time": rng.choice([130, 600, 1830], n),
            "frp": rng.uniform(1, 50, n),
        }
    )


@pytest.fixture(scope="session")
def raw_features(synthetic_weather, synthetic_firms):
    """Keluaran build_features sebelum finalize (masih ada NaN di tepi)."""
    frame, _, _ = bf.build_features(synthetic_weather, synthetic_firms, CUTOFF)
    return frame


@pytest.fixture(scope="session")
def processed_frame(raw_features):
    """Dataset skema v4 siap latih."""
    return bf.finalize(raw_features.copy())


@pytest.fixture(scope="session")
def processed_dir(tmp_path_factory, processed_frame):
    """Dataset skema v4 yang disimpan per partisi bulan."""
    folder = tmp_path_factory.mktemp("processed")
    bf.save_partitions(processed_frame, str(folder))
    return str(folder)
