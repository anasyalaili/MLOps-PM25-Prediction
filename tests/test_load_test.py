"""Tes pembangkit beban: pemilihan data tiap skenario."""

import load_test
from train import FEATURES, load_dataset


def test_drift_pool_has_higher_pm25(processed_dir):
    normal, source = load_test.load_pool(processed_dir, "normal", 1)
    drift, _ = load_test.load_pool(processed_dir, "drift", 1)
    assert source == "dataset"
    assert list(normal.columns) == FEATURES
    assert drift["pm2_5"].min() >= load_dataset(processed_dir)[
        "pm2_5"
    ].quantile(0.8)
    assert drift["pm2_5"].mean() > normal["pm2_5"].mean()


def test_fallback_without_dataset(tmp_path):
    normal, source = load_test.load_pool(str(tmp_path), "normal", 1)
    drift, _ = load_test.load_pool(str(tmp_path), "drift", 1)
    assert source == "contoh acak"
    assert drift["pm2_5"].mean() > 3 * normal["pm2_5"].mean()
