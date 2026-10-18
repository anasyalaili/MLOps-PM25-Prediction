"""Feature engineering: data/interim -> data/processed.

Alur: join weather+AQI, agregasi FIRMS per jam (haversine, radius 100 km,
jendela 24 jam), fitur lag, fitur temporal, label H+6. Hasil disimpan per
partisi bulan: data/processed/year=YYYY/month=MM/data.parquet.
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

LATITUDE = -2.21
LONGITUDE = 113.92
LOCATION = "Palangkaraya"
RADIUS_KM = 100.0
WINDOW_HOURS = 24
HORIZON_HOURS = 6
DRY_SEASON_MONTHS = (6, 7, 8, 9, 10)
LOCAL_TZ = "Asia/Jakarta"
INTERIM_DIR = os.path.join("data", "interim")
OUTPUT_DIR = os.path.join("data", "processed")
OUTPUT_COLUMNS = [
    "timestamp",
    "location",
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
    "target_pm2_5_h6",
]


def parse_arguments(argv=None):
    """Baca argumen command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interim-dir", default=INTERIM_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--cutoff",
        default=None,
        help="Jam observasi terakhir (WIB, 'YYYY-MM-DD HH:MM'). Baris "
        "setelahnya dianggap prakiraan dan dibuang. Default: jam "
        "sekarang WIB.",
    )
    return parser.parse_args(argv)


def haversine_km(lat1, lon1, lat2, lon2):
    """Jarak great-circle (km); lat2/lon2 boleh berupa array."""
    radius_earth = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = np.radians(lon2) - math.radians(lon1)
    a = (
        np.sin(d_phi / 2) ** 2
        + math.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_earth * np.arcsin(np.sqrt(a))


def firms_local_datetime(firms):
    """Kolom waktu lokal (WIB, naive) tiap titik api."""
    if "acq_datetime_local" in firms.columns:
        return pd.to_datetime(firms["acq_datetime_local"])
    hhmm = firms["acq_time"].astype(int).astype(str).str.zfill(4)
    utc = pd.to_datetime(
        firms["acq_date"].astype(str) + " " + hhmm, format="%Y-%m-%d %H%M"
    )
    return (
        utc.dt.tz_localize("UTC").dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    )


def count_hotspots(hours, hotspot_times):
    """Jumlah titik api pada (t - 24 jam, t] untuk setiap jam t."""
    times = np.sort(hotspot_times.values.astype("datetime64[ns]"))
    upper = np.searchsorted(
        times, hours.values.astype("datetime64[ns]"), side="right"
    )
    lower_edge = (hours - pd.Timedelta(hours=WINDOW_HOURS)).values
    lower = np.searchsorted(
        times, lower_edge.astype("datetime64[ns]"), side="right"
    )
    return upper - lower


def build_hotspot_feature(hours, firms):
    """Hitung hotspot_count_100km; jam di luar cakupan FIRMS jadi NaN."""
    firms = firms.copy()
    firms["dt_local"] = firms_local_datetime(firms)
    dist = haversine_km(
        LATITUDE,
        LONGITUDE,
        firms["latitude"].astype(float).values,
        firms["longitude"].astype(float).values,
    )
    near = firms.loc[dist <= RADIUS_KM, "dt_local"]
    counts = count_hotspots(hours, near).astype(float)
    start = firms["dt_local"].min().floor("h")
    end = firms["dt_local"].max().floor("h")
    covered = ((hours - pd.Timedelta(hours=WINDOW_HOURS - 1)) >= start) & (
        hours <= end
    )
    counts[~covered.values] = np.nan
    return counts, start, end


def build_features(weather, firms, cutoff):
    """Bangun tabel fitur lengkap sesuai skema LK-03."""
    df = weather.rename(columns={"time": "timestamp"}).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp", keep="last")
    df = df[df["timestamp"] <= cutoff].sort_values("timestamp")
    full = pd.date_range(
        df["timestamp"].min(), df["timestamp"].max(), freq="h"
    )
    df = df.set_index("timestamp").reindex(full)
    df.index.name = "timestamp"

    df["pm2_5_lag_1h"] = df["pm2_5"].shift(1)
    df["pm2_5_lag_6h"] = df["pm2_5"].shift(6)
    df["target_pm2_5_h6"] = df["pm2_5"].shift(-HORIZON_HOURS)
    df = df.reset_index()

    counts, start, end = build_hotspot_feature(df["timestamp"], firms)
    df["hotspot_count_100km"] = counts
    df["hotspot_count_log"] = np.log1p(counts)
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_dry_season"] = (
        df["timestamp"].dt.month.isin(DRY_SEASON_MONTHS).astype(int)
    )
    df["location"] = LOCATION
    return df, start, end


def finalize(df):
    """Buang baris tak lengkap dan atur tipe sesuai skema."""
    before = len(df)
    df = df.dropna(subset=[c for c in OUTPUT_COLUMNS if c != "location"])
    df = df[OUTPUT_COLUMNS].copy()
    df["hotspot_count_100km"] = df["hotspot_count_100km"].astype(int)
    df["european_aqi"] = df["european_aqi"].round().astype(int)
    print(
        f"Baris dibuang (tanpa label/lag/cakupan FIRMS): "
        f"{before - len(df)} dari {before}"
    )
    return df.reset_index(drop=True)


def save_partitions(df, output_dir):
    """Tulis per bulan; partisi bulan yang sama ditimpa (idempoten)."""
    try:
        import pyarrow  # noqa: F401

        extension = "parquet"
    except ImportError:
        extension = "csv"
        print("pyarrow tidak ada, menyimpan CSV (pasang pyarrow!).")
    paths = []
    for (year, month), part in df.groupby(
        [df["timestamp"].dt.year, df["timestamp"].dt.month]
    ):
        folder = os.path.join(output_dir, f"year={year}", f"month={month:02d}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"data.{extension}")
        if extension == "parquet":
            part.to_parquet(path, index=False)
        else:
            part.to_csv(path, index=False)
        paths.append(path)
        print(f"  {path}: {len(part)} baris")
    return paths


def main(argv=None):
    """Titik masuk skrip."""
    args = parse_arguments(argv)
    weather_path = os.path.join(args.interim_dir, "weather_aqi_clean.csv")
    firms_path = os.path.join(args.interim_dir, "firms_clean.csv")
    for path in (weather_path, firms_path):
        if not os.path.exists(path):
            print(f"ERROR: {path} tidak ada. Jalankan preprocess.py dulu.")
            return 1
    cutoff = (
        pd.Timestamp(args.cutoff)
        if args.cutoff
        else pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None).floor("h")
    )
    weather = pd.read_csv(weather_path)
    firms = pd.read_csv(firms_path)
    df, start, end = build_features(weather, firms, cutoff)
    print(f"Cutoff observasi (WIB): {cutoff}")
    print(f"Cakupan FIRMS (WIB): {start} s.d. {end}")
    result = finalize(df)
    if result.empty:
        print("ERROR: hasil kosong. Cek cutoff dan cakupan FIRMS.")
        return 1
    print(
        f"Rentang processed: {result['timestamp'].min()} s.d. "
        f"{result['timestamp'].max()} ({len(result)} baris)"
    )
    save_partitions(result, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
