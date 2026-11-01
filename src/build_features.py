"""Feature engineering: data/interim -> data/processed.

Alur: join weather+AQI, agregasi FIRMS per jam (haversine; beberapa radius,
jendela 24 jam, FRP, jarak terdekat, tren), fitur lag dan statistik
bergulir PM2.5, cuaca pada t+6, fitur temporal, label H+6. Hasil disimpan
per partisi bulan: data/processed/year=YYYY/month=MM/data.parquet.

Skema ini adalah skema v4 (lihat README, bagian Skema Dataset Processed).
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
RADII_KM = (50, 100, 200)
MAIN_RADIUS_KM = 100
WINDOW_HOURS = 24
# Fitur tren membandingkan dua jendela berurutan, jadi cakupan FIRMS
# minimal dua kali jendela agar jam awal tidak terhitung terlalu rendah.
COVERAGE_HOURS = 2 * WINDOW_HOURS
# Jarak yang dipakai jika tidak ada titik api di jendela (di luar bounding
# box FIRMS, sehingga selalu lebih jauh dari titik api yang mungkin ada).
NO_HOTSPOT_DISTANCE_KM = 400.0
HORIZON_HOURS = 6
LAG_HOURS = (1, 6, 12, 18, 24)
ROLLING_HOURS = (6, 24)
FUTURE_WEATHER = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
)
DRY_SEASON_MONTHS = (6, 7, 8, 9, 10)
LOCAL_TZ = "Asia/Jakarta"
INTERIM_DIR = os.path.join("data", "interim")
OUTPUT_DIR = os.path.join("data", "processed")
HOTSPOT_COLUMNS = [
    "hotspot_count_50km",
    "hotspot_count_100km",
    "hotspot_count_200km",
    "hotspot_count_log",
    "hotspot_frp_sum_100km",
    "hotspot_nearest_km",
    "hotspot_trend_100km",
]
HOTSPOT_INT_COLUMNS = [
    "hotspot_count_50km",
    "hotspot_count_100km",
    "hotspot_count_200km",
    "hotspot_trend_100km",
]
OUTPUT_COLUMNS = (
    [
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
    ]
    + [f"pm2_5_lag_{hours}h" for hours in LAG_HOURS]
    + [
        f"pm2_5_roll{stat}_{hours}h"
        for hours in ROLLING_HOURS
        for stat in ("mean", "max")
    ]
    + ["pm2_5_diff_1h", "pm2_5_diff_6h", "precip_sum_24h"]
    + [f"{column}_h{HORIZON_HOURS}" for column in FUTURE_WEATHER]
    + HOTSPOT_COLUMNS
    + [
        "hour_of_day",
        "hour_sin",
        "hour_cos",
        "is_dry_season",
        "target_pm2_5_h6",
    ]
)


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


def window_bounds(times, hours, window_hours, offset_hours=0):
    """Indeks [lower, upper) titik api pada (t - offset - w, t - offset].

    Args:
        times: array datetime64[ns] titik api, sudah terurut naik.
        hours: Series jam t.
        window_hours: Lebar jendela (jam).
        offset_hours: Geser jendela ke belakang (jam).
    """
    hours_ns = hours.values.astype("datetime64[ns]")
    upper_edge = hours_ns - np.timedelta64(offset_hours, "h")
    lower_edge = upper_edge - np.timedelta64(window_hours, "h")
    upper = np.searchsorted(times, upper_edge, side="right")
    lower = np.searchsorted(times, lower_edge, side="right")
    return lower, upper


def windowed_sum(values, bounds):
    """Jumlah values pada tiap jendela (jumlah kumulatif, tanpa loop)."""
    lower, upper = bounds
    cumulative = np.concatenate([[0.0], np.cumsum(values, dtype=float)])
    return cumulative[upper] - cumulative[lower]


def nearest_distance(dist, bounds):
    """Jarak titik api terdekat (km) pada tiap jendela."""
    lower, upper = bounds
    nearest = np.full(len(lower), NO_HOTSPOT_DISTANCE_KM)
    for i, (low, up) in enumerate(zip(lower, upper)):
        if up > low:
            nearest[i] = min(dist[low:up].min(), NO_HOTSPOT_DISTANCE_KM)
    return nearest


def build_hotspot_features(hours, firms):
    """Fitur hotspot per jam; jam di luar cakupan FIRMS jadi NaN.

    Semua fitur memakai titik api dengan waktu akuisisi <= jam t, jadi tidak
    ada kebocoran dari masa depan.
    """
    firms = firms.copy()
    local = firms_local_datetime(firms)
    order = np.argsort(local.values.astype("datetime64[ns]"))
    times = local.values.astype("datetime64[ns]")[order]
    dist = haversine_km(
        LATITUDE,
        LONGITUDE,
        firms["latitude"].astype(float).values,
        firms["longitude"].astype(float).values,
    )[order]
    if "frp" in firms.columns:
        frp = pd.to_numeric(firms["frp"], errors="coerce").fillna(0.0)
        frp = frp.to_numpy()[order]
    else:
        frp = np.zeros(len(firms))

    current = window_bounds(times, hours, WINDOW_HOURS)
    previous = window_bounds(times, hours, WINDOW_HOURS, WINDOW_HOURS)
    out = {}
    for radius in RADII_KM:
        within = (dist <= radius).astype(float)
        out[f"hotspot_count_{radius}km"] = windowed_sum(within, current)
    count_main = out[f"hotspot_count_{MAIN_RADIUS_KM}km"]
    within_main = (dist <= MAIN_RADIUS_KM).astype(float)
    count_previous = windowed_sum(within_main, previous)
    out["hotspot_count_log"] = np.log1p(count_main)
    out["hotspot_frp_sum_100km"] = windowed_sum(frp * within_main, current)
    out["hotspot_nearest_km"] = nearest_distance(dist, current)
    out["hotspot_trend_100km"] = count_main - count_previous

    features = pd.DataFrame(out)[HOTSPOT_COLUMNS]
    start = local.min().floor("h")
    end = local.max().floor("h")
    covered = ((hours - pd.Timedelta(hours=COVERAGE_HOURS - 1)) >= start) & (
        hours <= end
    )
    features[~covered.to_numpy()] = np.nan
    return features, start, end


def add_pm25_features(df):
    """Lag, statistik bergulir, dan selisih PM2.5 (hanya data <= t)."""
    pm25 = df["pm2_5"]
    for hours in LAG_HOURS:
        df[f"pm2_5_lag_{hours}h"] = pm25.shift(hours)
    for hours in ROLLING_HOURS:
        rolling = pm25.rolling(hours, min_periods=hours)
        df[f"pm2_5_rollmean_{hours}h"] = rolling.mean()
        df[f"pm2_5_rollmax_{hours}h"] = rolling.max()
    df["pm2_5_diff_1h"] = pm25 - df["pm2_5_lag_1h"]
    df["pm2_5_diff_6h"] = pm25 - df["pm2_5_lag_6h"]
    df["precip_sum_24h"] = (
        df["precipitation"].rolling(24, min_periods=24).sum()
    )
    return df


def add_future_weather(df):
    """Cuaca pada t+6 (di inferensi berasal dari prakiraan cuaca)."""
    for column in FUTURE_WEATHER:
        df[f"{column}_h{HORIZON_HOURS}"] = df[column].shift(-HORIZON_HOURS)
    return df


def build_features(weather, firms, cutoff):
    """Bangun tabel fitur lengkap (skema v4)."""
    df = weather.rename(columns={"time": "timestamp"}).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp", keep="last")
    df = df[df["timestamp"] <= cutoff].sort_values("timestamp")
    full = pd.date_range(
        df["timestamp"].min(), df["timestamp"].max(), freq="h"
    )
    df = df.set_index("timestamp").reindex(full)
    df.index.name = "timestamp"

    df = add_pm25_features(df)
    df = add_future_weather(df)
    df["target_pm2_5_h6"] = df["pm2_5"].shift(-HORIZON_HOURS)
    df = df.reset_index()

    hotspots, start, end = build_hotspot_features(df["timestamp"], firms)
    for column in HOTSPOT_COLUMNS:
        df[column] = hotspots[column].to_numpy()
    hour = df["timestamp"].dt.hour
    df["hour_of_day"] = hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
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
    for column in HOTSPOT_INT_COLUMNS:
        df[column] = df[column].round().astype(int)
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
