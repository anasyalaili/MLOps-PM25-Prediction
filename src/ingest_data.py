"""Data ingestion untuk proyek prediksi PM2.5 akibat karhutla.

Mengambil data dinamis dari tiga sumber API publik dan menyimpannya
sebagai data mentah (raw) di folder data/raw/:

    1. Open-Meteo Weather API -> data/raw/weather/*.json
    2. Open-Meteo Air Quality -> data/raw/air_quality/*.json
    3. NASA FIRMS (titik api) -> data/raw/firms/*.csv

Setiap run memberi timestamp UTC pada nama file, sehingga run
berulang tidak pernah menimpa data lama (non-destruktif).

Satu kali run mengambil PAST_DAYS hari ke belakang ditambah data
prakiraan, sehingga satu run saja sudah menghasilkan lebih dari 500
baris data per jam (target volume minimal).

Cara run:
    python src/ingest_data.py

Environment variable (disimpan di file .env, opsional):
    FIRMS_MAP_KEY : API key NASA FIRMS. Jika kosong, FIRMS dilewati.
"""

import csv
import io
import json
import os
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

# Lokasi target: Palangkaraya, Kalimantan Tengah.
LATITUDE = -2.21
LONGITUDE = 113.92
TIMEZONE_NAME = "Asia/Jakarta"

# Open-Meteo mengizinkan past_days 0-92. Dengan 30 hari ke belakang
# plus prakiraan (air quality 5 hari), hasil gabungan per jam
# berjumlah sekitar 840 baris.
PAST_DAYS = 30

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

FIRMS_SOURCE = "VIIRS_SNPP_NRT"
# Format bounding box: min_lon,min_lat,max_lon,max_lat (Kalimantan Tengah).
FIRMS_BOUNDING_BOX = "110.5,-3.5,115.5,1.0"
# Batas maksimal day range untuk source ini adalah 5 hari.
FIRMS_DAY_RANGE = 5
FIRMS_TARGET_DAYS = 30
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "")

WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
)
AIR_QUALITY_VARIABLES = ("pm2_5", "pm10", "european_aqi")

RAW_DIR = os.path.join("data", "raw")
RAW_SUBFOLDERS = ("weather", "air_quality", "firms")
REQUEST_TIMEOUT_SECONDS = 30


def build_run_timestamp():
    """Buat timestamp UTC (YYYYMMDD_HHMMSS) untuk nama file satu run.

    Returns:
        str: Timestamp yang dipakai bersama oleh semua file dalam satu
            run ingestion.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def create_raw_directories():
    """Buat folder data/raw/<sumber> jika belum ada."""
    for subfolder in RAW_SUBFOLDERS:
        os.makedirs(os.path.join(RAW_DIR, subfolder), exist_ok=True)


def build_raw_filepath(subfolder, prefix, run_timestamp, extension):
    """Susun path file mentah berdasarkan sumber dan timestamp.

    Args:
        subfolder (str): Nama subfolder sumber di dalam data/raw/.
        prefix (str): Awalan nama file.
        run_timestamp (str): Timestamp run ingestion.
        extension (str): Ekstensi file tanpa titik (json atau csv).

    Returns:
        str: Path lengkap file tujuan.
    """
    filename = f"{prefix}_{run_timestamp}.{extension}"
    return os.path.join(RAW_DIR, subfolder, filename)


def describe_request_error(error):
    """Ringkas error request tanpa membocorkan URL (berisi API key).

    Args:
        error (requests.exceptions.RequestException): Error request.

    Returns:
        str: Deskripsi singkat berisi nama error dan status HTTP.
    """
    status_code = getattr(error.response, "status_code", None)
    return f"{type(error).__name__} (status={status_code})"


def request_open_meteo(url, hourly_variables):
    """Panggil endpoint Open-Meteo dan kembalikan respons JSON.

    Args:
        url (str): URL endpoint Open-Meteo (weather atau air quality).
        hourly_variables (tuple): Nama variabel per jam yang diminta.

    Returns:
        dict: Respons JSON hasil parsing.

    Raises:
        requests.exceptions.RequestException: Jika request gagal.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(hourly_variables),
        "past_days": PAST_DAYS,
        "timezone": TIMEZONE_NAME,
    }
    response = requests.get(
        url, params=params, timeout=REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    return response.json()


def save_json(data, filepath):
    """Simpan dictionary ke file JSON.

    Args:
        data (dict): Data yang akan disimpan.
        filepath (str): Lokasi file tujuan.
    """
    with open(filepath, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)


def count_csv_rows(csv_text):
    """Hitung jumlah baris data (di luar header) pada teks CSV.

    Args:
        csv_text (str): Isi CSV lengkap termasuk header.

    Returns:
        int: Jumlah baris data.
    """
    return sum(1 for _ in csv.DictReader(io.StringIO(csv_text)))


def ingest_weather(run_timestamp):
    """Ambil data cuaca Open-Meteo dan simpan sebagai JSON mentah.

    Args:
        run_timestamp (str): Timestamp run untuk nama file.

    Returns:
        str: Path file yang disimpan.
    """
    data = request_open_meteo(WEATHER_URL, WEATHER_VARIABLES)
    filepath = build_raw_filepath(
        "weather", "weather", run_timestamp, "json"
    )
    save_json(data, filepath)
    row_count = len(data["hourly"]["time"])
    print(f"[OK] Weather: {row_count} baris -> {filepath}")
    return filepath


def ingest_air_quality(run_timestamp):
    """Ambil data kualitas udara Open-Meteo dan simpan sebagai JSON.

    Args:
        run_timestamp (str): Timestamp run untuk nama file.

    Returns:
        str: Path file yang disimpan.
    """
    data = request_open_meteo(AIR_QUALITY_URL, AIR_QUALITY_VARIABLES)
    filepath = build_raw_filepath(
        "air_quality", "air_quality", run_timestamp, "json"
    )
    save_json(data, filepath)
    row_count = len(data["hourly"]["time"])
    print(f"[OK] Air quality: {row_count} baris -> {filepath}")
    return filepath


def build_firms_url(date_str):
    """Susun URL endpoint area FIRMS untuk satu jendela tanggal.

    Args:
        date_str (str): Tanggal awal dalam format YYYY-MM-DD.

    Returns:
        str: URL lengkap endpoint area FIRMS.
    """
    return "/".join(
        [
            FIRMS_BASE_URL,
            FIRMS_MAP_KEY,
            FIRMS_SOURCE,
            FIRMS_BOUNDING_BOX,
            str(FIRMS_DAY_RANGE),
            date_str,
        ]
    )


def fetch_firms_window(date_str):
    """Ambil satu jendela FIRMS dan kembalikan teks CSV-nya.

    Args:
        date_str (str): Tanggal awal jendela (YYYY-MM-DD).

    Returns:
        str: Isi CSV dari endpoint FIRMS.
    """
    url = build_firms_url(date_str)
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def ingest_firms(run_timestamp):
    """Ambil data titik api NASA FIRMS dan simpan sebagai CSV mentah.

    Endpoint area FIRMS dibatasi day_range 1-10 hari per panggilan.
    Untuk menutup FIRMS_TARGET_DAYS hari ke belakang, fungsi ini
    melakukan beberapa panggilan berurutan dengan tanggal awal yang
    bergeser, lalu menggabungkan hasilnya jadi satu file CSV.

    Args:
        run_timestamp (str): Timestamp run untuk nama file.

    Returns:
        str: Path file yang disimpan, atau None jika dilewati/kosong.
    """
    if not FIRMS_MAP_KEY:
        print("[SKIP] FIRMS_MAP_KEY belum diset, FIRMS dilewati.")
        return None

    today = datetime.now(timezone.utc).date()
    n_windows = FIRMS_TARGET_DAYS // FIRMS_DAY_RANGE
    collected_rows = []

    for i in range(n_windows):
        # Yang lebih lama dulu, lalu bergeser ke depan.
        start_date = today - timedelta(
            days=FIRMS_TARGET_DAYS - i * FIRMS_DAY_RANGE
        )
        date_str = start_date.isoformat()

        csv_text = fetch_firms_window(date_str)
        rows = list(csv.DictReader(io.StringIO(csv_text)))

        if rows:
            print(
                f"  [OK] FIRMS {date_str} "
                f"(+{FIRMS_DAY_RANGE}h): {len(rows)} titik api."
            )
            collected_rows.extend(rows)
        else:
            print(f"  [INFO] FIRMS {date_str}: kosong.")

    if not collected_rows:
        print(
            "[INFO] FIRMS: tidak ada titik api di seluruh jendela. "
            "Ini normal di luar musim kemarau."
        )
        return None

    filepath = build_raw_filepath("firms", "firms", run_timestamp, "csv")
    with open(filepath, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=collected_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(collected_rows)

    print(f"[OK] FIRMS total: {len(collected_rows)} titik api -> {filepath}")
    return filepath


def run_ingestion_step(source_name, ingest_function, run_timestamp):
    """Jalankan satu langkah ingestion dengan penanganan error.

    Kegagalan satu sumber tidak menghentikan sumber lainnya.

    Args:
        source_name (str): Nama sumber data untuk pesan log.
        ingest_function (callable): Fungsi ingestion sumber tersebut.
        run_timestamp (str): Timestamp run untuk nama file.

    Returns:
        str: Path file hasil ingestion, atau None jika gagal/dilewati.
    """
    try:
        return ingest_function(run_timestamp)
    except requests.exceptions.RequestException as error:
        print(f"[ERROR] {source_name}: {describe_request_error(error)}")
        return None


def main():
    """Jalankan ingestion untuk ketiga sumber dan cetak ringkasannya."""
    create_raw_directories()
    run_timestamp = build_run_timestamp()

    print("=" * 60)
    print(f"DATA INGESTION - run timestamp: {run_timestamp}")
    print("=" * 60)

    steps = {
        "weather": ingest_weather,
        "air_quality": ingest_air_quality,
        "firms": ingest_firms,
    }
    results = {}
    for source_name, ingest_function in steps.items():
        results[source_name] = run_ingestion_step(
            source_name, ingest_function, run_timestamp
        )

    print("=" * 60)
    print("RINGKASAN INGESTION")
    print("=" * 60)
    for source_name, filepath in results.items():
        status = "BERHASIL" if filepath else "GAGAL/DILEWATI"
        print(f"  {source_name:<12}: {status}")


if __name__ == "__main__":
    main()