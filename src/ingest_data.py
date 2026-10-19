"""Data ingestion berkala untuk proyek prediksi PM2.5 akibat karhutla.

Dijalankan otomatis oleh GitHub Actions (cron tiap jam pada menit
ke-5). Mengambil data dari tiga sumber API publik dan menyimpannya
sebagai data mentah di data/raw/:

    1. Open-Meteo Weather API -> data/raw/weather/*.json
    2. Open-Meteo Air Quality -> data/raw/air_quality/*.json
    3. NASA FIRMS (titik api) -> data/raw/firms/*.csv

Mode:
    incremental (default): ambil data terbaru saja. Jendelanya sengaja
        tumpang tindih dengan run sebelumnya sebagai pengaman jika ada
        run yang terlewat; duplikat dibuang di preprocess.py.
    backfill: dijalankan sekali untuk mengisi riwayat awal.

Setiap file diberi timestamp UTC sehingga run berulang tidak pernah
menimpa data lama. Kegagalan satu sumber tidak menghentikan sumber
lain dan dicatat ke logs/ingestion_YYYYMMDD.log. Jika ada sumber yang
gagal, exit code program adalah 1.

Cara run manual (uji coba):
    python src/ingest_data.py
    python src/ingest_data.py --mode backfill --backfill-days 30

Environment variable:
    FIRMS_MAP_KEY : API key NASA FIRMS. Di GitHub Actions diambil dari
        Secrets, di lokal dari file .env.
"""

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

LATITUDE = -2.21
LONGITUDE = 113.92
TIMEZONE_NAME = "Asia/Jakarta"

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_SOURCE = "VIIRS_SNPP_NRT"
# Format: min_lon,min_lat,max_lon,max_lat (Kalimantan Tengah).
FIRMS_BOUNDING_BOX = "110.5,-3.5,115.5,1.0"
# Batas aman yang sudah terbukti diterima API untuk key ini.
FIRMS_MAX_DAY_RANGE = 5

WEATHER_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "cloud_cover",
)
AIR_QUALITY_VARIABLES = ("pm2_5", "pm10", "european_aqi")

# Mode incremental: 2 hari menutup run yang terlewat atau tertunda.
INCREMENTAL_PAST_DAYS = 2
INCREMENTAL_FIRMS_DAYS = 2
DEFAULT_BACKFILL_DAYS = 30
MAX_BACKFILL_DAYS = 92  # batas past_days Open-Meteo

RAW_DIR = os.path.join("data", "raw")
RAW_SUBFOLDERS = ("weather", "air_quality", "firms")
LOG_DIR = "logs"

REQUEST_TIMEOUT_SECONDS = 30
RETRY_DELAYS_SECONDS = (5, 15, 45)
SERVER_ERROR_MIN_STATUS = 500

logger = logging.getLogger("ingest")


class MissingCredentialError(Exception):
    """Dilempar saat API key yang dibutuhkan belum diset."""


def parse_arguments(argv=None):
    """Baca argumen command line.

    Args:
        argv (list): Daftar argumen; None berarti memakai sys.argv.

    Returns:
        argparse.Namespace: Berisi mode dan backfill_days.
    """
    parser = argparse.ArgumentParser(description="Data ingestion berkala.")
    parser.add_argument(
        "--mode",
        choices=("incremental", "backfill"),
        default="incremental",
        help="incremental: data terbaru; backfill: isi riwayat awal.",
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=DEFAULT_BACKFILL_DAYS,
        help="Jumlah hari ke belakang untuk mode backfill.",
    )
    return parser.parse_args(argv)


def configure_logging(run_date):
    """Siapkan logging ke konsol dan ke logs/ingestion_YYYYMMDD.log.

    Args:
        run_date (str): Tanggal UTC format YYYYMMDD.

    Returns:
        str: Path file log.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"ingestion_{run_date}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


def create_raw_directories():
    """Buat folder data/raw/<sumber> jika belum ada."""
    for subfolder in RAW_SUBFOLDERS:
        os.makedirs(os.path.join(RAW_DIR, subfolder), exist_ok=True)


def describe_error(error):
    """Ringkas error tanpa membocorkan URL (berisi API key).

    Args:
        error (Exception): Error yang terjadi.

    Returns:
        str: Deskripsi singkat berisi nama error dan status HTTP.
    """
    if isinstance(error, requests.exceptions.RequestException):
        status_code = getattr(error.response, "status_code", None)
        return f"{type(error).__name__} (status={status_code})"
    return str(error)


def is_retryable(error):
    """Tentukan apakah error layak dicoba ulang.

    Hanya timeout, gangguan koneksi, dan status 5xx yang dicoba ulang.
    Error 4xx (key salah, parameter salah) tidak akan membaik.

    Args:
        error (requests.exceptions.RequestException): Error request.

    Returns:
        bool: True jika sebaiknya dicoba ulang.
    """
    if isinstance(
        error,
        (requests.exceptions.Timeout, requests.exceptions.ConnectionError),
    ):
        return True
    response = getattr(error, "response", None)
    return (
        response is not None
        and response.status_code >= SERVER_ERROR_MIN_STATUS
    )


def get_with_retry(url, params=None, sleep=time.sleep):
    """GET dengan retry exponential backoff (5s, 15s, 45s).

    Args:
        url (str): URL tujuan.
        params (dict): Query parameter opsional.
        sleep (callable): Fungsi jeda; dapat diganti saat pengujian.

    Returns:
        requests.Response: Respons sukses.

    Raises:
        requests.exceptions.RequestException: Jika semua percobaan
            gagal atau error tidak layak dicoba ulang.
    """
    max_retries = len(RETRY_DELAYS_SECONDS)
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(
                url, params=params, timeout=REQUEST_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as error:
            if attempt == max_retries or not is_retryable(error):
                raise
            delay = RETRY_DELAYS_SECONDS[attempt]
            logger.warning(
                "Retry %d/%d dalam %ds: %s",
                attempt + 1,
                max_retries,
                delay,
                describe_error(error),
            )
            sleep(delay)


def clamp_backfill_days(backfill_days):
    """Batasi jumlah hari backfill ke rentang 1 sampai 92.

    Args:
        backfill_days (int): Nilai yang diminta pengguna.

    Returns:
        int: Nilai yang sudah dibatasi.
    """
    return max(1, min(backfill_days, MAX_BACKFILL_DAYS))


def determine_past_days(mode, backfill_days):
    """Tentukan nilai past_days Open-Meteo sesuai mode.

    Args:
        mode (str): incremental atau backfill.
        backfill_days (int): Jumlah hari untuk mode backfill.

    Returns:
        int: Nilai parameter past_days.
    """
    if mode == "backfill":
        return backfill_days
    return INCREMENTAL_PAST_DAYS


def build_raw_filepath(subfolder, run_timestamp, extension):
    """Susun path file mentah berdasarkan sumber dan timestamp.

    Args:
        subfolder (str): Nama subfolder sumber di dalam data/raw/.
        run_timestamp (str): Timestamp run ingestion.
        extension (str): Ekstensi file tanpa titik.

    Returns:
        str: Path lengkap file tujuan.
    """
    filename = f"{subfolder}_{run_timestamp}.{extension}"
    return os.path.join(RAW_DIR, subfolder, filename)


def request_open_meteo(url, hourly_variables, past_days):
    """Panggil endpoint Open-Meteo dan kembalikan respons JSON.

    Args:
        url (str): URL endpoint Open-Meteo.
        hourly_variables (tuple): Nama variabel per jam yang diminta.
        past_days (int): Jumlah hari ke belakang.

    Returns:
        dict: Respons JSON hasil parsing.
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(hourly_variables),
        "past_days": past_days,
        "timezone": TIMEZONE_NAME,
    }
    return get_with_retry(url, params=params).json()


def save_json(data, filepath):
    """Simpan dictionary ke file JSON.

    Args:
        data (dict): Data yang akan disimpan.
        filepath (str): Lokasi file tujuan.
    """
    with open(filepath, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)


def ingest_weather(run_timestamp, mode, backfill_days):
    """Ambil data cuaca Open-Meteo dan simpan sebagai JSON mentah.

    Args:
        run_timestamp (str): Timestamp run untuk nama file.
        mode (str): incremental atau backfill.
        backfill_days (int): Jumlah hari untuk mode backfill.

    Returns:
        str: Path file yang disimpan.
    """
    past_days = determine_past_days(mode, backfill_days)
    data = request_open_meteo(WEATHER_URL, WEATHER_VARIABLES, past_days)
    filepath = build_raw_filepath("weather", run_timestamp, "json")
    save_json(data, filepath)
    row_count = len(data["hourly"]["time"])
    logger.info("Weather: %d baris -> %s", row_count, filepath)
    return filepath


def ingest_air_quality(run_timestamp, mode, backfill_days):
    """Ambil data kualitas udara Open-Meteo dan simpan sebagai JSON.

    Args:
        run_timestamp (str): Timestamp run untuk nama file.
        mode (str): incremental atau backfill.
        backfill_days (int): Jumlah hari untuk mode backfill.

    Returns:
        str: Path file yang disimpan.
    """
    past_days = determine_past_days(mode, backfill_days)
    data = request_open_meteo(
        AIR_QUALITY_URL, AIR_QUALITY_VARIABLES, past_days
    )
    filepath = build_raw_filepath("air_quality", run_timestamp, "json")
    save_json(data, filepath)
    row_count = len(data["hourly"]["time"])
    logger.info("Air quality: %d baris -> %s", row_count, filepath)
    return filepath


def build_firms_plan(mode, backfill_days, today):
    """Susun daftar request FIRMS (day_range, tanggal mulai).

    Mode incremental cukup satu request tanpa tanggal. Mode backfill
    dipecah per 5 hari karena batas day_range per request.

    Args:
        mode (str): incremental atau backfill.
        backfill_days (int): Jumlah hari untuk mode backfill.
        today (datetime.date): Tanggal hari ini (UTC).

    Returns:
        list: Pasangan (day_range, start_date); start_date bisa None.
    """
    if mode == "incremental":
        return [(INCREMENTAL_FIRMS_DAYS, None)]
    plan = []
    current = today - timedelta(days=backfill_days - 1)
    while current <= today:
        days = min(FIRMS_MAX_DAY_RANGE, (today - current).days + 1)
        plan.append((days, current))
        current += timedelta(days=FIRMS_MAX_DAY_RANGE)
    return plan


def build_firms_url(map_key, day_range, start_date):
    """Susun URL FIRMS area endpoint.

    Args:
        map_key (str): API key FIRMS.
        day_range (int): Jumlah hari per request.
        start_date (datetime.date): Tanggal mulai, atau None untuk
            data terbaru.

    Returns:
        str: URL lengkap.
    """
    parts = [
        FIRMS_BASE_URL,
        map_key,
        FIRMS_SOURCE,
        FIRMS_BOUNDING_BOX,
        str(day_range),
    ]
    if start_date is not None:
        parts.append(start_date.isoformat())
    return "/".join(parts)


def merge_csv_texts(csv_texts):
    """Gabungkan beberapa teks CSV dengan satu header saja.

    Args:
        csv_texts (list): Isi CSV lengkap beserta header.

    Returns:
        str: CSV gabungan, atau string kosong jika tidak ada isi.
    """
    merged_lines = []
    for text in csv_texts:
        lines = text.strip().splitlines()
        if not lines:
            continue
        merged_lines.extend(lines if not merged_lines else lines[1:])
    return "\n".join(merged_lines) + "\n" if merged_lines else ""


def ingest_firms(run_timestamp, mode, backfill_days):
    """Ambil data titik api NASA FIRMS dan simpan sebagai CSV mentah.

    Args:
        run_timestamp (str): Timestamp run untuk nama file.
        mode (str): incremental atau backfill.
        backfill_days (int): Jumlah hari untuk mode backfill.

    Returns:
        str: Path file yang disimpan.

    Raises:
        MissingCredentialError: Jika FIRMS_MAP_KEY belum diset.
    """
    map_key = os.getenv("FIRMS_MAP_KEY", "")
    if not map_key:
        raise MissingCredentialError("FIRMS_MAP_KEY belum diset")

    today = datetime.now(timezone.utc).date()
    csv_texts = []
    for day_range, start_date in build_firms_plan(mode, backfill_days, today):
        url = build_firms_url(map_key, day_range, start_date)
        csv_texts.append(get_with_retry(url).text)

    merged_text = merge_csv_texts(csv_texts)
    filepath = build_raw_filepath("firms", run_timestamp, "csv")
    with open(filepath, "w", encoding="utf-8") as output_file:
        output_file.write(merged_text)

    hotspot_count = sum(1 for _ in csv.DictReader(io.StringIO(merged_text)))
    logger.info("FIRMS: %d titik api -> %s", hotspot_count, filepath)
    return filepath


def run_ingestion_step(
    source_name, ingest_function, run_timestamp, mode, backfill_days
):
    """Jalankan satu langkah ingestion tanpa menghentikan yang lain.

    Args:
        source_name (str): Nama sumber untuk pesan log.
        ingest_function (callable): Fungsi ingestion sumber tersebut.
        run_timestamp (str): Timestamp run untuk nama file.
        mode (str): incremental atau backfill.
        backfill_days (int): Jumlah hari untuk mode backfill.

    Returns:
        str: Path file hasil ingestion, atau None jika gagal.
    """
    try:
        return ingest_function(run_timestamp, mode, backfill_days)
    except (
        requests.exceptions.RequestException,
        MissingCredentialError,
    ) as error:
        logger.error("%s GAGAL: %s", source_name, describe_error(error))
        return None


def main(argv=None):
    """Jalankan ingestion untuk ketiga sumber.

    Args:
        argv (list): Argumen command line; None memakai sys.argv.

    Returns:
        int: 0 jika semua sumber berhasil, 1 jika ada yang gagal.
    """
    args = parse_arguments(argv)
    backfill_days = clamp_backfill_days(args.backfill_days)
    now_utc = datetime.now(timezone.utc)
    configure_logging(now_utc.strftime("%Y%m%d"))
    create_raw_directories()
    run_timestamp = now_utc.strftime("%Y%m%d_%H%M%S")

    logger.info("INGESTION mode=%s run=%s", args.mode, run_timestamp)
    steps = (
        ("weather", ingest_weather),
        ("air_quality", ingest_air_quality),
        ("firms", ingest_firms),
    )
    results = {}
    for source_name, ingest_function in steps:
        results[source_name] = run_ingestion_step(
            source_name,
            ingest_function,
            run_timestamp,
            args.mode,
            backfill_days,
        )

    for source_name, filepath in results.items():
        status = "BERHASIL" if filepath else "GAGAL"
        logger.info("  %-12s: %s", source_name, status)

    failed = [name for name, path in results.items() if path is None]
    if failed:
        logger.error("Sumber gagal: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
