"""Prapemrosesan (cleaning) data mentah hasil ingest_data.py.

Membaca SEMUA file mentah di data/raw/, menggabungkannya, lalu
membersihkan data agar siap masuk ke tahap ekstraksi fitur.

Tahapan:
    1. Muat dan gabungkan seluruh file mentah per sumber. Run yang
       dilakukan berulang terakumulasi. Jika ada baris duplikat,
       baris dari file terbaru yang dipertahankan.
    2. Gabungkan weather dan air quality berdasarkan kolom waktu.
    3. Buang baris yang memiliki nilai kosong (null).
    4. Validasi rentang nilai (mis. kelembapan harus 0-100).
    5. Filter tingkat kepercayaan (confidence) titik api FIRMS.
    6. Konversi waktu FIRMS dari UTC ke Asia/Jakarta (UTC+7).
    7. Cek cakupan histori FIRMS terhadap weather/AQI.
    8. Simpan hasil ke data/interim/.

Cara run:
    python src/preprocess.py
"""

import glob
import json
import os

import pandas as pd

RAW_DIR = os.path.join("data", "raw")
INTERIM_DIR = os.path.join("data", "interim")

MIN_ROWS_TARGET = 500

# Ambang minimal cakupan FIRMS (hari). Di bawah ini, model berisiko
# belajar "nol titik api" sebagai fakta padahal itu hanya "tidak ada
# data".
MIN_FIRMS_COVERAGE_DAYS = 7

PERCENT_COLUMNS = ("relative_humidity_2m", "cloud_cover")
NON_NEGATIVE_COLUMNS = ("pm2_5", "pm10", "wind_speed_10m", "precipitation")

FIRMS_DEDUP_COLUMNS = ["latitude", "longitude", "acq_date", "acq_time"]
# FIRMS memakai huruf untuk VIIRS (l/n/h) dan angka 0-100 untuk MODIS.
CONFIDENCE_LEVELS = {
    "l": 0,
    "low": 0,
    "n": 1,
    "nominal": 1,
    "h": 2,
    "high": 2,
}
NUMERIC_CONFIDENCE_THRESHOLD = 70

# Zona waktu yang dipakai weather & AQI. FIRMS (acq_time) memakai UTC.
JAKARTA_TZ = "Asia/Jakarta"

def report_step(step_name, rows_before, rows_after):
    """Cetak ringkasan jumlah baris sebelum dan sesudah satu langkah.

    Args:
        step_name (str): Nama langkah cleaning.
        rows_before (int): Jumlah baris sebelum langkah dijalankan.
        rows_after (int): Jumlah baris sesudah langkah dijalankan.
    """
    removed = rows_before - rows_after
    print(
        f"[CLEAN] {step_name}: {removed} baris dibuang "
        f"({rows_before} -> {rows_after})"
    )


def list_raw_files(subfolder, extension):
    """Daftar file mentah satu sumber, urut dari yang terlama.

    Nama file memuat timestamp (YYYYMMDD_HHMMSS), sehingga urutan
    alfabet sama dengan urutan kronologis.

    Args:
        subfolder (str): Nama subfolder sumber di dalam data/raw/.
        extension (str): Ekstensi file tanpa titik.

    Returns:
        list: Path file yang ditemukan, terurut dari terlama.
    """
    pattern = os.path.join(RAW_DIR, subfolder, f"*.{extension}")
    return sorted(glob.glob(pattern))


def load_hourly_json_files(filepaths):
    """Muat file JSON Open-Meteo dan gabungkan menjadi satu DataFrame.

    Baris dengan waktu yang sama di beberapa file (karena rentang
    data saling tumpang tindih) di-deduplikasi dengan mempertahankan
    baris dari file paling baru.

    Args:
        filepaths (list): Path file JSON, terurut dari terlama.

    Returns:
        pd.DataFrame: Data per jam dengan kolom time bertipe datetime.
    """
    frames = []
    for filepath in filepaths:
        with open(filepath, encoding="utf-8") as input_file:
            payload = json.load(input_file)
        frames.append(pd.DataFrame(payload["hourly"]))

    combined_df = pd.concat(frames, ignore_index=True)
    combined_df["time"] = pd.to_datetime(combined_df["time"])
    return combined_df.drop_duplicates(subset="time", keep="last")


def load_firms_files(filepaths):
    """Muat file CSV FIRMS dan gabungkan menjadi satu DataFrame.

    Semua kolom dibaca sebagai teks agar nilai seperti acq_time
    (mis. 0453) tidak kehilangan angka nol di depan.

    Args:
        filepaths (list): Path file CSV, terurut dari terlama.

    Returns:
        pd.DataFrame: Data titik api tanpa baris duplikat. Kosong jika
            semua file hanya berisi header.
    """
    frames = []
    for filepath in filepaths:
        try:
            frames.append(pd.read_csv(filepath, dtype=str))
        except pd.errors.EmptyDataError:
            continue

    if not frames:
        return pd.DataFrame()
    combined_df = pd.concat(frames, ignore_index=True)
    return combined_df.drop_duplicates(
        subset=FIRMS_DEDUP_COLUMNS, keep="last"
    )


def add_firms_jakarta_time(hotspots_df):
    """Tambahkan kolom waktu FIRMS dalam zona Asia/Jakarta.

    Kolom acq_date (YYYY-MM-DD) dan acq_time (HHMM) dari FIRMS
    berformat UTC, sedangkan weather & AQI memakai Asia/Jakarta
    (UTC+7). Tanpa konversi ini, hotspot_count_100km pada tahap
    feature engineering akan bergeser 7 jam dan leading indicator
    kehilangan maknanya.

    Args:
        hotspots_df (pd.DataFrame): Data FIRMS dengan kolom acq_date
            dan acq_time.

    Returns:
        pd.DataFrame: Data dengan kolom tambahan:
            - acq_datetime_utc : timestamp tz-aware UTC
            - acq_datetime_jkt : timestamp tz-aware Asia/Jakarta
            - acq_hour_jkt     : acq_datetime_jkt di-floor ke jam
                                 (siap dipakai untuk agregasi per jam)
    """
    if hotspots_df.empty:
        return hotspots_df

    df = hotspots_df.copy()

    # acq_time bisa kehilangan leading zero saat dibaca sebagai angka
    # (mis. "453" bukan "0453"). Pad ke 4 digit.
    acq_time_padded = df["acq_time"].astype(str).str.strip().str.zfill(4)

    # Normalisasi acq_date: terima "YYYY-MM-DD" maupun "YYYYMMDD".
    acq_date_str = df["acq_date"].astype(str).str.strip()
    acq_date_str = acq_date_str.str.replace(
        r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True
    )

    utc_dt = pd.to_datetime(
        acq_date_str + " " + acq_time_padded,
        format="%Y-%m-%d %H%M",
        utc=True,
        errors="coerce",
    )

    df["acq_datetime_utc"] = utc_dt
    df["acq_datetime_jkt"] = utc_dt.dt.tz_convert(JAKARTA_TZ)
    df["acq_hour_jkt"] = df["acq_datetime_jkt"].dt.floor("h")

    n_invalid = int(utc_dt.isna().sum())
    if n_invalid > 0:
        print(
            f"[WARNING] {n_invalid} baris FIRMS gagal diparse waktunya "
            "dan akan dibuang."
        )
        df = df[utc_dt.notna()].reset_index(drop=True)

    print(
        f"[INFO] Konversi waktu FIRMS UTC -> {JAKARTA_TZ} selesai "
        f"({len(df)} baris)."
    )
    return df


def drop_missing_values(weather_air_df):
    """Buang baris yang memiliki nilai kosong pada kolom manapun.

    Nilai kosong umumnya muncul di ujung horizon prakiraan yang belum
    tersedia. Nilai target pm2_5 tidak boleh diimputasi, sehingga
    baris tersebut dibuang.

    Args:
        weather_air_df (pd.DataFrame): Data gabungan weather + AQI.

    Returns:
        pd.DataFrame: Data tanpa nilai kosong.
    """
    cleaned_df = weather_air_df.dropna()
    report_step(
        "Buang nilai kosong", len(weather_air_df), len(cleaned_df)
    )
    return cleaned_df


def remove_out_of_range_values(weather_air_df):
    """Buang baris dengan nilai di luar rentang yang wajar.

    Kelembapan dan tutupan awan harus 0-100, sedangkan polutan,
    kecepatan angin, dan curah hujan tidak boleh negatif.

    Args:
        weather_air_df (pd.DataFrame): Data gabungan weather + AQI.

    Returns:
        pd.DataFrame: Data yang seluruh nilainya berada dalam rentang.
    """
    is_valid = pd.Series(True, index=weather_air_df.index)
    for column in PERCENT_COLUMNS:
        if column in weather_air_df.columns:
            is_valid &= weather_air_df[column].between(0, 100)
    for column in NON_NEGATIVE_COLUMNS:
        if column in weather_air_df.columns:
            is_valid &= weather_air_df[column] >= 0

    cleaned_df = weather_air_df[is_valid]
    report_step(
        "Validasi rentang nilai", len(weather_air_df), len(cleaned_df)
    )
    return cleaned_df


def is_confident_detection(confidence_value, min_level="n"):
    """Cek apakah satu deteksi titik api memenuhi tingkat kepercayaan.

    Format huruf (VIIRS) dibandingkan dengan min_level, sedangkan
    format angka (MODIS) dianggap valid jika lebih dari 70.

    Args:
        confidence_value (str): Nilai kolom confidence dari FIRMS.
        min_level (str): Level minimum untuk format huruf.

    Returns:
        bool: True jika deteksi lolos filter.
    """
    normalized = str(confidence_value).strip().lower()
    try:
        return float(normalized) > NUMERIC_CONFIDENCE_THRESHOLD
    except ValueError:
        minimum_level = CONFIDENCE_LEVELS.get(min_level, 1)
        return CONFIDENCE_LEVELS.get(normalized, -1) >= minimum_level


def filter_firms_confidence(hotspots_df, min_level="n"):
    """Pertahankan hanya deteksi titik api berkepercayaan cukup.

    Args:
        hotspots_df (pd.DataFrame): Data titik api mentah.
        min_level (str): Level minimum untuk format huruf.

    Returns:
        pd.DataFrame: Data titik api yang lolos filter.
    """
    is_confident = hotspots_df["confidence"].apply(
        is_confident_detection, min_level=min_level
    )
    filtered_df = hotspots_df[is_confident]
    report_step(
        "Filter confidence FIRMS", len(hotspots_df), len(filtered_df)
    )
    return filtered_df


def check_volume_target(row_count):
    """Cetak apakah jumlah baris memenuhi target volume minimal.

    Args:
        row_count (int): Jumlah baris data bersih.
    """
    if row_count >= MIN_ROWS_TARGET:
        print(f"[OK] Volume {row_count} baris (target > {MIN_ROWS_TARGET}).")
    else:
        print(
            f"[WARNING] Volume {row_count} baris masih di bawah target "
            f"{MIN_ROWS_TARGET}. Naikkan PAST_DAYS di ingest_data.py "
            "atau jalankan ingestion lagi."
        )


def check_firms_coverage(hotspots_df, weather_air_df):
    """Cek cakupan waktu FIRMS terhadap weather/AQI.

    Ingestion yang hanya menarik 1 hari FIRMS per run menyebabkan
    jam-jam sebelum run pertama tampak 'nol titik api', padahal
    sebenarnya tidak diketahui. Fungsi ini mendeteksi gap tersebut
    dan mencetak peringatan actionable.

    Args:
        hotspots_df (pd.DataFrame): Data FIRMS bersih dengan kolom
            acq_datetime_jkt. Boleh None atau kosong.
        weather_air_df (pd.DataFrame): Data weather + AQI bersih
            dengan kolom time (naive, dianggap Asia/Jakarta).
    """
    if weather_air_df is None or weather_air_df.empty:
        return

    weather_min = weather_air_df["time"].min()
    weather_max = weather_air_df["time"].max()
    expected_days = (weather_max - weather_min).total_seconds() / 86400

    print(
        f"[INFO] Cakupan weather/AQI : {weather_min} s/d {weather_max} "
        f"({expected_days:.1f} hari)"
    )

    if hotspots_df is None or hotspots_df.empty:
        print(
            "[WARNING] FIRMS kosong. Semua jam akan tampak 'nol titik api'. "
            "Tarik histori FIRMS sebelum training."
        )
        return

    firms_min = hotspots_df["acq_datetime_jkt"].min().tz_localize(None)
    firms_max = hotspots_df["acq_datetime_jkt"].max().tz_localize(None)
    coverage_days = (firms_max - firms_min).total_seconds() / 86400

    print(
        f"[INFO] Cakupan FIRMS       : {firms_min} s/d {firms_max} "
        f"({coverage_days:.1f} hari)"
    )

    if firms_min > weather_min:
        gap_hours = (firms_min - weather_min).total_seconds() / 3600
        print(
            f"[WARNING] FIRMS mulai {gap_hours:.0f} jam setelah weather/AQI. "
            "Jam-jam awal akan tampak 'nol titik api' padahal tidak "
            "diketahui. Tarik histori FIRMS dengan parameter tanggal "
            "di endpoint area (day range resmi 1-10 hari per panggilan)."
        )

    if coverage_days < MIN_FIRMS_COVERAGE_DAYS:
        print(
            f"[WARNING] Cakupan FIRMS hanya {coverage_days:.1f} hari "
            f"(< {MIN_FIRMS_COVERAGE_DAYS} hari). Model berisiko belajar "
            "'nol titik api' sebagai fakta. Perpanjang histori ingestion."
        )
    else:
        print(
            f"[OK] Cakupan FIRMS {coverage_days:.1f} hari, cukup untuk "
            "leading indicator."
        )


def process_weather_air_quality():
    """Bersihkan data weather + air quality dan simpan ke interim.

    Returns:
        pd.DataFrame: Data bersih, atau None jika data mentah belum ada.
    """
    weather_files = list_raw_files("weather", "json")
    air_quality_files = list_raw_files("air_quality", "json")
    if not weather_files or not air_quality_files:
        print(
            "[ERROR] Data mentah weather/air_quality tidak ditemukan. "
            "Jalankan ingest_data.py terlebih dahulu."
        )
        return None

    print(f"Weather      : {len(weather_files)} file mentah")
    print(f"Air quality  : {len(air_quality_files)} file mentah")

    weather_df = load_hourly_json_files(weather_files)
    air_quality_df = load_hourly_json_files(air_quality_files)
    merged_df = pd.merge(weather_df, air_quality_df, on="time", how="inner")
    merged_df = merged_df.sort_values("time").reset_index(drop=True)
    print(f"[INFO] Hasil gabungan weather + AQI: {len(merged_df)} baris")

    cleaned_df = drop_missing_values(merged_df)
    cleaned_df = remove_out_of_range_values(cleaned_df)

    output_path = os.path.join(INTERIM_DIR, "weather_aqi_clean.csv")
    cleaned_df.to_csv(output_path, index=False)
    print(f"[OK] Data bersih: {len(cleaned_df)} baris -> {output_path}")
    check_volume_target(len(cleaned_df))
    return cleaned_df


def process_firms():
    """Bersihkan data titik api FIRMS dan simpan ke interim.

    Urutan: gabung file -> dedup -> filter confidence -> konversi
    waktu UTC ke Asia/Jakarta.

    Returns:
        pd.DataFrame: Data bersih, atau None jika tidak ada data.
    """
    firms_files = list_raw_files("firms", "csv")
    if not firms_files:
        print("[INFO] Tidak ada file FIRMS mentah, langkah dilewati.")
        return None

    print(f"FIRMS        : {len(firms_files)} file mentah")

    hotspots_df = load_firms_files(firms_files)
    if hotspots_df.empty:
        print(
            "[INFO] Tidak ada titik api pada data FIRMS. Ini kondisi "
            "normal di luar musim kemarau."
        )
        return None

    cleaned_df = filter_firms_confidence(hotspots_df)

    if cleaned_df.empty:
        print(
            "[WARNING] Semua deteksi FIRMS tersaring oleh filter "
            "confidence. Tidak ada hotspot untuk diproses."
        )
        return None

    cleaned_df = add_firms_jakarta_time(cleaned_df)

    output_path = os.path.join(INTERIM_DIR, "firms_clean.csv")
    cleaned_df.to_csv(output_path, index=False)
    print(f"[OK] Data FIRMS bersih: {len(cleaned_df)} baris -> {output_path}")
    return cleaned_df


def main():
    """Jalankan seluruh tahap prapemrosesan."""
    os.makedirs(INTERIM_DIR, exist_ok=True)

    print("=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    weather_air_df = process_weather_air_quality()
    firms_df = process_firms()

    print("-" * 60)
    check_firms_coverage(firms_df, weather_air_df)

    print("=" * 60)
    print("SELESAI - Data siap untuk tahap feature engineering")
    print("=" * 60)


if __name__ == "__main__":
    main()