# MLOps-PM25-Prediction
"PREDIKSI LONJAKAN PM2.5 AKIBAT KARHUTLA DENGAN LEADING INDICATOR TITIK API SATELIT"

Sistem prediksi lonjakan konsentrasi PM2.5 akibat kebakaran hutan dan lahan (karhutla), menggunakan pendekatan leading indicator dari data titik api satelit (NASA FIRMS) yang dikombinasikan dengan data cuaca dan kualitas udara real-time.

## Tujuan Proyek
Proyek ini bertujuan membangun sistem end-to-end MLOps yang mampu:
1. Memprediksi konsentrasi PM2.5 6 jam ke depan (H+6) di wilayah rawan karhutla dengan fokus awal Kalimantan Tengah
2. Memanfaatkan data titik api satelit sebagai leading indicator, sinyal peringatan dini sebelum lonjakan PM2.5 benar-benar terukur di sensor darat
3. Menerapkan siklus continual learning penuh: data terus diperbarui secara otomatis, model di-retrain berdasarkan jadwal maupun trigger performa/drift, dan divalidasi sebelum dipromosikan ke production
Berbeda dari pendekatan prediksi AQI konvensional yang hanya mengandalkan data cuaca dan historis polutan, proyek ini menambahkan data titik api satelit (NASA FIRMS) sebagai fitur tambahan untuk meningkatkan horizon prediksi dan nilai preventif sistem.

## Struktur Direktori
```
MLOps-PM25-Prediction/
├── .github/
│   └── workflows/
├── .devcontainer/
│   └── devcontainer.json
├── .dvc/
│   ├── cache
│   ├── .gitignore
│   ├── config
│   └── config.local
├── config/
├── data/
├── docs/
├── models/
├── notebooks/
├── src/
│   ├── ingest_data.py
│   ├── preprocess.py
│   └── ...                 # Feature engineering, training, serving (tahap berikutnya)
├── tests/
├── .env
├── .gitignore
├── .flake8
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── requirements.txt
├── LICENSE
└── README.md
```

*(Struktur ini akan berkembang secara bertahap mengikuti progres LK-02 s.d. LK-14 — beberapa folder seperti `models/` dan `docker-compose.yml` baru benar-benar terisi di tahap-tahap selanjutnya.)*

## Cara Menjalankan via GitHub Codespaces

Proyek ini menggunakan GitHub Codespaces sebagai *cloud development environment* yang sudah dikonfigurasi otomatis lewat `.devcontainer/devcontainer.json` — dependency dan extension VS Code akan ter-install sendiri tanpa setup manual.

1. Buka repository ini di GitHub
2. Klik tombol **Code** (hijau) di kanan atas
3. Pilih tab **Codespaces**, lalu klik **Create codespace on main**
4. Tunggu beberapa menit — Codespace akan otomatis:
   - Menyiapkan environment Python 3.11
   - Menginstall semua dependency dari `requirements.txt`
   - Menginstall extension VS Code yang dibutuhkan (Python, Jupyter, Docker, GitHub Actions)
5. Setelah selesai, langsung jalankan script tanpa instalasi tambahan (lihat bagian *Cara Menjalankan Skrip* di bawah).

Port berikut sudah otomatis di-forward untuk kebutuhan tahap selanjutnya (MLflow, Prometheus, Grafana, API serving): `8000`, `5000`, `9090`, `3000`.

## Instalasi Dependencies

**Prasyarat:** Python 3.10 atau lebih baru.

Di **GitHub Codespaces**, dependency terinstall otomatis lewat `.devcontainer/devcontainer.json`. Untuk **komputer lokal**, gunakan virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dependency yang dipakai kedua skrip

| Library | Fungsi | Dipakai di |
|---|---|---|
| `requests` | HTTP request ke API Open-Meteo dan NASA FIRMS | `ingest_data.py` |
| `pandas` | Menggabungkan dan membersihkan data tabular | `preprocess.py` |
| `python-dotenv` | Membaca `FIRMS_MAP_KEY` dari file `.env` | `ingest_data.py` |
| `black` | Auto-format kode sesuai PEP 8 | pengembangan |
| `flake8` | Linter untuk memeriksa kepatuhan PEP 8 | pengembangan |

Library lain di `requirements.txt` (`numpy`, `haversine`, `matplotlib`, `scikit-learn`, `xgboost`, `mlflow`, `dvc`, `ipykernel`) disiapkan untuk tahap berikutnya (feature engineering, training, tracking, dan versioning data) dan belum dipakai kedua skrip ini. Sumber data berupa API sehingga tidak memerlukan library scraping seperti BeautifulSoup atau Selenium.

### Konfigurasi API key

Buat file `.env` di root repositori (sudah masuk `.gitignore`, jangan di-commit):

```
FIRMS_MAP_KEY=isi_map_key_kamu
```

MAP_KEY gratis dari `https://firms.modaps.eosdis.nasa.gov/api/map_key/`. Jika belum diset, ingestion FIRMS dilewati dan weather + air quality tetap berjalan.

## Cara Menjalankan Skrip Pengumpulan Data

Jalankan dari **root repositori** (bukan dari dalam folder `src/`), karena path data bersifat relatif.

## Menjalankan Ingestion

Skrip `src/ingest_data.py` mengambil data dari tiga sumber (Open-Meteo Weather Open-Meteo Air Quality, NASA FIRMS) dan menyimpannya ke
`data/raw/` dengan nama berstempel waktu UTC, sehingga data lama tidak pernah tertimpa.

    # data terbaru (2 hari terakhir, tumpang tindih sebagai pengaman)
    python src/ingest_data.py

    # isi riwayat, mis. 30 hari ke belakang
    python src/ingest_data.py --mode backfill --backfill-days 30

Prasyarat: `FIRMS_MAP_KEY` di file `.env` (tidak pernah di-commit).

Penanganan kegagalan:
- Jika API mengembalikan status 5xx atau timeout, skrip mencoba ulang maksimal 3 kali dengan jeda 5s, 15s, 45s.
- Jika ketiga percobaan gagal, sumber lain tetap diproses dan kegagalan dicatat di `logs/ingestion_YYYYMMDD.log`. Exit code program 1 jika ada sumber yang gagal.
- Jika `FIRMS_MAP_KEY` belum diset, sumber FIRMS dihitung gagal (bukan dilewati) karena FIRMS adalah leading indicator proyek ini.

Struktur keluaran:

| Folder | Isi |
|---|---|
| `data/raw/weather/` | JSON cuaca per jam |
| `data/raw/air_quality/` | JSON PM2.5, PM10, AQI per jam |
| `data/raw/firms/` | CSV titik api per deteksi satelit |
| `logs/` | Log ingestion harian |

## Prapemrosesan

    python src/preprocess.py

Membaca semua file di `data/raw/`, membuang duplikat (baris dari file terbaru yang dipertahankan), membuang nilai kosong, memvalidasi rentang nilai, dan menyaring titik api FIRMS berdasarkan confidence. Hasilnya disimpan di `data/interim/`.

## Versioning Data dengan DVC

Data mentah (`data/raw/`) dilacak DVC, bukan Git. Git hanya menyimpan file penunjuk `data/raw.dvc` (berisi hash), sedangkan data aslinya disimpan di remote storage **Nextcloud** (WebDAV).

### Setup

    pip install -r requirements.txt
    dvc remote modify --local nextcloud user <USERNAME>
    dvc remote modify --local nextcloud password '<APP_PASSWORD>'
    dvc pull

- Gunakan *app password* (Nextcloud: Settings, Security, Devices & sessions), bukan password akun.
- Kredensial hanya disimpan lokal di `.dvc/config.local` yang tidak ikut ter-commit.

### Alur menambah versi data baru

1. Ambil data tambahan: `python src/ingest_data.py` (atau mode backfill)
2. Cek perubahan: `dvc status`
3. Lacak versi baru: `dvc add data/raw`
4. Commit penunjuknya dan beri tag:

       git add data/raw.dvc
       git commit -m "feat: perbarui dataset"
       git tag -a data-vN -m "keterangan"

5. Kirim data ke remote: `dvc push`

### Membandingkan dan memulihkan versi

    dvc diff data-v1 data-v2
    git diff data-v1 data-v2 -- data/raw.dvc

    git checkout data-v1 && dvc checkout     # kembali ke v1
    git checkout main && dvc checkout        # kembali ke versi terbaru

### Riwayat versi

| Tag | Isi | Keterangan |
|---|---|---|
| data-v1 | `5` file, `1832359` baris bersih | data awal + backfill 7 hari |
| data-v2 | `8` file, `8815819` baris bersih | ditambah backfill 30 hari |

Hash `data/raw.dvc` tiap versi: v1 `1c96e53f1abbef87d533d54b73df9029.dir`, v2 `cf18d7de25f3a9fbe1dcd25acde45a7c.dir`.

### Catatan

- `data/raw` tidak lagi dilacak Git. Setelah clone, jalankan `dvc pull`
  untuk mengambil data.
- Ingestion otomatis berkala (GitHub Actions) belum aktif dan akan
  ditambahkan pada tahap berikutnya.

## Deskripsi Format Output Data

### Data mentah — `data/raw/`

| Lokasi | Format | Isi |
|---|---|---|
| `weather/weather_<YYYYMMDD_HHMMSS>.json` | JSON | Respons Open-Meteo apa adanya |
| `air_quality/air_quality_<YYYYMMDD_HHMMSS>.json` | JSON | Respons Open-Meteo apa adanya |
| `firms/firms_<YYYYMMDD_HHMMSS>.csv` | CSV | Deteksi titik api NASA FIRMS |

File JSON berisi metadata (`latitude`, `longitude`, `timezone`, `elevation`, `hourly_units`) dan objek `hourly` berisi larik yang sejajar per jam: `time` dan variabel-variabelnya. Nilai `null` muncul di ujung horizon prakiraan yang belum tersedia.

File CSV FIRMS memiliki kolom `latitude`, `longitude`, `bright_ti4`, `scan`, `track`, `acq_date`, `acq_time` (UTC, format HHMM), `satellite`, `instrument`, `confidence`, `version`, `bright_ti5`, `frp`, `daynight`. File hanya berisi header jika tidak ada titik api pada periode itu (kondisi normal di luar musim kemarau).

### Data bersih — `data/interim/`

**`weather_aqi_clean.csv`** — satu baris per jam, terurut naik berdasarkan `time`, tanpa duplikat waktu.

| Kolom | Tipe | Satuan | Deskripsi |
|---|---|---|---|
| `time` | datetime | Asia/Jakarta (UTC+7) | Waktu pengamatan per jam |
| `temperature_2m` | float | °C | Suhu udara 2 meter |
| `relative_humidity_2m` | numerik | % | Kelembapan relatif (0-100) |
| `wind_speed_10m` | float | km/h | Kecepatan angin 10 meter |
| `precipitation` | float | mm | Curah hujan per jam |
| `cloud_cover` | numerik | % | Tutupan awan (0-100) |
| `pm2_5` | float | µg/m³ | Konsentrasi PM2.5 (calon target prediksi) |
| `pm10` | float | µg/m³ | Konsentrasi PM10 |
| `european_aqi` | numerik | indeks EAQI | Indeks kualitas udara Eropa |

Contoh baris: `2026-08-29 00:00:00,25.2,76,9.7,0.0,15,81.7,84.3,126`

**`firms_clean.csv`** — kolom sama dengan CSV FIRMS mentah, hanya berisi deteksi berkepercayaan cukup (VIIRS level `n`/`h`, atau MODIS di atas 70) tanpa duplikat. File ini tidak dibuat jika tidak ada titik api.

## Standar Kode (PEP 8)

Kode mengikuti PEP 8: nama variabel deskriptif, docstring pada setiap modul dan fungsi, dan maksimal 79 karakter per baris. Konfigurasi ada di `pyproject.toml` (black) dan `.flake8`.

```bash
black src/           # format otomatis
flake8 src/          # cek kepatuhan PEP 8
```

## Branching Strategy

Proyek ini mengikuti **GitHub Flow**:
- Branch `main` selalu dalam kondisi stabil
- Setiap eksperimen/fitur baru dikerjakan di branch terpisah dengan prefix `feat/` (contoh: `feat/initial-eda`, `feat/feature-engineering`)
- Merge ke `main` dilakukan melalui Pull Request setelah validasi (review kode, cek hasil eksperimen, pastikan tidak ada credential ter-commit)

## Tech Stack

| Kategori | Tools |
|---|---|
| Version Control & Collaboration | GitHub |
| Cloud Development | GitHub Codespaces |
| Data Versioning | DVC |
| CI/CD & Automation | GitHub Actions |
| Experiment Tracking & Registry | MLflow |
| Containerization | Docker, Docker Compose |
| Model Serving | MLflow Models |
| Monitoring | Prometheus, Grafana |

## Sumber Data

| Sumber | Kegunaan | Akses |
|---|---|---|
| [Open-Meteo Weather API](https://open-meteo.com/) | Data cuaca (suhu, kelembapan, angin, curah hujan) | Gratis, tanpa API key |
| [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) | Data PM2.5, PM10, AQI | Gratis, tanpa API key |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | Data titik api satelit (leading indicator) | Gratis, perlu MAP_KEY |
