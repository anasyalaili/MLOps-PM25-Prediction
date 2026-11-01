# MLOps-PM25-Prediction
"PREDIKSI LONJAKAN PM2.5 AKIBAT KARHUTLA DENGAN LEADING INDICATOR TITIK API SATELIT"

Sistem prediksi lonjakan konsentrasi PM2.5 akibat kebakaran hutan dan lahan (karhutla), menggunakan pendekatan leading indicator dari data titik api satelit (NASA FIRMS) yang dikombinasikan dengan data cuaca dan kualitas udara real-time.

## Tujuan Proyek
Proyek ini bertujuan membangun sistem end-to-end MLOps yang mampu:
1. Memprediksi konsentrasi PM2.5 6 jam ke depan (H+6) di wilayah rawan karhutla dengan fokus awal Kalimantan Tengah
2. Memanfaatkan data titik api satelit sebagai leading indicator, sinyal peringatan dini sebelum lonjakan PM2.5 benar-benar terukur di sensor darat
3. Menerapkan siklus continual learning penuh: data terus diperbarui secara otomatis, model di-retrain berdasarkan jadwal maupun trigger performa/drift, dan divalidasi sebelum dipromosikan ke production
Berbeda dari pendekatan prediksi AQI konvensional yang hanya mengandalkan data cuaca dan historis polutan, proyek ini menambahkan data titik api satelit (NASA FIRMS) sebagai fitur tambahan untuk meningkatkan horizon prediksi dan nilai preventif sistem.

## Status Proyek

| Komponen | Status |
|---|---|
| Pengambilan data (Open-Meteo, NASA FIRMS), termasuk backfill historis | Berjalan; otomatis tiap 3 jam lewat GitHub Actions |
| Pembersihan dan feature engineering | Berjalan; otomatis mingguan lewat GitHub Actions |
| Versioning data (DVC, remote Nextcloud) | Berjalan |
| Training dan pelacakan eksperimen (MLflow) | Berjalan; dijalankan manual |
| Model Registry (registrasi, stage, silsilah, inferensi) | Berjalan; dijalankan manual lewat `src/register_model.py` |
| Retraining otomatis berdasarkan jadwal atau drift | Belum diimplementasi |
| Serving model (API), monitoring, dan containerization | Belum diimplementasi |

Hasil terbaik saat ini: ExtraTrees dengan target selisih (`delta`) dan 28 fitur
tanpa hotspot, CV RMSE 315,3 µg/m³ dibanding 354,3 pada baseline "jam yang sama
kemarin" (sekitar 11% lebih baik). Fitur titik api tidak terbukti menambah
akurasi pada horizon 6 jam di data ini (lihat *Hasil Eksperimen Skema v4*).

## Mulai Cepat

Mereproduksi model terbaik dari nol (butuh akses ke remote data Nextcloud):

```bash
git clone https://github.com/anasyalaili/MLOps-PM25-Prediction.git
cd MLOps-PM25-Prediction
python3 -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# kredensial remote data (disimpan lokal, tidak di-commit)
dvc remote modify --local nextcloud user <USERNAME>
dvc remote modify --local nextcloud password '<APP_PASSWORD>'
dvc pull                                             # ambil data sesuai commit ini

# latih ulang champion
python src/train.py --model extra --n-estimators 300 --max-depth 12 \
    --min-samples-leaf 10 --max-features 0.5 --target-mode delta --no-hotspot \
    --cv-folds 4 --data-version data-v4 --experiment pm25-karhutla-h6-v4 \
    --run-name 18-extra-delta-nohotspot

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

`dvc pull` mengambil data sesuai pointer di commit yang sedang di-checkout, jadi
pastikan commit itu memuat dataset skema v4 (tag `data-v4` atau yang lebih baru).

Untuk menambah data, membangun ulang dataset, atau mendaftarkan model, ikuti
bagian *Alur Lengkap* dan *Model Registry*. Dengan Codespaces, lingkungan Python
sudah siap (lihat *Cara Menjalankan via GitHub Codespaces*).

## Daftar Isi

- [Tujuan Proyek](#tujuan-proyek) dan [Status Proyek](#status-proyek)
- [Mulai Cepat](#mulai-cepat)
- [Struktur Direktori](#struktur-direktori)
- [Instalasi](#instalasi-dependencies) dan [Codespaces](#cara-menjalankan-via-github-codespaces)
- Pipeline data: [Ingestion](#menjalankan-ingestion), [Prapemrosesan](#prapemrosesan), [Feature Engineering](#feature-engineering)
- [Versioning Data dengan DVC](#versioning-data-dengan-dvc)
- Otomatisasi: [Ingestion](#ingestion-otomatis-github-actions), [Transform dan Load](#transform-dan-load-mingguan-github-actions)
- [Deskripsi Format Output Data](#deskripsi-format-output-data)
- [Eksperimen Training dan MLflow](#eksperimen-training-dan-mlflow)
- [Model Registry](#model-registry-mlflow)
- [Alur Lengkap](#alur-lengkap)
- [Standar Kode](#standar-kode-pep-8), [Branching](#branching-strategy), [Tech Stack](#tech-stack), [Sumber Data](#sumber-data)

## Struktur Direktori
```
MLOps-PM25-Prediction/
├── .github/
│   └── workflows/
│       ├── etl_pipeline.yml          # ingestion tiap 3 jam
│       └── etl_transform_load.yml    # transform dan load mingguan
├── .devcontainer/
│   └── devcontainer.json
├── .dvc/
│   ├── .gitignore
│   └── config                        # remote Nextcloud (tanpa kredensial)
├── data/
│   ├── raw.dvc                       # pointer DVC data mentah
│   └── processed.dvc                 # pointer DVC dataset siap latih
├── models/
├── notebooks/
├── src/
│   ├── ingest_data.py
│   ├── preprocess.py
│   ├── build_features.py
│   ├── train.py
│   └── register_model.py
├── .dvcignore
├── .flake8
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

File dan folder berikut dibuat saat dijalankan dan tidak masuk Git:

| Lokasi | Isi |
|---|---|
| `data/raw/`, `data/interim/`, `data/processed/` | Data hasil pipeline (raw dan processed dilacak DVC, interim tidak dilacak) |
| `.env` | `FIRMS_MAP_KEY` |
| `.dvc/config.local`, `.dvc/cache` | Kredensial Nextcloud dan cache DVC lokal |
| `mlflow.db`, `mlruns/` | Backend SQLite dan artefak MLflow |
| `models/registry/` | Metadata model (JSON), dilacak DVC lewat `models/registry.dvc` |
| `logs/` | Log ingestion harian |

## Cara Menjalankan via GitHub Codespaces

Proyek ini menggunakan GitHub Codespaces sebagai *cloud development environment* yang sudah dikonfigurasi otomatis lewat `.devcontainer/devcontainer.json` — dependency dan extension VS Code akan ter-install sendiri tanpa setup manual.

1. Buka repository ini di GitHub
2. Klik tombol **Code** (hijau) di kanan atas
3. Pilih tab **Codespaces**, lalu klik **Create codespace on main**
4. Tunggu beberapa menit — Codespace akan otomatis:
   - Menyiapkan environment Python 3.11
   - Menginstall semua dependency dari `requirements.txt`
   - Menginstall extension VS Code yang dibutuhkan (Python, Docker, GitHub Actions)
5. Setelah selesai, langsung jalankan script tanpa instalasi tambahan (lihat bagian *Alur Lengkap* di bawah).

Port berikut sudah otomatis di-forward untuk kebutuhan tahap selanjutnya (MLflow, Prometheus, Grafana, API serving): `8000`, `5000`, `9090`, `3000`.

## Instalasi Dependencies

**Prasyarat:** Python 3.11 atau lebih baru (versi `pandas` dan `numpy` di `requirements.txt` tidak mendukung Python 3.10).

Di **GitHub Codespaces**, dependency terinstall otomatis lewat `.devcontainer/devcontainer.json`. Untuk **komputer lokal**, gunakan virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dependency utama

| Library | Fungsi | Dipakai di |
|---|---|---|
| `requests` | HTTP request ke API Open-Meteo dan NASA FIRMS | `ingest_data.py` |
| `python-dotenv` | Membaca `FIRMS_MAP_KEY` dari file `.env` | `ingest_data.py` |
| `pandas` | Menggabungkan dan membersihkan data tabular | `preprocess.py`, `build_features.py`, `train.py` |
| `numpy` | Operasi numerik dan transformasi log | `build_features.py`, `train.py` |
| `pyarrow` | Membaca dan menulis parquet (`data/processed`) | `build_features.py`, `train.py` |
| `scikit-learn`, `xgboost` | Model regresi (Linear, RF, GBR, XGBoost) dan metrik | `train.py` |
| `mlflow` | Pencatatan eksperimen dan model | `train.py` |
| `matplotlib` | Grafik feature importance (opsional, dilewati jika tidak ada) | `train.py` |
| `dvc[webdav]` | Versioning data dengan remote Nextcloud | seluruh pipeline data |
| `black`, `flake8` | Auto-format dan linter PEP 8 | pengembangan |
| `ipykernel` | Kernel Jupyter | notebook |

Library `haversine` ada di `requirements.txt` tetapi belum dipakai: jarak
great-circle dihitung oleh fungsi `haversine_km` di `build_features.py`.
Sumber data berupa API sehingga tidak memerlukan library scraping seperti
BeautifulSoup atau Selenium.

### Konfigurasi API key

Buat file `.env` di root repositori (sudah masuk `.gitignore`, jangan di-commit):

```
FIRMS_MAP_KEY=isi_map_key_kamu
```

MAP_KEY gratis dari `https://firms.modaps.eosdis.nasa.gov/api/map_key/`. Jika belum diset, sumber FIRMS dihitung gagal (bukan dilewati) karena FIRMS adalah leading indicator proyek ini, sedangkan weather dan air quality tetap diproses.

## Menjalankan Ingestion

Semua skrip dijalankan dari **root repositori** (bukan dari dalam folder `src/`), karena path data bersifat relatif.

Skrip `src/ingest_data.py` mengambil data dari tiga sumber (Open-Meteo Weather, Open-Meteo Air Quality, NASA FIRMS) dan menyimpannya ke
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

### Backfill historis (mode `historical`)

Mode `backfill` hanya bisa mundur 92 hari. Untuk riwayat yang lebih panjang
(mis. musim kemarau tahun-tahun sebelumnya) gunakan mode `historical`
dengan rentang tanggal:

    python src/ingest_data.py --mode historical \
        --start-date 2023-06-01 --end-date 2023-11-30
    python src/ingest_data.py --mode historical \
        --start-date 2024-06-01 --end-date 2024-11-30
    python src/ingest_data.py --mode historical \
        --start-date 2025-06-01 --end-date 2025-11-30

Sumber arsip yang dipakai:

| Data | Sumber di mode `historical` | Catatan |
|---|---|---|
| Cuaca | `historical-forecast-api.open-meteo.com` | Potongan jam-jam awal tiap run model yang disambung. Dokumentasi Open-Meteo menyebut kurang cocok untuk deret waktu sangat panjang karena versi model berubah |
| Air quality | `air-quality-api.open-meteo.com` dengan `start_date`/`end_date` | CAMS global, tersedia sejak Agustus 2022 |
| FIRMS | `VIIRS_SNPP_SP` (standard processing) | Arsip permanen, dipanggil per 5 hari. Tanggal awalnya diundur 2 hari dari `--start-date` agar fitur hotspot 24 jam dan tren 48 jam di jam-jam awal sudah terisi |

Aturan dan batasan:
- `--start-date` dan `--end-date` wajib, `--end-date` harus sebelum hari ini,
  `--start-date` tidak boleh sebelum 2022-08-01, dan satu run maksimal 366 hari.
- File hasilnya bernama `weather_hist_<mulai>_<akhir>_<timestamp>.json`
  (dan padanannya untuk `air_quality` dan `firms`). Penanda `_hist_` membuat
  `preprocess.py` memprioritaskan data live pada tanggal yang tumpang tindih,
  sehingga data hasil ingestion otomatis tidak tertimpa arsip.
- Pilih `--end-date` yang tidak tumpang tindih dengan data live. Arsip SP
  FIRMS tertinggal beberapa bulan dari hari ini, jadi biasanya akan ada
  jeda antara akhir data historis dan awal data live. Jeda ini aman:
  `build_features.py` menyusun deret per jam penuh dan membuang jam tanpa
  data, sehingga lag dan statistik bergulir tidak menyambung melewati jeda.
- Arsip FIRMS SP di akun kamu belum diverifikasi. Jika run menghasilkan
  FIRMS gagal atau kosong, cek ketersediaan tanggalnya lewat endpoint
  `data_availability` NASA FIRMS.
- Setelah selesai, lacak dan versikan seperti biasa (`dvc add data/raw`,
  `dvc push`, commit pointer). Menambah data tanpa mengubah skema adalah
  kenaikan MINOR menurut aturan versioning (mis. `data-v4.1`), dan dataset
  processed perlu dibangun ulang dari seluruh raw sebelum training.

## Prapemrosesan

    python src/preprocess.py

Membaca semua file di `data/raw/`, membuang duplikat (baris dari file terbaru yang dipertahankan), membuang nilai kosong, memvalidasi rentang nilai, dan menyaring titik api FIRMS berdasarkan confidence. Hasilnya disimpan di `data/interim/`.

## Feature Engineering

    python src/build_features.py

Membaca `data/interim/` lalu membangun dataset skema v4 (rincian kolom di bagian *Data siap latih* di bawah):

- join weather dan AQI pada `time`, lalu label `target_pm2_5_h6` (PM2.5 pada t+6 jam);
- fitur PM2.5: lag 1, 6, 12, 18, 24 jam, rata-rata dan maksimum bergulir 6 dan 24 jam, selisih 1 dan 6 jam;
- cuaca pada t+6 jam dan jumlah curah hujan 24 jam terakhir;
- fitur hotspot FIRMS dalam jendela 24 jam (jarak haversine): jumlah pada radius 50, 100, 200 km, jumlah FRP, jarak titik api terdekat, dan tren terhadap 24 jam sebelumnya;
- fitur waktu (`hour_of_day`, `hour_sin`, `hour_cos`, `is_dry_season`).

Hasilnya disimpan per bulan di `data/processed/year=YYYY/month=MM/data.parquet`. Partisi bulan yang sama ditimpa, sehingga skrip aman dijalankan ulang. Opsi `--cutoff 'YYYY-MM-DD HH:MM'` (WIB) menetapkan jam observasi terakhir, default jam sekarang. Semua fitur hanya memakai data pada waktu t atau sebelumnya, kecuali cuaca `_h6` yang di inferensi berasal dari prakiraan cuaca.

## Versioning Data dengan DVC

Data mentah (`data/raw/`) dilacak DVC, bukan Git. Git hanya menyimpan file penunjuk `data/raw.dvc` (berisi hash), sedangkan data aslinya disimpan di remote storage **Nextcloud** (WebDAV).

### Setup

    pip install -r requirements.txt
    dvc remote modify --local nextcloud user <USERNAME>
    dvc remote modify --local nextcloud password '<APP_PASSWORD>'
    dvc pull

- Gunakan *app password* (Nextcloud: Settings, Security, Devices & sessions), bukan password akun.
- Kredensial hanya disimpan lokal di `.dvc/config.local` yang tidak ikut ter-commit.

### Skema penomoran versi

Tag data memakai pola `data-v<MAJOR>.<MINOR>`:
- **MAJOR** naik saat skema berubah: kolom fitur ditambah atau dihapus, definisi
  label berubah, atau horizon prediksi berubah (mis. `data-v3` ke `data-v4`).
- **MINOR** naik saat hanya ada penambahan data tanpa perubahan skema (mis.
  `data-v4` ke `data-v4.1`).

### Alur menambah versi data baru

Ringkasan di bawah untuk data mentah saja. Alur lengkap sampai dataset siap latih
ada di bagian *Alur Lengkap*.

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
| data-v1 | raw: `5` file, 1.832.359 byte | data awal + backfill 7 hari |
| data-v2 | raw: `8` file, 8.815.819 byte | ditambah backfill 30 hari |
| data-v2.1 | raw: `11` file, 20.743.552 byte | ingestion tambahan, skema tidak berubah |
| data-v3 | raw sama dengan v2.1, processed: `3` file, 87.210 byte | dataset fitur skema v3, dipakai eksperimen awal |
| data-v4 | raw ditambah backfill historis 2023-2025: `23` file, 29.469.125 byte, processed: `22` file, 1.534.629 byte | skema v4 (39 kolom), dipakai eksperimen Run 10-19 |

Tag berikutnya (`data-v4.1` dan seterusnya) dibuat otomatis oleh workflow mingguan, daftar lengkap lewat `git tag -n1`.

Hash `data/raw.dvc` tiap versi: v1 `1c96e53f1abbef87d533d54b73df9029.dir`, v2 `cf18d7de25f3a9fbe1dcd25acde45a7c.dir`, v2.1 dan v3 `85cd1cc2307dfde7156aa306e9968204.dir`; processed v3: `cafaf391ad2d33c101b58c881cf8cc59.dir`, v4: `d62a6bcb751cf2ff60c80c518898d958.dir`; processed v4: `8efd9610359d4039c6f7493a714f144b.dir`.

### Catatan

- `data/raw` tidak lagi dilacak Git. Setelah clone, jalankan `dvc pull`
  untuk mengambil data.
- Ingestion otomatis berkala dijalankan oleh GitHub Actions (lihat bagian
  *Ingestion Otomatis* di bawah). Pointer `data/raw.dvc` di-commit otomatis
  oleh workflow. Tag versi dibuat otomatis oleh workflow mingguan
  Transform dan Load, atau manual untuk perubahan skema.

## Ingestion Otomatis (GitHub Actions)

Workflow `.github/workflows/etl_pipeline.yml` menjalankan tahap Extract
secara otomatis. Tahap Transform dan Load punya workflow terpisah dengan
jadwal mingguan (lihat bagian *Transform dan Load Mingguan* di bawah),
karena `data/processed` dipakai sebagai versi dataset training yang harus
reprodusibel dan tidak boleh berubah tiap 3 jam.

| Aspek | Keterangan |
|---|---|
| Jadwal | Tiap 3 jam (`17 */3 * * *`, UTC) dan bisa dijalankan manual lewat `workflow_dispatch` |
| Alasan 3 jam | Sinyal H+6 tetap segar, tanpa mengambil berulang-ulang data yang hanya berubah sekitar 2 kali sehari (model CAMS di Open-Meteo Air Quality dan lintasan satelit FIRMS) |
| Alasan menit ke-17 | Cron GitHub Actions sering tertunda di awal jam. Run yang terlewat atau telat tertutup overlap 2 hari di mode `incremental` |
| Langkah | `dvc pull data/raw.dvc`, `ingest_data.py`, `dvc add data/raw`, `dvc push`, commit `data/raw.dvc` |
| Sumber gagal | Data dari sumber yang berhasil tetap disimpan, lalu job ditandai gagal. Log tersedia sebagai artifact `ingestion-logs-<run_id>` (7 hari) |
| Run manual | Actions, `ETL Pipeline (Ingestion)`, `Run workflow`, pilih mode `incremental` atau `backfill` |

Workflow melakukan `dvc pull` lebih dulu karena runner GitHub selalu kosong,
sedangkan `dvc add data/raw` melacak seluruh folder. Pointer juga harus
di-commit di setiap run, kalau tidak data yang sudah di-push ke Nextcloud
tidak terhubung ke commit mana pun. Akibatnya ada sekitar 8 commit otomatis
per hari dengan pesan `chore(data): ingestion otomatis <waktu> [skip ci]`.

### Setup GitHub Secrets

File `.env` dan `.dvc/config.local` hanya ada di komputer lokal dan tidak
ikut ter-commit, sehingga runner GitHub Actions tidak bisa membacanya.
Kredensial yang sama harus didaftarkan di **Settings, Secrets and variables,
Actions, New repository secret**:

| Secret | Isi | Setara dengan |
|---|---|---|
| `NEXTCLOUD_USER` | Username Nextcloud | `user` di `.dvc/config.local` |
| `NEXTCLOUD_APP_PASSWORD` | App password Nextcloud | `password` di `.dvc/config.local` |
| `FIRMS_MAP_KEY` | MAP_KEY NASA FIRMS | `FIRMS_MAP_KEY` di `.env` |

Catatan:
- Disarankan membuat app password Nextcloud khusus GitHub Actions (Settings,
  Security, Devices & sessions), terpisah dari yang dipakai di komputer lokal,
  agar mudah dicabut tanpa mengganggu akses lain.
- Di **Settings, Actions, General, Workflow permissions** pilih *Read and
  write permissions* agar workflow bisa mem-push pointer `raw.dvc`.
- Jika `main` diproteksi (wajib Pull Request), push dari bot akan ditolak.
  Beri pengecualian untuk `github-actions[bot]` atau pindahkan commit pointer
  ke branch khusus.
- Untuk repositori publik, GitHub menonaktifkan workflow terjadwal setelah 60
  hari tanpa aktivitas repositori.

## Transform dan Load Mingguan (GitHub Actions)

Workflow `.github/workflows/etl_transform_load.yml` mengolah data raw yang
sudah terkumpul menjadi dataset siap latih di `data/processed/`.

| Aspek | Keterangan |
|---|---|
| Jadwal | Mingguan, Senin 00:47 WIB (`47 17 * * 0`, UTC), dan bisa dijalankan manual lewat `workflow_dispatch` |
| Langkah | `dvc pull data/raw.dvc`, `preprocess.py`, `build_features.py`, verifikasi parquet, `dvc add data/processed`, `dvc push`, commit `data/processed.dvc`, tag |
| Cakupan data | `data/processed` dibangun ulang penuh dari seluruh raw, bukan di-append, sehingga hasilnya selalu konsisten dengan raw |
| Tag | Jika isi dataset berubah, dibuat tag MINOR berikutnya dari tag `data-v*` tertinggi (mis. `data-v4` menjadi `data-v4.1`, lalu `data-v4.2`, dan seterusnya) |
| Tanpa perubahan | Jika `data/processed.dvc` tidak berubah, tidak ada commit maupun tag |
| Perubahan skema | Kenaikan MAJOR (mis. `data-v4`) tetap manual, sesuai aturan versioning |

Catatan:
- Workflow ini berbagi grup `concurrency` dengan ingestion, sehingga keduanya
  tidak pernah berjalan bersamaan dan tidak saling menimpa pointer DVC.
- Pada run manual, centang `create_tag` untuk membuat tag baru atau kosongkan
  agar hanya memperbarui pointer.
- Secret dan pengaturan yang dibutuhkan sama dengan workflow ingestion
  (lihat *Setup GitHub Secrets*), kecuali `FIRMS_MAP_KEY` yang tidak dipakai.
- Tag `data-v4` harus sudah ada sebelum workflow ini aktif (lihat *Data siap latih*), kalau tidak run pertama membuat `data-v3.1` untuk dataset skema v4.
- Nilai `cutoff` di `build_features.py` default-nya jam sekarang (WIB), jadi
  isi dataset bergantung pada waktu run.
- Tag otomatis berarti riwayat versi di README tidak lagi memuat semua tag.
  Daftar lengkap bisa dilihat dengan `git tag -n1`.

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

### Data siap latih — `data/processed/` (skema v4)

Satu baris per jam, partisi bulanan (parquet), 39 kolom. Semua kolom skema v3 tetap ada dengan arti yang sama. Skema v4 menambah 22 kolom baru.

| Kelompok | Kolom | Keterangan |
|---|---|---|
| Kunci | `timestamp`, `location` | Waktu (WIB) dan nama lokasi |
| Cuaca saat t | `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `precipitation`, `cloud_cover` | Sama dengan v3 |
| Polutan saat t | `pm2_5`, `pm10`, `european_aqi` | Sama dengan v3 |
| Lag PM2.5 | `pm2_5_lag_1h`, `pm2_5_lag_6h`, **`pm2_5_lag_12h`, `pm2_5_lag_18h`, `pm2_5_lag_24h`** | Lag 18 jam sama dengan nilai yang dipakai baseline "jam yang sama kemarin" |
| Statistik bergulir | **`pm2_5_rollmean_6h`, `pm2_5_rollmax_6h`, `pm2_5_rollmean_24h`, `pm2_5_rollmax_24h`** | Rata-rata dan maksimum, termasuk jam t |
| Tren | **`pm2_5_diff_1h`, `pm2_5_diff_6h`** | PM2.5 saat t dikurangi lag 1 dan 6 jam |
| Hujan | **`precip_sum_24h`** | Jumlah curah hujan 24 jam terakhir |
| Cuaca pada t+6 | **`temperature_2m_h6`, `relative_humidity_2m_h6`, `wind_speed_10m_h6`, `precipitation_h6`, `cloud_cover_h6`** | Diambil dari jam t+6; di inferensi dari prakiraan cuaca |
| Hotspot | `hotspot_count_100km`, `hotspot_count_log`, **`hotspot_count_50km`, `hotspot_count_200km`, `hotspot_frp_sum_100km`, `hotspot_nearest_km`, `hotspot_trend_100km`** | Jendela 24 jam, lihat di bawah |
| Waktu | `hour_of_day`, **`hour_sin`, `hour_cos`**, `is_dry_season` | Siklik: jam 23 dan jam 0 berdekatan |
| Label | `target_pm2_5_h6` | PM2.5 pada t+6 jam |

Cetak tebal adalah kolom baru di v4. Kolom hotspot:

| Kolom | Definisi |
|---|---|
| `hotspot_count_{50,100,200}km` | Jumlah titik api dalam radius tersebut dari pusat kota, pada (t-24 jam, t) |
| `hotspot_frp_sum_100km` | Jumlah FRP (daya radiasi kebakaran, MW) titik api dalam 100 km, 24 jam terakhir |
| `hotspot_nearest_km` | Jarak titik api terdekat (km) dalam 24 jam terakhir. Jika tidak ada diisi 400 |
| `hotspot_trend_100km` | Jumlah 24 jam terakhir dikurangi jumlah 24 jam sebelumnya (dalam 100 km). Positif berarti kebakaran meningkat |

Alasan perubahan:
- **Lag musiman (12, 18, 24 jam) dan statistik bergulir.** Pada eksperimen awal (Run 01-09), baseline "jam yang sama kemarin" hampir sama baiknya dengan model, dan model hanya mengandalkan suhu dan jam. Nilai PM2.5 kemarin pada jam yang sama sebelumnya tidak ada di fitur model.
- **Selisih dan tren** membantu mendeteksi lonjakan, bukan hanya level PM2.5.
- **Cuaca pada t+6.** Hujan dan angin enam jam ke depan memengaruhi asap, dan datanya tersedia lewat prakiraan Open-Meteo saat prediksi dibuat.
- **Fitur hotspot lebih kaya.** Satu hitungan pada radius 100 km tidak membedakan api dekat dan jauh, api besar dan kecil, atau kebakaran yang membesar dan mengecil. `hotspot_count_200km` terpotong bounding box FIRMS di timur dan selatan (jarak batas sekitar 175 dan 145 km dari kota), jadi cakupan radius 200 km tidak penuh.
- **`hour_sin` dan `hour_cos`** menggantikan ketidaksinambungan jam 23 ke 0 bagi model berbasis jarak atau linear. Model pohon tetap memakai `hour_of_day`.

Dampak ke dataset:
- **Jumlah baris berkurang sekitar 1 hari.** Fitur tren hotspot membutuhkan 48 jam cakupan FIRMS (sebelumnya 24 jam), dan lag serta statistik 24 jam membutuhkan 24 jam riwayat PM2.5.
- **Perubahan skema = kenaikan MAJOR**, sesuai aturan penomoran versi data (lihat *Skema penomoran versi*): penambahan kolom fitur. Dataset ini harus diberi tag `data-v4`, bukan `data-v3.x`.
- `hotspot_count_log` dan `is_dry_season` tetap ada di dataset, tetapi tidak dipakai `train.py` secara default: `hotspot_count_log` identik dengan `hotspot_count_100km` bagi model pohon, dan `is_dry_season` konstan (selalu 1) selama data hanya mencakup satu musim kemarau. Keduanya bisa ditambahkan dengan `--add-features`.
- Dataset `data-v3` tidak bisa dipakai `train.py` versi ini. Skrip akan berhenti dengan pesan kolom tidak ada. Hasil eksperimen awal (Run 01-09) dibuat dengan skema v3 dan tidak langsung sebanding dengan run di skema v4.

Cara membuat dataset v4 dan memberi versi:

    dvc pull data/raw.dvc
    python src/preprocess.py
    python src/build_features.py
    dvc add data/processed
    dvc push
    git add data/processed.dvc
    git commit -m "feat: dataset skema v4"
    git tag -a data-v4 -m "skema v4: fitur lag, rolling, cuaca t+6, hotspot multi-radius"

Buat tag `data-v4` sebelum workflow mingguan berjalan dengan kode baru. Workflow memilih tag `data-v*` tertinggi lalu menaikkan MINOR; jika `data-v4` belum ada, ia akan membuat `data-v3.1` untuk dataset skema v4.

## Eksperimen Training dan MLflow

Eksperimen awal (Run 01-09) dilatih di atas dataset ber-versi DVC **`data-v3`**
(`data/processed`, skema v3) dan dicatat ke MLflow (backend SQLite
`mlflow.db`, experiment `pm25-karhutla-h6`). Setiap run mencatat parameter,
metrik, grafik feature importance, model, dan jejak silsilah data
(`git_commit`, `data_version`, md5 pointer DVC).

Run 01-09 memakai skema v3 (14 fitur). Eksperimen berikutnya (Run 10-19)
memakai skema v4 (34 fitur secara default) di experiment terpisah
`pm25-karhutla-h6-v4` (lihat *Hasil Eksperimen Skema v4*). `train.py` sekarang
membaca skema v4 dan punya opsi tambahan (lihat *Opsi training*), sehingga
hasil kedua kelompok perlu dibandingkan dengan hati-hati.

### Metodologi Evaluasi

- **Tugas:** regresi PM2.5 enam jam ke depan (`target_pm2_5_h6`). Eksperimen awal
  memakai 14 fitur (cuaca, polutan, lag 1 jam dan 6 jam, hotspot FIRMS, waktu);
  skema v4 memakai 34 fitur.
- **Split kronologis** (tanpa diacak): test = 7 hari terakhir, validasi = 7 hari
  sebelumnya, dengan jeda 6 jam agar tidak ada kebocoran label.
- **Walk-forward cross-validation:** 4 fold berurutan, masing-masing 7 hari.
  Tiap fold hanya dilatih dari data yang lebih lama.
- **Dua baseline** dihitung di setiap run:
  persistensi (prediksi = PM2.5 saat ini) dan musiman
  ("jam yang sama kemarin").
- **Aturan pemilihan model:** kriteria utama `cv_rmse_mean`; selisih yang masih
  di dalam simpangan baku dianggap setara; set test hanya dilaporkan, tidak
  dipakai untuk memilih.
- **Metrik kelas:** `gt150` dan `gt500` (PM2.5 di atas ambang) serta
  **lonjakan** (PM2.5 jam t+6 lebih dari 1,5 kali nilai saat ini). Pada skema v4
  ambangnya 150,5, 250,5, dan 500 (kategori ISPU).

### Hasil Eksperimen Awal (skema v3)

| No | Run | CV RMSE (± std) | CV MAE | Val RMSE | Test RMSE | CV spike P / R | Menang vs baseline* |
|---|---|---|---|---|---|---|---|
| 01 | Linear Regression | 892,7 ± 807,7 | 365,6 | 340,5 | 634,4 | 0,769 / 0,838 | 0 |
| 02 | RF 100 pohon, depth 5 | 434,1 ± 117,7 | 288,4 | 300,5 | 610,0 | 0,902 / 0,769 | 2 |
| **03** | **RF 300 pohon, depth 10 (champion)** | **423,8 ± 124,1** | **285,0** | 261,8 | 609,3 | 0,915 / 0,759 | 2 |
| 04 | RF 300/10, min leaf 3 | 426,9 ± 125,1 | 285,5 | 262,3 | 608,9 | 0,915 / 0,759 | 2 |
| 05 | Gradient Boosting 300/3 (challenger) | 432,3 ± 126,4 | 293,5 | 281,9 | 588,8 | 0,896 / 0,763 | 2 |
| 06 | XGBoost 200/4 | 434,9 ± 122,4 | 290,7 | 268,7 | 605,6 | 0,913 / 0,764 | 2 |
| 07 | RF tanpa hotspot | 431,8 ± 83,2 | 288,5 | 308,6 | 580,7 | 0,911 / 0,779 | 3 |
| 08 | RF tanpa `european_aqi` | 427,7 ± 131,5 | 288,1 | 259,3 | 623,0 | 0,903 / 0,759 | 2 |
| 09 | RF dengan log-target | 446,6 ± 163,2 | 299,5 | 298,1 | 671,6 | 0,922 / 0,713 | 3 |
| – | Baseline "jam yang sama kemarin" | 442,8 ± 144,8 | 327,7 | 340,5 | 556,9 | 0,667 / 0,835 | – |
| – | Baseline persistensi | – | – | – | 1104,5 | 0 / 0 | – |

\* Jumlah fold (dari 4) tempat RMSE model lebih kecil daripada baseline musiman.
Satuan RMSE dan MAE: µg/m³. Spike P / R = precision / recall kelas lonjakan.

### Champion dan challenger

- **Champion: Run 03, Random Forest (300 pohon, depth 10, 14 fitur)**,
  karena `cv_rmse_mean` terendah (423,8).
- **Challenger: Run 05, Gradient Boosting (300 pohon, depth 3)**, model
  terbaik dari keluarga algoritma lain. Performanya setara dengan champion
  dan tidak membutuhkan pustaka tambahan.
- Model berbasis pohon (RF, GBR, XGBoost) berselisih sekitar 3% di CV, jauh di
  bawah simpangan baku antar fold (sekitar 125), sehingga secara statistik
  dianggap setara. Linear Regression tidak stabil (std 807,7).

### Temuan Utama

1. **Champion vs baseline musiman:** CV RMSE turun 4,3% (423,8 vs 442,8) dan CV
   MAE turun 13,0% (285,0 vs 327,7). Selisih RMSE berada di dalam simpangan
   baku antar fold, model hanya menang di 2 dari 4 fold, dan di minggu test
   terakhir baseline lebih baik (RMSE 556,9 vs 609,3). Nilai tambah model
   moderat dan belum konsisten.
2. **Ablation hotspot FIRMS (Run 07 vs 03):** penurunan CV RMSE akibat hotspot
   hanya (431,8 - 423,8) / 431,8 = **1,8%**, di bawah target awal proyek sebesar 10%.
   Hasil validasi menguntungkan hotspot (-15,2%), tetapi di test hotspot
   justru 4,9% lebih buruk, dan model tanpa hotspot menang di 3 dari 4 fold.
   Importance fitur hotspot di GBR, XGBoost, dan RF juga tidak berbeda dari
   nol. Kontribusi hotspot pada horizon 6 jam tidak terbukti pada data ini.
3. **`european_aqi` (Run 08 vs 03):** selisih CV RMSE di bawah 1%, tidak
   berpengaruh berarti.
4. **Log-target (Run 09):** lebih buruk di semua evaluasi dan CV RMSE-nya
   melewati baseline musiman (446,6 vs 442,8). Tidak digunakan.
5. **Hyperparameter:** RF lebih dalam dengan lebih banyak pohon (Run 03)
   sedikit lebih baik daripada 100 pohon depth 5 (Run 02); `min_samples_leaf` 3
   (Run 04) tidak mengubah hasil.
6. **Feature importance (permutation, data validasi):** model terutama
   mengandalkan `temperature_2m` dan `hour_of_day`, yaitu siklus harian
   PM2.5, bukan sinyal kebakaran.


### Hasil Eksperimen Skema v4

Dilatih pada dataset **`data-v4`** (backfill musim kemarau 2023-2025 ditambah
data live) dan dicatat di experiment `pm25-karhutla-h6-v4`. Metodologi sama
dengan eksperimen awal: split kronologis, walk-forward CV 4 fold x 7 hari, pemilihan
berdasarkan `cv_rmse_mean` dengan selisih dalam simpangan baku dianggap setara.
Data latih fold terkecil kini 13.983 baris (di eksperimen awal hanya beberapa ratus).
Baseline musiman "jam yang sama kemarin" memiliki CV RMSE **354,3**.

| No | Run | Model (parameter) | Fitur | CV RMSE (± std) | Test RMSE | Menang vs baseline* |
|---|---|---|---|---|---|---|
| 10 | `10-rf-level` | RF 300/10, target `level` | 34 | 338,1 ± 166,7 | 72,6 | 3 |
| 11 | `11-rf-delta` | RF 300/10, target `delta` | 34 | 327,3 ± 187,4 | 40,0 | 3 |
| 12 | `12-rf-logratio` | RF 300/10, target `log-ratio` | 34 | 365,3 ± 202,7 | 42,4 | 2 |
| 13 | `13-extra-delta` | ExtraTrees 300/12, leaf 10, max_features 0,5, `delta` | 34 | 328,0 ± 182,6 | 40,6 | 4 |
| 14 | `14-rf-delta-reg` | RF 300/12, leaf 10, max_features 0,5, `delta` | 34 | 332,0 ± 185,1 | 41,2 | 3 |
| 15 | `15-gbr-huber-delta` | GBR 300/3, loss huber, subsample 0,8, `delta` | 34 | 349,2 ± 184,8 | 41,5 | 2 |
| 16 | `16-xgb-mae-delta` | XGBoost 300/4, lr 0,05, loss MAE, subsample 0,8, `delta` | 34 | 392,8 ± 213,9 | 45,6 | 2 |
| 17 | `17-linear-delta` | Linear Regression, `delta` | 34 | 363,9 ± 183,2 | 51,8 | 3 |
| **18** | **`18-extra-delta-nohotspot`** | **Konfigurasi Run 13 tanpa fitur hotspot** | **28** | **315,3 ± 180,9** | **37,5** | **4** |
| 19 | `19-extra-delta-dryseason` | Konfigurasi Run 13 + `is_dry_season` + `hotspot_count_log` | 36 | 325,2 ± 182,1 | 41,0 | 4 |
| - | Baseline "jam yang sama kemarin" | - | - | 354,3 | - | - |

\* Jumlah fold (dari 4) tempat RMSE model lebih kecil daripada baseline musiman.
Satuan RMSE: ug/m3. Hasil v4 tidak sebanding langsung dengan eksperimen awal: data dan
periode fold berbeda.

**Champion: Run 18** (ExtraTrees 300 pohon, depth 12, `min_samples_leaf` 10,
`max_features` 0,5, target `delta`, 28 fitur tanpa hotspot). Run 18 memiliki CV
RMSE terendah, menang di 4 dari 4 fold, dan test RMSE terendah. **Challenger:
Run 13**, konfigurasi yang sama dengan fitur hotspot.

Temuan:
1. **Hotspot tidak memberi manfaat terukur.** Membuang semua fitur hotspot
   (Run 18 vs 13) menurunkan CV RMSE 3,9% (328,0 menjadi 315,3) dan test RMSE
   7,6% (40,6 menjadi 37,5). Selisih berada di dalam simpangan baku antar fold
   (sekitar 180), jadi tidak signifikan; kesimpulan yang aman adalah hotspot
   tidak terbukti berkontribusi. Menambah `is_dry_season` dan
   `hotspot_count_log` (Run 19) hanya mengubah CV RMSE 0,9%.
2. **Feature importance (Run 13).** Fitur terpenting adalah `pm2_5_lag_18h`,
   diikuti `pm2_5`. Fitur hotspot praktis tidak terlihat pada permutation
   importance, dan nilai impurity-nya hanya sekitar 0,001-0,008. Dua ukuran
   berbeda sepakat dengan hasil ablation. Importance dihitung pada 7 hari
   validasi, sehingga tidak menutup kemungkinan hotspot berguna pada episode
   kebakaran besar atau horizon yang lebih panjang.
3. **`pm2_5_lag_18h` adalah nilai pada jam yang sama kemarin untuk target t+6**
   (t+6-24 = t-18), informasi yang sama dengan baseline musiman. Ini menjelaskan
   kenapa model jauh lebih baik dibanding eksperimen awal (yang tidak punya fitur ini) dan
   kenapa keunggulan atas baseline musiman moderat: CV RMSE 315,3 vs 354,3,
   sekitar 11% lebih baik dan menang di 4 dari 4 fold.
4. **Target `delta` lebih baik daripada `level`.** Pada konfigurasi RF yang sama
   (Run 11 vs 10), CV RMSE turun 3,2% dan test RMSE turun 45% (72,6 menjadi
   40,0). `log-ratio` (Run 12) tidak lebih baik.
5. **Model pohon mengungguli Linear Regression** sekitar 10% di CV (Run 17).
   XGBoost dengan loss MAE (Run 16) dan GBR (Run 15) tidak lebih baik daripada
   ExtraTrees/RF.
6. **Perbedaan antar model teratas lebih kecil dari simpangan baku antar fold**
   (selisih sekitar 5-13 poin, simpangan baku sekitar 180), sehingga pemilihan di
   antara Run 11, 13, 18, dan 19 tidak signifikan secara statistik.

Keterbatasan: fold CV terakhir sama dengan minggu test (7 hari terakhir),
sehingga test tidak sepenuhnya independen dari seleksi berbasis CV. Nilai PM2.5
berasal dari model CAMS Open-Meteo, bukan sensor darat, dan CAMS sudah
memasukkan emisi kebakaran, yang bisa menjelaskan kecilnya kontribusi hotspot.
Data mencakup satu wilayah dan belum mencakup musim hujan secara utuh.

### Opsi training

| Opsi | Fungsi |
|---|---|
| `--model` | `rf`, `extra` (ExtraTrees), `gbr`, `xgb`, `linear` |
| `--target-mode` | `level` (default, target langsung), `delta` (target dikurangi PM2.5 saat ini), `log-ratio` (selisih log1p). Pohon tidak bisa memprediksi di atas rentang data latih, sehingga memprediksi selisih lebih tahan pergeseran level PM2.5 antar minggu |
| `--max-features` | `rf`/`extra`: pecahan fitur per split (mis. `0.5`), `sqrt`, `log2`, `none` |
| `--subsample` | `gbr`/`xgb`: pecahan baris per pohon |
| `--loss` | `gbr`/`xgb`: `squared_error`, `absolute_error`, `huber` (lebih tahan nilai ekstrem) |
| `--add-features` | Tambah fitur opsional: `hotspot_count_log`, `is_dry_season` |
| `--refit-full` | Setelah evaluasi, latih ulang model akhir pada seluruh data sebelum disimpan. Metrik tetap dari model yang hanya dilatih pada data train |

Metrik baru: `*_rmse_log` (RMSE pada log1p, tidak didominasi nilai ekstrem),
kelas `gt250` (ambang berbahaya ISPU 250,5), dan `cv_train_rows_min` (ukuran
data latih fold terkecil). Ambang `gt150` kini 150,5. Model yang disimpan ke
MLflow memprediksi target yang sudah ditransformasi, jadi untuk
`--target-mode` selain `level` hasilnya dikembalikan dengan
`restore_level()` di `train.py` (tag `target_mode` mencatat modenya).

### Cara mereproduksi

```bash
pip install -r requirements.txt
dvc pull                       # ambil data dari remote Nextcloud sesuai pointer di commit ini
# Jika data/processed belum berskema v4 (pointer masih data-v3), bangun ulang,
# lihat bagian Alur Lengkap.

# contoh (skema v4, setelah tag data-v4 dibuat)
python src/train.py --n-estimators 300 --max-depth 10 --cv-folds 4 \
    --data-version data-v4 --experiment pm25-karhutla-h6-v4 \
    --run-name 10-rf-level
python src/train.py --n-estimators 300 --max-depth 10 --cv-folds 4 \
    --target-mode delta --data-version data-v4 \
    --experiment pm25-karhutla-h6-v4 --run-name 11-rf-delta

# dashboard MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Jalankan training setelah semua perubahan kode di-commit: tiap run mencatat
`git_commit` dari `HEAD`, jadi kode yang belum di-commit tidak terlacak.

Run skema v4 dicatat di experiment terpisah `pm25-karhutla-h6-v4`, sedangkan
experiment lama `pm25-karhutla-h6` (run 01-09 skema v3) dibiarkan apa adanya.
Setiap run memuat tag `data_version` dan `git_commit` serta parameter `n_features`, sehingga
run lama dan baru bisa dibedakan dan dibandingkan di dashboard.

## Model Registry (MLflow)

Skrip `src/register_model.py` menjalankan alur MLflow Model Registry:
registrasi, versioning, perpindahan stage, pencatatan silsilah ke
DVC, dan uji inferensi. Semua perintah dijalankan dari root repositori dan
memakai `sqlite:///mlflow.db` (ubah dengan `--tracking-uri`). Nama model
default `pm25-karhutla-h6` (ubah dengan `--model-name`), sama dengan nama
yang dipakai `train.py --register-model`.

| Perintah | Fungsi |
|---|---|
| `list` | Semua versi model beserta stage, alias, run, dan `data_version` |
| `register` | Daftarkan artefak model sebuah run (`--run-name` atau `--run-id`) sebagai versi baru |
| `promote` | Ubah stage sebuah versi (`--version`, `--stage`), opsi `--archive-existing` dan `--alias` |
| `metadata` | Tulis metadata silsilah satu versi ke `models/registry/<model>_v<N>.json` |
| `verify` | Muat model lewat `mlflow.pyfunc.load_model` (`--stage`, `--alias`, atau `--version`) dan jalankan prediksi uji |
| `active` | Cetak ringkasan model aktif (default stage Production) sebagai tabel Markdown untuk README |

### Langkah-langkah registry

```bash
# 1. Registrasi model terbaik eksperimen awal (champion Run 03) sebagai versi 1
python src/register_model.py register --experiment pm25-karhutla-h6 \
    --run-name 03-rf-300-d10 \
    --description "Champion eksperimen awal: RF 300 pohon depth 10, data-v3"

# 2. Registrasi model hasil perbaikan (champion Run 18) sebagai versi 2
python src/register_model.py register --experiment pm25-karhutla-h6-v4 \
    --run-name 18-extra-delta-nohotspot \
    --description "Champion v4: ExtraTrees delta tanpa hotspot, 28 fitur, data-v4"
python src/register_model.py list

# 3. Transisi stage: None, Staging, Production
python src/register_model.py promote --version 1 --stage Staging
python src/register_model.py promote --version 1 --stage Production
python src/register_model.py promote --version 2 --stage Staging
python src/register_model.py promote --version 2 --stage Production --archive-existing

# 4. Catat metadata dengan DVC (setelah semua perpindahan stage selesai)
python src/register_model.py metadata --version 1
python src/register_model.py metadata --version 2
dvc add models/registry
git add models/registry.dvc models/.gitignore
git commit -m "feat: catat metadata model registry dengan DVC"
dvc push
git tag -a model-v2 -m "pm25-karhutla-h6 versi 2 (Production)"

# 5. Verifikasi inferensi model Production
python src/register_model.py verify --stage Production
```

Nama run persis (mis. `03-rf-300-d10`) bisa dilihat di MLflow UI atau lewat
`mlflow.search_runs`. Untuk perintah panjang, sambung baris dengan `\` di akhir
baris tanpa spasi setelahnya.

Perpindahan stage juga bisa dilakukan di MLflow UI (`mlflow ui --backend-store-uri
sqlite:///mlflow.db --port 5000`, menu **Models**, pilih versi, dropdown Stage).

### Dokumentasi alur registry

Artefak yang menjadi bukti alur registry:

| Artefak | Sumber |
|---|---|
| Daftar versi dan stage | Tab **Models** di MLflow UI, atau `register_model.py list` |
| Bukti transisi ke Production | Halaman versi sebelum dan sesudah dipromosikan, dan `list` (versi lama berstatus `Archived`) |
| Metadata silsilah | `models/registry.dvc` (pointer DVC) dan `models/registry/*.json` (isi metadata), dihasilkan oleh langkah 4 |
| Bukti inferensi | Keluaran `verify --stage Production` (tabel prediksi dan baris `VERIFIKASI BERHASIL`) |
| Model aktif dan alasannya | Bagian *Model aktif untuk inferensi* di bawah |

Riwayat Git (`git log --oneline`, `git tag`) menunjukkan commit metadata dan tag
`model-v<N>` yang mengaitkan versi model dengan versi data.

### Model aktif untuk inferensi

Bagian ini menjawab model versi berapa yang saat ini dipakai untuk
inferensi dan mengapa. Isinya diperbarui setiap kali versi Production
berubah, dan memuat tiga hal:

1. **Identitas model aktif.** Hasilkan tabelnya dari registry, bukan diketik
   tangan, supaya selalu sesuai kondisi sebenarnya:

       python src/register_model.py active

   Tempel keluarannya (tabel Markdown berisi versi, URI inferensi, run asal,
   versi data, git commit, jumlah fitur, parameter, CV RMSE, baseline musiman,
   dan test RMSE) di bawah judul ini.
2. **Alasan pemilihan.** Tulis singkat berdasarkan angka di tabel. Kriteria
   yang dipakai proyek ini, sama dengan aturan di *Metodologi Evaluasi*:
   - `cv_rmse_mean` terendah pada walk-forward CV; selisih di dalam simpangan
     baku antar fold dianggap setara;
   - mengungguli baseline musiman "jam yang sama kemarin" (`cv_rmse_mean`
     lebih kecil dan `cv_rmse_wins_vs_seasonal` mayoritas fold);
   - `cv_rmse_log_mean` tidak memburuk, agar model tidak hanya unggul pada
     nilai ekstrem;
   - set test hanya dilaporkan, tidak dipakai memilih;
   - idealnya `target-mode` `level`, supaya `pyfunc.predict` langsung mengembalikan
     PM2.5; jika model `delta` yang dipilih, dokumentasikan batasannya;
   - silsilah lengkap: versi data, git commit, dan md5 pointer DVC tercatat.
3. **Batasan penggunaan.** Misalnya skema fitur yang dibutuhkan (daftar fitur
   ada di `models/registry/<model>_v<N>.json`), data latih yang terbatas
   (belum mencakup musim hujan jika datanya masih satu musim), dan bahwa target
   PM2.5 berasal dari model CAMS, bukan sensor darat.

### Model aktif saat ini

Versi 2 (Run 18 `18-extra-delta-nohotspot`, ExtraTrees dengan target `delta`,
28 fitur tanpa hotspot, data-v4) dipromosikan ke Production.

| Item | Nilai |
|---|---|
| Model | pm25-karhutla-h6 |
| Versi aktif | 2 (stage Production) |
| URI inferensi | `models:/pm25-karhutla-h6/Production` |
| Run asal | 18-extra-delta-nohotspot (`59ad0f28`) |
| Versi data | data-v4 |
| Git commit | `15b9374ae9` |
| Jumlah fitur | 28 |
| Target mode | delta |
| Parameter | model=extra, n_estimators=300, max_depth=12, min_samples_leaf=10, max_features=0.5, log_target=False |
| CV RMSE (± std) | 315.3 ± 180.9 |
| CV RMSE baseline musiman | 354.3 |
| Test RMSE | 37.5 |

**Alasan:** Run 18 memiliki CV RMSE terendah (315,3 ± 180,9) pada walk-forward
CV, lebih baik 11% dari baseline musiman (354,3), menang di 4 dari 4 fold, dan
test RMSE terendah (37,5), dengan silsilah data-v4 yang tercatat (versi data,
git commit, dan md5 pointer DVC). Selisih terhadap model teratas lain berada di
dalam simpangan baku antar fold, jadi pilihan ini terbaik menurut aturan
pemilihan proyek tetapi tidak jauh unggul secara statistik. Fitur hotspot tidak
dipakai karena ablation menunjukkan tidak ada kontribusi terukur (lihat *Hasil
Eksperimen Skema v4*).

**Batasan penggunaan:** model dilatih dengan `--target-mode delta`, sehingga
`pyfunc.predict` langsung mengembalikan selisih terhadap PM2.5 saat ini.
`register_model.py verify` mengembalikannya ke skala PM2.5. Sistem inferensi
lain harus menambahkan nilai `pm2_5` saat ini ke keluaran model. Input harus
berisi 28 fitur yang tercantum di `models/registry/pm25-karhutla-h6_v2.json`.

**Versi sebelumnya:** versi 1 (champion eksperimen awal Run 03, data-v3, 14 fitur), berstatus
Archived setelah versi 2 dipromosikan.

Jika Production berpindah versi, perbarui bagian ini dan simpan metadata versi
baru (langkah 4) agar riwayatnya tetap terhubung.

### Catatan

- **Stage dan alias.** Stage (`None`, `Staging`, `Production`, `Archived`) sudah
  ditandai deprecated oleh MLflow dan digantikan alias. Skrip memakai stage
  sesuai alur registry klasik; `--alias` ikut menetapkan alias padanannya (`staging`,
  `production`) dan `verify --alias production` memuat model lewat alias.
  Jika stage tidak muncul di UI versi MLflow kamu, gunakan alias.
- **Silsilah.** Metadata berisi `data_version`, md5 pointer `data/raw.dvc` dan
  `data/processed.dvc`, `git_commit`, daftar fitur, parameter, dan metrik run
  asal model. Rantainya: versi model, run MLflow, tag data, pointer DVC, data di
  Nextcloud. Commit `models/registry.dvc` bersama tag `model-vN` mengaitkan
  versi model dengan versi data pada riwayat Git.
- **Skema fitur tiap versi.** Fitur yang dipakai versi model dicatat di param
  `features` pada run-nya. Versi 1 (data-v3, 14 fitur) dan versi 2 (data-v4,
  28 fitur, tanpa hotspot) berbeda. `verify` membaca daftar fitur versi yang diuji, sehingga
  tiap versi mendapat inputnya sendiri dari `data/processed`. Pemanggil lain
  harus menyesuaikan input saat Production berpindah versi.
- **Target-mode.** Model yang dilatih dengan `--target-mode delta` atau
  `log-ratio` memprediksi nilai transformasi. `verify` mengembalikannya ke skala
  PM2.5 memakai `restore_level()`, tetapi pemanggil langsung `pyfunc.predict`
  akan menerima nilai transformasi. Untuk sistem inferensi yang memanggil model
  langsung, `--target-mode level` (default) paling sederhana; model `delta`
  memerlukan penambahan `pm2_5` saat ini ke keluarannya.
- Run eksperimen awal (Run 01-09) ada di `mlflow.db` dan `mlruns/` tempat training
  dilakukan. Langkah 1 hanya bisa dijalankan di lingkungan yang memiliki keduanya.

## Alur Lengkap

Urutan dari data mentah sampai model, dijalankan dari root repositori.

### 1. Menambah data dan membangun dataset

```bash
dvc pull data/raw.dvc                  # wajib dulu, agar raw lama ikut terlacak

python src/ingest_data.py              # data terbaru (atau --mode historical, lihat di atas)
rm -rf data/interim data/processed     # buang hasil lama agar tidak tercampur skema lama
python src/preprocess.py
python src/build_features.py

dvc add data/raw data/processed
dvc push
git add data/raw.dvc data/processed.dvc
git commit -m "feat: perbarui dataset"
git tag -a data-v4 -m "keterangan"     # MAJOR jika skema berubah, MINOR jika hanya data baru
git push && git push origin data-v4
```

`data/interim` dan `data/processed` boleh dihapus dengan aman: keduanya bisa
dibangun ulang dari raw, versi lama `data/processed` tetap tersimpan di remote
DVC dan bisa dipulihkan lewat `git checkout data-v3 && dvc checkout`. Jangan
menghapus `data/raw`. `build_features.py` tidak membersihkan folder keluaran,
jadi partisi bulan skema lama yang tidak ikut dibangun ulang akan tercampur
dengan skema baru dan membuat training gagal atau diam-diam membuang baris.

### 2. Training dan registry

Lihat *Opsi training* dan *Cara mereproduksi* di bagian eksperimen, dan bagian *Model Registry* untuk mendaftarkan serta mempromosikan model.

### 3. Menjalankan workflow GitHub Actions

Workflow hanya muncul di tab **Actions** dan berjalan terjadwal setelah
file-nya ada di branch `main`.

1. Daftarkan secret dan atur permission (lihat *Setup GitHub Secrets*).
2. Actions, `ETL Pipeline (Ingestion)`, `Run workflow`. Cek bahwa muncul commit
   `chore(data): ingestion otomatis ...`.
3. Actions, `ETL Pipeline (Transform and Load)`, `Run workflow`. Kosongkan
   `create_tag` untuk uji pertama.

Dengan GitHub CLI:

```bash
gh workflow run "ETL Pipeline (Ingestion)"
gh workflow run "ETL Pipeline (Transform and Load)" -f create_tag=false
gh run list --limit 5
gh run watch
```

Urutan mengaktifkan workflow: selesaikan dulu langkah 1 (termasuk backfill
historis dan tag `data-v4`) lalu push, baru gabungkan workflow ke `main`.
Setelah aktif, workflow meng-commit `data/raw.dvc` ke `main` tiap 3 jam,
sehingga pointer `raw.dvc` yang di-commit dari lokal akan bentrok dan harus
di-`git pull --rebase` dulu.

## Standar Kode (PEP 8)

Kode mengikuti PEP 8: nama variabel deskriptif, docstring pada setiap modul dan fungsi, dan maksimal 79 karakter per baris. Konfigurasi ada di `pyproject.toml` (black) dan `.flake8`.

```bash
black src/           # format otomatis
flake8 src/          # cek kepatuhan PEP 8
```

## Branching Strategy

Proyek ini mengikuti **GitHub Flow**:
- Branch `main` selalu dalam kondisi stabil
- Setiap eksperimen/fitur baru dikerjakan di branch terpisah dengan prefix sesuai jenis perubahan: `feat/` (contoh: `feat/feature-engineering`, `feat/model-training`), `chore/` (contoh: `chore/pin-requirements-versions`), atau `docs/`
- Merge ke `main` dilakukan melalui Pull Request setelah validasi (review kode, cek hasil eksperimen, pastikan tidak ada credential ter-commit)

## Tech Stack

| Kategori | Tools |
|---|---|
| Version Control & Collaboration | GitHub |
| Cloud Development | GitHub Codespaces |
| Data Versioning | DVC |
| CI/CD & Automation | GitHub Actions |
| Experiment Tracking & Registry | MLflow |
| Containerization | Docker, Docker Compose (rencana, belum diimplementasi) |
| Model Serving | MLflow Models (rencana, belum diimplementasi) |
| Monitoring | Prometheus, Grafana (rencana, belum diimplementasi) |

## Sumber Data

| Sumber | Kegunaan | Akses |
|---|---|---|
| [Open-Meteo Weather API](https://open-meteo.com/) | Data cuaca (suhu, kelembapan, angin, curah hujan) | Gratis, tanpa API key |
| [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) | Data PM2.5, PM10, AQI | Gratis, tanpa API key |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | Data titik api satelit (leading indicator) | Gratis, perlu MAP_KEY |
