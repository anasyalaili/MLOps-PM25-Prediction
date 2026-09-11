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
├── config/
├── data/
├── docs/
├── models/
├── notebooks/
├── src/
│   ├── fetch/
│   ├── features/
│   ├── training/
│   └── serving/
├── tests/
├── .env
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── requirements.txt
├── LICENSE
└── README.md
```

## Cara Menjalankan Via Codespaces
Proyek ini menggunakan GitHub Codespaces sebagai cloud development environment yang sudah dikonfigurasi otomatis lewat .devcontainer/devcontainer.json dengan dependency dan extension VS Code yang akan terinstall otomatis tanpa setup manual.

1. Buka repository ini di GitHub
2. Klik tombol Code (hijau) di kanan atas
3. Pilih tab Codespaces, lalu klik Create codespace on main
4. Tunggu beberapa saat lalu Codespace akan otomatis:
   - Menyiapkan environment Python 3.11
   - Menginstall semua dependency dari requirements.txt
   - Menginstall extension VS Code yang dibutuhkan (Python, Jupyter, Docker, GitHub Actions)
5. Setelah selesai, langsung bisa menjalankan script tanpa instalasi tambahan:
   python notebooks/test_setup.py
