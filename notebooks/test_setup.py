import requests


def test_open_meteo_connection():
    """Cek koneksi ke Open-Meteo Weather API"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": -2.21,
        "longitude": 113.92,
        "hourly": "temperature_2m",
        "timezone": "Asia/Jakarta",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    print("Koneksi ke Open-Meteo Weather API: BERHASIL")
    print(f"Lokasi: lat={data['latitude']}, lon={data['longitude']}")
    print(f"Jumlah data jam tersedia: {len(data['hourly']['time'])}")
    return True


def test_air_quality_connection():
    """Cek koneksi ke Open-Meteo Air Quality API"""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": -2.21,
        "longitude": 113.92,
        "hourly": "pm2_5",
        "timezone": "Asia/Jakarta",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    print("\nKoneksi ke Open-Meteo Air Quality API: BERHASIL")
    print(f"Contoh nilai PM2.5 pertama: {data['hourly']['pm2_5'][0]} µg/m³")
    return True


if __name__ == "__main__":
    print("-" * 50)
    print("TEST SETUP - branch feat/initial-eda")
    print("-" * 50)

    test_open_meteo_connection()
    test_air_quality_connection()

    print("\n" + "-" * 50)
    print("Semua tes berhasil. Environment siap untuk EDA lanjutan.")
    print("-" * 50)
