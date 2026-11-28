r"""Simulasi beban pada API model untuk mengisi dashboard Grafana.

Skenario:
    normal  baris acak dari seluruh dataset (mirip data training).
    drift   baris dari 10% jam dengan PM2.5 tertinggi (episode asap).

Bila dataset tidak tersedia, dipakai nilai contoh yang diacak (PM2.5 untuk
skenario drift dinaikkan).

Contoh:
    python src/load_test.py --scenario normal --duration 120
    python src/load_test.py --scenario drift --duration 120 --rps 10
"""

import argparse
import collections
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from package_model import example_row
from train import CURRENT_COLUMN, DEFAULT_DATA_PATH, FEATURES, load_dataset

DEFAULT_URL = "http://localhost:8000/invocations"
DRIFT_QUANTILE = 0.9


def parse_arguments(argv=None):
    """Baca argumen command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--scenario", choices=("normal", "drift"), default="normal"
    )
    parser.add_argument("--duration", type=float, default=60, help="Detik.")
    parser.add_argument("--rps", type=float, default=5)
    parser.add_argument(
        "--bad-ratio",
        type=float,
        default=0.0,
        help="Porsi request sengaja salah (balasan 400).",
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def load_pool(path, scenario, seed):
    """Kumpulan baris fitur yang akan dikirim, beserta asalnya."""
    try:
        data = load_dataset(path)
    except FileNotFoundError:
        return synthetic_pool(scenario, seed), "contoh acak"
    data = data[FEATURES].dropna()
    if scenario == "drift":
        limit = data[CURRENT_COLUMN].quantile(DRIFT_QUANTILE)
        data = data[data[CURRENT_COLUMN] >= limit]
    return data.reset_index(drop=True), "dataset"


def synthetic_pool(scenario, seed, rows=500):
    """Baris contoh acak bila dataset tidak ada."""
    rng = np.random.default_rng(seed)
    base = pd.DataFrame([example_row(FEATURES)] * rows)
    pool = base * rng.uniform(0.7, 1.3, size=base.shape)
    if scenario == "drift":
        pool[CURRENT_COLUMN] *= 5
    return pool


def send(session, url, body):
    """Kirim satu request, kembalikan status HTTP atau 'gagal'."""
    try:
        return session.post(url, json=body, timeout=10).status_code
    except requests.RequestException:
        return "gagal"


def run(args):
    """Kirim request dengan laju tetap selama durasi yang diminta."""
    pool, source = load_pool(args.data, args.scenario, args.seed)
    rng = np.random.default_rng(args.seed)
    total = max(1, int(args.duration * args.rps))
    print(
        f"Skenario {args.scenario}: {total} request, {args.rps}/detik, "
        f"sumber {source} ({len(pool)} baris)."
    )
    futures = []
    start = time.monotonic()
    with requests.Session() as session, ThreadPoolExecutor(16) as workers:
        for number in range(total):
            row = pool.iloc[int(rng.integers(len(pool)))].to_dict()
            if rng.random() < args.bad_ratio:
                row = {CURRENT_COLUMN: row[CURRENT_COLUMN]}
            body = {"dataframe_records": [row]}
            futures.append(workers.submit(send, session, args.url, body))
            delay = start + (number + 1) / args.rps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        counts = collections.Counter(f.result() for f in futures)
    print(f"Selesai. Status: {dict(counts)}")
    return counts


def main(argv=None):
    """Titik masuk CLI; 1 bila tidak satu pun request berhasil."""
    return 0 if run(parse_arguments(argv))[200] else 1


if __name__ == "__main__":
    raise SystemExit(main())
