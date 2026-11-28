r"""Tambahkan endpoint /metrics (Prometheus) pada server model MLflow.

Membungkus aplikasi scoring MLflow dengan middleware ASGI, sehingga /ping dan
/invocations tetap sama persis, ditambah:
    - jumlah dan latensi request /invocations,
    - sebaran input PM2.5 dan prediksi PM2.5 H+6,
    - PSI input PM2.5 pada jendela request terakhir (indikasi data drift),
    - metrik uji model saat training dan CPU/RAM proses.

Dijalankan di dalam image hasil `mlflow models build-docker`:
    uvicorn metrics_app:build --factory --host 0.0.0.0 --port 8080

Metrik disimpan per proses, jadi pakai satu worker per replika.

Variabel lingkungan:
    MODEL_PATH   folder model di dalam image (default /opt/ml/model)
    PSI_WINDOW   jumlah input terakhir untuk menghitung PSI (default 500)
"""

import json
import math
import os
import threading
import time
from bisect import bisect_right
from collections import deque

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

FEATURE = "pm2_5"
PSI_MIN_SAMPLES = 100
PSI_EPSILON = 1e-4
CONCENTRATION_BUCKETS = (15, 35, 55, 100, 150.5, 250.5, 500, 1000, 2000)
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5)
CGROUP_MEMORY_FILES = (
    "/sys/fs/cgroup/memory.current",
    "/sys/fs/cgroup/memory/memory.usage_in_bytes",
)


def population_stability_index(expected, values, edges):
    """PSI antara proporsi baseline per bin dan nilai yang diamati."""
    counts = [0] * len(expected)
    for value in values:
        counts[bisect_right(edges, value)] += 1
    psi = 0.0
    for share, count in zip(expected, counts):
        base = max(share, PSI_EPSILON)
        seen = max(count / len(values), PSI_EPSILON)
        psi += (seen - base) * math.log(seen / base)
    return psi


def load_monitoring(model_path):
    """Baca monitoring.json dari artefak model; {} bila tidak ada."""
    for root, _, files in os.walk(model_path):
        if "monitoring.json" in files:
            with open(os.path.join(root, "monitoring.json")) as handle:
                return json.load(handle)
    return {}


def input_rows(body):
    """Baris input dari format dataframe_records atau dataframe_split."""
    if "dataframe_records" in body:
        return body["dataframe_records"]
    split = body.get("dataframe_split", {})
    columns = split.get("columns", [])
    return [dict(zip(columns, row)) for row in split.get("data", [])]


def memory_bytes(files=CGROUP_MEMORY_FILES):
    """Pemakaian RAM kontainer dari cgroup (v2 lalu v1); None bila tiada.

    Angka cgroup tetap benar saat image amd64 diemulasi, sedangkan RSS dari
    /proc tidak.
    """
    for path in files:
        try:
            with open(path) as handle:
                return int(handle.read())
        except (OSError, ValueError):
            continue
    return None


class ResourceCollector:
    """CPU dan RAM proses.

    Pengganti metrik process_* bawaan, yang bernilai nol saat image amd64
    diemulasi (mis. Mac ARM) karena membaca /proc/self/stat.
    """

    def collect(self):
        used = os.times()
        yield CounterMetricFamily(
            "pm25_process_cpu_seconds",
            "Waktu CPU proses (user + system)",
            value=used.user + used.system,
        )
        memory = memory_bytes()
        if memory is not None:
            yield GaugeMetricFamily(
                "pm25_container_memory_bytes",
                "RAM yang dipakai kontainer",
                value=memory,
            )


class Monitor:
    """Kumpulan metrik Prometheus untuk satu replika."""

    def __init__(self, monitoring, registry=REGISTRY, window=500):
        self.registry = registry
        registry.register(ResourceCollector())
        self.baseline = monitoring.get("baseline", {}).get(FEATURE)
        self.window = deque(maxlen=window)
        self.lock = threading.Lock()
        self.requests = Counter(
            "pm25_requests",
            "Request /invocations per status HTTP",
            ["status"],
            registry=registry,
        )
        self.latency = Histogram(
            "pm25_request_duration_seconds",
            "Latensi request /invocations",
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )
        self.inputs = Histogram(
            "pm25_input_ugm3",
            "Input PM2.5 saat ini (ug/m3)",
            buckets=CONCENTRATION_BUCKETS,
            registry=registry,
        )
        self.predictions = Histogram(
            "pm25_prediction_ugm3",
            "Prediksi PM2.5 H+6 (ug/m3)",
            buckets=CONCENTRATION_BUCKETS,
            registry=registry,
        )
        self.samples = Gauge(
            "pm25_drift_window_samples",
            "Jumlah input di jendela PSI",
            registry=registry,
        )
        # PSI baru diekspos setelah sampel cukup, agar 0 tidak menyesatkan.
        self.psi = Gauge(
            "pm25_drift_psi",
            "PSI input terhadap data training",
            ["feature"],
            registry=registry,
        )
        offline = Gauge(
            "pm25_model_offline_metric",
            "Metrik model yang diukur saat training",
            ["metric"],
            registry=registry,
        )
        for name, value in monitoring.get("offline_metrics", {}).items():
            offline.labels(metric=name).set(value)
        info = Gauge(
            "pm25_model_info",
            "Identitas model yang dilayani",
            ["base_version", "data_version"],
            registry=registry,
        )
        info.labels(
            base_version=str(monitoring.get("base_version", "")),
            data_version=str(monitoring.get("data_version", "")),
        ).set(1)

    def observe(self, status, seconds, request_body, response_body):
        """Catat satu request; isi hanya dibaca bila berhasil."""
        self.requests.labels(status=str(status)).inc()
        self.latency.observe(seconds)
        if status != 200:
            return
        for value in json.loads(response_body)["predictions"]:
            self.predictions.observe(float(value))
        values = [
            float(row[FEATURE])
            for row in input_rows(json.loads(request_body))
            if row.get(FEATURE) is not None
        ]
        with self.lock:
            for value in values:
                self.inputs.observe(value)
                self.window.append(value)
            self.samples.set(len(self.window))
            if self.baseline and len(self.window) >= PSI_MIN_SAMPLES:
                self.psi.labels(feature=FEATURE).set(
                    population_stability_index(
                        self.baseline["expected"],
                        self.window,
                        self.baseline["edges"],
                    )
                )


def instrument(inner, monitor):
    """Bungkus aplikasi ASGI: layani /metrics dan ukur /invocations."""

    async def app(scope, receive, send):
        if scope["type"] != "http":
            await inner(scope, receive, send)
            return
        if scope["path"] == "/metrics":
            body = generate_latest(monitor.registry)
            headers = [(b"content-type", CONTENT_TYPE_LATEST.encode())]
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": headers,
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        if scope["path"] != "/invocations":
            await inner(scope, receive, send)
            return

        request, response = bytearray(), bytearray()
        status = [500]

        async def read():
            message = await receive()
            request.extend(message.get("body", b""))
            return message

        async def write(message):
            if message["type"] == "http.response.start":
                status[0] = message["status"]
            else:
                response.extend(message.get("body", b""))
            await send(message)

        start = time.perf_counter()
        try:
            await inner(scope, read, write)
        finally:
            elapsed = time.perf_counter() - start
            try:
                monitor.observe(
                    status[0], elapsed, bytes(request), bytes(response)
                )
            except Exception as exc:  # pemantauan tidak boleh ganggu API
                print(f"Gagal mencatat metrik: {exc}")

    return app


def build():
    """Factory uvicorn: muat model MLflow lalu pasang pemantauan."""
    from mlflow.pyfunc import scoring_server

    model_path = os.getenv("MODEL_PATH", "/opt/ml/model")
    inner = scoring_server.init(
        scoring_server.load_model_with_mlflow_config(model_path)
    )
    window = int(os.getenv("PSI_WINDOW", "500"))
    return instrument(
        inner, Monitor(load_monitoring(model_path), window=window)
    )
