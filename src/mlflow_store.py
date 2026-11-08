r"""Sinkronisasi store MLflow untuk CI dengan Nextcloud (WebDAV).

Runner GitHub Actions bersifat sementara, sedangkan registry model harus
bertahan antar run. Store MLflow khusus CI (database SQLite mlflow-ci.db dan
folder artefak mlruns/) karena itu disimpan sebagai satu arsip tar.gz di
Nextcloud: diunduh di awal job dan diunggah di akhir job.

Pengaman terhadap saling menimpa: saat mengunduh, ETag arsip disimpan; saat
mengunggah, ETag itu dikirim sebagai If-Match. Jika arsip sudah diubah pihak
lain, unggahan ditolak (HTTP 412) dan perintah gagal, bukan menimpa diam-diam.
Untuk run pertama (arsip belum ada), unggahan memakai If-None-Match: *.

Subperintah:
    pull    unduh arsip dan ekstrak ke direktori kerja.
    push    kemas mlflow-ci.db dan mlruns/ lalu unggah.
    prune   hapus artefak model yang tidak terdaftar di registry dan bukan
            bagian dari beberapa model terbaru (arsip tidak membengkak).

Kredensial dibaca dari environment, tidak pernah dicetak:
    NEXTCLOUD_USER, NEXTCLOUD_APP_PASSWORD
Alamat folder penyimpanan:
    MLFLOW_STORE_URL (atau --url), mis.
    https://cloud.contoh.id/remote.php/dav/files/user/mlflow_store

Contoh:
    python src/mlflow_store.py pull
    python src/mlflow_store.py prune --keep-recent 2
    python src/mlflow_store.py push
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile

import mlflow
import requests
from mlflow import MlflowClient
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

STORE_DB = "mlflow-ci.db"
STORE_DIR = "mlruns"
ARCHIVE_NAME = "mlflow-ci-store.tar.gz"
ETAG_FILE = ".mlflow_store_etag"
REQUEST_TIMEOUT_SECONDS = 300
HTTP_RETRIES = 3


class StoreConflictError(Exception):
    """Dilempar saat arsip di Nextcloud berubah sejak diunduh."""


def parse_arguments(argv=None):
    """Baca argumen command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", default=os.getenv("MLFLOW_STORE_URL"))
    parser.add_argument("--workdir", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pull", help="Unduh dan ekstrak store.")
    sub.add_parser("push", help="Kemas dan unggah store.")
    prune_parser = sub.add_parser(
        "prune", help="Hapus artefak model yang tidak terpakai."
    )
    prune_parser.add_argument("--keep-recent", type=int, default=2)
    return parser.parse_args(argv)


def build_session(user, password):
    """Sesi HTTP dengan autentikasi dasar dan retry untuk error 5xx."""
    session = requests.Session()
    session.auth = (user, password)
    retry = Retry(
        total=HTTP_RETRIES,
        backoff_factor=2,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset({"GET", "PUT", "MKCOL", "HEAD"}),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def archive_url(base_url):
    """URL arsip di dalam folder penyimpanan."""
    return f"{base_url.rstrip('/')}/{ARCHIVE_NAME}"


def pull(base_url, session, workdir="."):
    """Unduh arsip dan ekstrak ke workdir.

    Returns:
        bool: True jika arsip ada, False jika ini run pertama.
    """
    etag_path = os.path.join(workdir, ETAG_FILE)
    response = session.get(
        archive_url(base_url), stream=True, timeout=REQUEST_TIMEOUT_SECONDS
    )
    if response.status_code == 404:
        with open(etag_path, "w", encoding="utf-8") as handle:
            handle.write("")
        print("Store belum ada di Nextcloud (run pertama), mulai dari kosong.")
        return False
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        for chunk in response.iter_content(chunk_size=1 << 20):
            tmp.write(chunk)
        tmp_path = tmp.name
    try:
        with tarfile.open(tmp_path, "r:gz") as archive:
            archive.extractall(workdir, filter="data")
    finally:
        os.remove(tmp_path)
    with open(etag_path, "w", encoding="utf-8") as handle:
        handle.write(response.headers.get("ETag", ""))
    print(f"Store diunduh dan diekstrak ke '{workdir}'.")
    return True


def make_archive(workdir, target_path):
    """Kemas database dan folder artefak ke target_path (tar.gz)."""
    members = [STORE_DB, STORE_DIR]
    present = [m for m in members if os.path.exists(os.path.join(workdir, m))]
    if not present:
        raise FileNotFoundError(
            f"Tidak ada {STORE_DB} atau {STORE_DIR}/ di '{workdir}'."
        )
    with tarfile.open(target_path, "w:gz") as archive:
        for member in present:
            archive.add(os.path.join(workdir, member), arcname=member)
    return present


def ensure_collection(base_url, session):
    """Pastikan folder penyimpanan ada (MKCOL); sudah ada bukan masalah."""
    response = session.request(
        "MKCOL", base_url.rstrip("/"), timeout=REQUEST_TIMEOUT_SECONDS
    )
    if response.status_code not in (201, 405):
        response.raise_for_status()


def push(base_url, session, workdir="."):
    """Kemas dan unggah store dengan pengaman ETag.

    Raises:
        StoreConflictError: Jika arsip di Nextcloud berubah sejak diunduh.
    """
    etag_path = os.path.join(workdir, ETAG_FILE)
    etag = ""
    if os.path.exists(etag_path):
        with open(etag_path, encoding="utf-8") as handle:
            etag = handle.read().strip()
    ensure_collection(base_url, session)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        make_archive(workdir, tmp_path)
        headers = {"If-Match": etag} if etag else {"If-None-Match": "*"}
        with open(tmp_path, "rb") as payload:
            response = session.put(
                archive_url(base_url),
                data=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        size_mb = os.path.getsize(tmp_path) / (1 << 20)
    finally:
        os.remove(tmp_path)
    if response.status_code == 412:
        raise StoreConflictError(
            "Store di Nextcloud berubah sejak diunduh (HTTP 412). Unggahan "
            "dibatalkan agar tidak menimpa perubahan pihak lain; jalankan "
            "ulang dari pull."
        )
    response.raise_for_status()
    new_etag = response.headers.get("ETag", "")
    if new_etag:
        with open(etag_path, "w", encoding="utf-8") as handle:
            handle.write(new_etag)
    print(f"Store diunggah ({size_mb:.1f} MB).")


def local_path(location):
    """Ubah lokasi artefak (path atau URI file) menjadi path lokal."""
    return (
        location[len("file://") :]
        if location.startswith("file://")
        else location
    )


def registered_model_ids(client):
    """ID logged model yang dipakai oleh versi model terdaftar."""
    ids = set()
    for model in client.search_registered_models():
        versions = client.search_model_versions(f"name = '{model.name}'")
        for version in versions:
            if version.source.startswith("models:/m-"):
                ids.add(version.source.rsplit("/", 1)[-1])
    return ids


def prune(workdir=".", keep_recent=2):
    """Hapus artefak logged model yang tidak terdaftar dan sudah lama.

    Model terdaftar (versi registry) selalu dipertahankan, begitu pula
    keep_recent model terbaru (untuk gate evaluasi dan registrasi).

    Returns:
        tuple: (jumlah model dihapus, ukuran dibebaskan dalam MB).
    """
    db_path = os.path.abspath(os.path.join(workdir, STORE_DB))
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    client = MlflowClient()
    experiment_ids = [e.experiment_id for e in client.search_experiments()]
    models = mlflow.search_logged_models(
        experiment_ids=experiment_ids, output_format="list"
    )
    models = sorted(models, key=lambda m: m.creation_timestamp, reverse=True)
    protected = registered_model_ids(client) | {
        m.model_id for m in models[:keep_recent]
    }
    store_root = os.path.abspath(os.path.join(workdir, STORE_DIR))
    removed, freed = 0, 0
    for model in models:
        if model.model_id in protected:
            continue
        folder = os.path.dirname(local_path(model.artifact_location))
        folder = os.path.abspath(folder)
        if not folder.startswith(store_root + os.sep) or not os.path.isdir(
            folder
        ):
            continue
        freed += sum(
            os.path.getsize(os.path.join(root, name))
            for root, _, names in os.walk(folder)
            for name in names
        )
        shutil.rmtree(folder)
        removed += 1
    freed_mb = freed / (1 << 20)
    print(f"Prune: {removed} model dihapus, {freed_mb:.1f} MB dibebaskan.")
    return removed, freed_mb


def main(argv=None):
    """Titik masuk skrip; return 0 jika sukses."""
    args = parse_arguments(argv)
    try:
        if args.command == "prune":
            prune(args.workdir, args.keep_recent)
            return 0
        user = os.getenv("NEXTCLOUD_USER")
        password = os.getenv("NEXTCLOUD_APP_PASSWORD")
        if not (args.url and user and password):
            print(
                "ERROR: butuh --url (atau MLFLOW_STORE_URL), NEXTCLOUD_USER, "
                "dan NEXTCLOUD_APP_PASSWORD."
            )
            return 1
        session = build_session(user, password)
        if args.command == "pull":
            pull(args.url, session, args.workdir)
        else:
            push(args.url, session, args.workdir)
    except (
        StoreConflictError,
        FileNotFoundError,
        requests.exceptions.RequestException,
    ) as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
