"""Tes mlflow_store: sinkronisasi ke WebDAV palsu dan prune artefak."""

import glob
import os
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import mlflow
import pytest
from mlflow import MlflowClient

import mlflow_store as ms
import train


def make_server():
    """Server WebDAV mini dengan ETag, If-Match, dan If-None-Match."""
    state = {"files": {}, "collections": set(), "counter": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, etag=None, body=b""):
            self.send_response(code)
            if etag:
                self.send_header("ETag", etag)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in state["files"]:
                data, etag = state["files"][self.path]
                self.reply(200, etag, data)
            else:
                self.reply(404)

        def do_PUT(self):
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            current = state["files"].get(self.path)
            if_match = self.headers.get("If-Match")
            if if_match and (current is None or current[1] != if_match):
                return self.reply(412)
            if self.headers.get("If-None-Match") == "*" and current:
                return self.reply(412)
            state["counter"] += 1
            etag = f'"v{state["counter"]}"'
            state["files"][self.path] = (data, etag)
            self.reply(201 if current is None else 204, etag)

        def do_MKCOL(self):
            if self.path in state["collections"]:
                return self.reply(405)
            state["collections"].add(self.path)
            self.reply(201)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


@pytest.fixture
def webdav():
    server, state = make_server()
    base = f"http://127.0.0.1:{server.server_port}/dav/mlflow_store"
    yield base, state
    server.shutdown()
    server.server_close()


def session():
    return ms.build_session("user", "secret")


def fake_store(folder, text="v1"):
    """Isi minimal store: file database dan satu artefak."""
    (folder / ms.STORE_DB).write_text(text, encoding="utf-8")
    artifacts = folder / ms.STORE_DIR / "1" / "run"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "metrics.txt").write_text(text, encoding="utf-8")


class TestSync:
    def test_first_run_has_no_store(self, webdav, tmp_path):
        base, _ = webdav
        assert ms.pull(base, session(), str(tmp_path)) is False
        assert (tmp_path / ms.ETAG_FILE).read_text() == ""

    def test_push_then_pull_roundtrip(self, webdav, tmp_path):
        base, _ = webdav
        source, target = tmp_path / "a", tmp_path / "b"
        source.mkdir()
        target.mkdir()
        fake_store(source, "isi-pertama")
        ms.pull(base, session(), str(source))
        ms.push(base, session(), str(source))

        assert ms.pull(base, session(), str(target)) is True
        assert (target / ms.STORE_DB).read_text() == "isi-pertama"
        copied = target / ms.STORE_DIR / "1" / "run" / "metrics.txt"
        assert copied.read_text() == "isi-pertama"

    def test_archive_contains_only_store_files(self, webdav, tmp_path):
        base, state = webdav
        fake_store(tmp_path)
        (tmp_path / "bukan-store.txt").write_text("x")
        ms.pull(base, session(), str(tmp_path))
        ms.push(base, session(), str(tmp_path))
        data, _ = state["files"]["/dav/mlflow_store/" + ms.ARCHIVE_NAME]
        archive = tmp_path / "out.tar.gz"
        archive.write_bytes(data)
        with tarfile.open(archive) as handle:
            names = handle.getnames()
        assert all(n.startswith((ms.STORE_DB, ms.STORE_DIR)) for n in names)
        assert not any("bukan-store" in n or ms.ETAG_FILE in n for n in names)

    def test_concurrent_change_is_rejected(self, webdav, tmp_path):
        base, _ = webdav
        mine, other = tmp_path / "mine", tmp_path / "other"
        mine.mkdir()
        other.mkdir()
        fake_store(mine, "awal")
        ms.pull(base, session(), str(mine))
        ms.push(base, session(), str(mine))

        ms.pull(base, session(), str(mine))
        ms.pull(base, session(), str(other))
        fake_store(other, "pihak-lain")
        ms.push(base, session(), str(other))

        fake_store(mine, "perubahan-saya")
        with pytest.raises(ms.StoreConflictError):
            ms.push(base, session(), str(mine))

    def test_first_push_race_is_rejected(self, webdav, tmp_path):
        base, _ = webdav
        first, second = tmp_path / "first", tmp_path / "second"
        first.mkdir()
        second.mkdir()
        fake_store(first)
        fake_store(second)
        ms.pull(base, session(), str(first))
        ms.pull(base, session(), str(second))
        ms.push(base, session(), str(first))
        with pytest.raises(ms.StoreConflictError):
            ms.push(base, session(), str(second))

    def test_push_without_store_files_fails(self, webdav, tmp_path):
        base, _ = webdav
        with pytest.raises(FileNotFoundError):
            ms.push(base, session(), str(tmp_path))

    def test_missing_credentials_returns_error(self, monkeypatch, capsys):
        monkeypatch.delenv("NEXTCLOUD_USER", raising=False)
        monkeypatch.delenv("NEXTCLOUD_APP_PASSWORD", raising=False)
        code = ms.main(["--url", "http://x/y", "pull"])
        assert code == 1
        assert "NEXTCLOUD_USER" in capsys.readouterr().out

    def test_main_pull_and_push_with_env(
        self, webdav, tmp_path, monkeypatch, capsys
    ):
        base, _ = webdav
        monkeypatch.setenv("NEXTCLOUD_USER", "user")
        monkeypatch.setenv("NEXTCLOUD_APP_PASSWORD", "secret")
        fake_store(tmp_path)
        common = ["--url", base, "--workdir", str(tmp_path)]
        assert ms.main([*common, "pull"]) == 0
        assert ms.main([*common, "push"]) == 0
        output = capsys.readouterr().out
        assert "secret" not in output


class TestPrune:
    def make_runs(self, processed_dir, folder, monkeypatch):
        """Tiga run kecil; run pertama didaftarkan ke registry."""
        monkeypatch.chdir(folder)
        uri = f"sqlite:///{folder}/{ms.STORE_DB}"
        for name in ("r1", "r2", "r3"):
            args = [
                "--data", processed_dir,
                "--tracking-uri", uri,
                "--experiment", "pm25-karhutla-h6-ci",
                "--run-name", name,
                "--data-version", "test",
                "--n-estimators", "5",
                "--max-depth", "3",
            ]  # fmt: skip
            if name == "r1":
                args.append("--register-model")
            assert train.main(args) == 0
        return uri

    def model_dirs(self, folder):
        pattern = os.path.join(folder, ms.STORE_DIR, "*", "models", "m-*")
        return {os.path.basename(p) for p in glob.glob(pattern)}

    def test_prune_keeps_registered_and_recent(
        self, processed_dir, tmp_path, monkeypatch
    ):
        uri = self.make_runs(processed_dir, tmp_path, monkeypatch)
        before = self.model_dirs(tmp_path)
        assert len(before) == 3

        removed, freed = ms.prune(str(tmp_path), keep_recent=1)
        after = self.model_dirs(tmp_path)

        assert removed == 1
        assert freed > 0
        assert len(after) == 2

        mlflow.set_tracking_uri(uri)
        client = MlflowClient()
        registered = client.get_model_version("pm25-karhutla-h6", "1")
        registered_id = registered.source.rsplit("/", 1)[-1]
        assert registered_id in after
        # Model terdaftar tetap bisa dimuat setelah prune.
        loaded = mlflow.pyfunc.load_model("models:/pm25-karhutla-h6/1")
        assert loaded is not None

    def test_prune_with_large_keep_removes_nothing(
        self, processed_dir, tmp_path, monkeypatch
    ):
        self.make_runs(processed_dir, tmp_path, monkeypatch)
        removed, _ = ms.prune(str(tmp_path), keep_recent=10)
        assert removed == 0
        assert len(self.model_dirs(tmp_path)) == 3
