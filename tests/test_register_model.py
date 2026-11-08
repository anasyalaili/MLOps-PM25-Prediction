"""Tes register_model: siklus registry penuh pada store MLflow sementara."""

import json

import mlflow
import pytest
from mlflow import MlflowClient

import register_model as rm
import train

NAME = "pm25-karhutla-h6"


@pytest.fixture(scope="module")
def store(processed_dir, tmp_path_factory):
    """Store MLflow sementara berisi dua versi model terdaftar."""
    folder = tmp_path_factory.mktemp("registry")
    uri = f"sqlite:///{folder}/mlflow.db"
    # Artefak MLflow ditulis relatif terhadap direktori kerja.
    previous = __import__("os").getcwd()
    __import__("os").chdir(folder)
    try:
        for run_name, depth in (("r1", "3"), ("r2", "5")):
            code = train.main(
                [
                    "--data", processed_dir,
                    "--tracking-uri", uri,
                    "--run-name", run_name,
                    "--data-version", "test-data",
                    "--n-estimators", "5",
                    "--max-depth", depth,
                    "--cv-folds", "2",
                    "--register-model",
                ]
            )  # fmt: skip
            assert code == 0
        yield {"uri": uri, "folder": folder}
    finally:
        __import__("os").chdir(previous)


def run_cli(store, *args):
    return rm.main(["--tracking-uri", store["uri"], *args])


class TestArguments:
    def test_subcommand_required(self):
        with pytest.raises(SystemExit):
            rm.parse_arguments([])

    def test_promote_rejects_unknown_stage(self):
        with pytest.raises(SystemExit):
            rm.parse_arguments(
                ["promote", "--version", "1", "--stage", "Nope"]
            )

    def test_version_selector_must_be_exactly_one(self, store):
        mlflow.set_tracking_uri(store["uri"])
        args = rm.parse_arguments(
            ["verify", "--stage", "Production", "--version", "1"]
        )
        with pytest.raises(ValueError, match="satu"):
            rm.resolve_model_uri(MlflowClient(), args)


class TestLifecycle:
    def test_two_versions_registered(self, store, capsys):
        assert run_cli(store, "list") == 0
        out = capsys.readouterr().out
        assert "v1" in out and "v2" in out
        assert "test-data" in out

    def test_promote_through_stages(self, store):
        assert (
            run_cli(store, "promote", "--version", "1", "--stage", "Staging")
            == 0
        )
        assert (
            run_cli(
                store,
                "promote",
                "--version",
                "1",
                "--stage",
                "Production",
                "--alias",
            )
            == 0
        )
        client = MlflowClient()
        v1 = client.get_model_version(NAME, "1")
        assert v1.current_stage == "Production"
        assert "production" in v1.aliases

    def test_archive_existing_on_new_production(self, store):
        assert (
            run_cli(store, "promote", "--version", "2", "--stage", "Staging")
            == 0
        )
        assert (
            run_cli(
                store,
                "promote",
                "--version", "2",
                "--stage", "Production",
                "--archive-existing",
                "--alias",
            )
            == 0
        )  # fmt: skip
        client = MlflowClient()
        assert client.get_model_version(NAME, "1").current_stage == "Archived"
        assert (
            client.get_model_version(NAME, "2").current_stage == "Production"
        )
        assert "production" in client.get_model_version(NAME, "2").aliases

    def test_metadata_has_lineage(self, store, tmp_path):
        out_dir = tmp_path / "meta"
        assert (
            run_cli(
                store,
                "metadata",
                "--version",
                "2",
                "--output-dir",
                str(out_dir),
            )
            == 0
        )
        data = json.loads(
            (out_dir / f"{NAME}_v2.json").read_text(encoding="utf-8")
        )
        assert data["version"] == 2
        assert data["stage"] == "Production"
        assert data["lineage"]["data_version"] == "test-data"
        assert len(data["features"]) == 34
        assert data["target_mode"] == "level"
        assert "cv_rmse_mean" in data["metrics"]

    def test_verify_production_model(self, store, processed_dir, capsys):
        code = run_cli(
            store,
            "verify",
            "--stage",
            "Production",
            "--data",
            processed_dir,
            "--rows",
            "3",
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "VERIFIKASI BERHASIL" in out

    def test_verify_by_alias_and_version(self, store, processed_dir):
        assert (
            run_cli(
                store,
                "verify",
                "--alias",
                "production",
                "--data",
                processed_dir,
            )
            == 0
        )
        assert (
            run_cli(store, "verify", "--version", "1", "--data", processed_dir)
            == 0
        )

    def test_active_summary(self, store, capsys):
        assert run_cli(store, "active") == 0
        out = capsys.readouterr().out
        assert "| Versi aktif | 2 (stage Production) |" in out
        assert "test-data" in out


class TestErrors:
    def test_unknown_version_returns_error(self, store, capsys):
        assert (
            run_cli(store, "promote", "--version", "99", "--stage", "Staging")
            == 1
        )
        assert "ERROR" in capsys.readouterr().out

    def test_unknown_run_name_returns_error(self, store, capsys):
        assert run_cli(store, "register", "--run-name", "tidak-ada") == 1
        assert "tidak ditemukan" in capsys.readouterr().out

    def test_verify_fails_on_incompatible_dataset(
        self, store, tmp_path, capsys
    ):
        import pandas as pd

        path = tmp_path / "old.parquet"
        pd.DataFrame(
            {"timestamp": pd.date_range("2026-08-01", periods=5, freq="h")}
        ).to_parquet(path)
        assert (
            run_cli(
                store, "verify", "--stage", "Production", "--data", str(path)
            )
            == 1
        )
        assert "tidak punya kolom" in capsys.readouterr().out


class TestCiSupport:
    def test_register_with_tags_and_github_output(
        self, store, tmp_path, monkeypatch
    ):
        output = tmp_path / "gh_output.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))
        code = run_cli(
            store,
            "register",
            "--run-name", "r1",
            "--description", "dari CI",
            "--tag", "data_version=data-v4",
            "--tag", "ci_run=123",
        )  # fmt: skip
        assert code == 0
        version = output.read_text(encoding="utf-8").strip().split("=")[1]
        client = MlflowClient()
        found = client.get_model_version(NAME, version)
        assert found.description == "dari CI"
        assert found.tags["data_version"] == "data-v4"
        assert found.tags["ci_run"] == "123"

    def test_malformed_tag_returns_error(self, store, capsys):
        code = run_cli(store, "register", "--run-name", "r1", "--tag", "salah")
        assert code == 1
        assert "KEY=VALUE" in capsys.readouterr().out
