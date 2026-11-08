"""Tes ingest_data: validasi argumen, rencana FIRMS, retry, nama file."""

from datetime import date

import pytest
import requests

import ingest_data as ing

TODAY = date(2026, 11, 1)


class FakeResponse:
    """Respons palsu dengan status tertentu."""

    def __init__(self, status):
        self.status_code = status
        self.text = "ok"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return {"hourly": {"time": []}}


def patch_get(monkeypatch, statuses):
    """Ganti requests.get dengan urutan status; kembalikan daftar panggilan."""
    calls = []
    queue = list(statuses)

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        return FakeResponse(queue.pop(0))

    monkeypatch.setattr(ing.requests, "get", fake_get)
    return calls


class TestCheckDateRange:
    def test_valid_range(self):
        assert (
            ing.check_date_range(
                "historical", date(2023, 6, 1), date(2023, 11, 30), TODAY
            )
            is None
        )

    @pytest.mark.parametrize(
        "start,end",
        [
            (None, None),
            (date(2023, 6, 1), None),
            (date(2023, 6, 1), date(2023, 5, 1)),
            (date(2023, 6, 1), TODAY),
            (date(2022, 1, 1), date(2022, 2, 1)),
            (date(2023, 1, 1), date(2025, 1, 1)),
        ],
    )
    def test_invalid_ranges(self, start, end):
        assert ing.check_date_range("historical", start, end, TODAY)

    @pytest.mark.parametrize("mode", ["incremental", "backfill"])
    def test_dates_rejected_outside_historical(self, mode):
        message = ing.check_date_range(mode, date(2023, 6, 1), None, TODAY)
        assert "historical" in message

    def test_parse_date(self):
        assert ing.parse_date("2025-06-01") == date(2025, 6, 1)
        with pytest.raises(Exception):
            ing.parse_date("01-06-2025")


class TestFirmsPlan:
    def test_incremental_is_single_request(self):
        assert ing.build_firms_plan("incremental", 30, TODAY) == [(2, None)]

    def test_backfill_covers_requested_days(self):
        plan = ing.build_firms_plan("backfill", 30, TODAY)
        assert sum(days for days, _ in plan) == 30
        assert all(days <= ing.FIRMS_MAX_DAY_RANGE for days, _ in plan)
        assert plan[0][1] == date(2026, 10, 3)

    def test_historical_starts_with_margin(self):
        window = (date(2023, 6, 1), date(2023, 6, 12))
        plan = ing.build_firms_plan("historical", 0, TODAY, window)
        assert plan[0][1] == date(2023, 5, 30)
        assert sum(days for days, _ in plan) == 12 + 2
        assert all(days <= ing.FIRMS_MAX_DAY_RANGE for days, _ in plan)

    def test_url_uses_source_and_date(self):
        url = ing.build_firms_url(
            "KEY", 5, date(2023, 5, 30), ing.FIRMS_HISTORICAL_SOURCE
        )
        assert "VIIRS_SNPP_SP" in url
        assert url.endswith("/5/2023-05-30")

    def test_url_without_date(self):
        assert ing.build_firms_url("KEY", 2, None).endswith("/2")


class TestFiles:
    def test_live_filename(self):
        path = ing.build_raw_filepath("weather", "20260101_000000", "json")
        assert path.endswith("weather_20260101_000000.json")

    def test_historical_filename_has_marker_and_range(self):
        path = ing.build_raw_filepath(
            "firms",
            "20260101_000000",
            "csv",
            (date(2023, 6, 1), date(2023, 11, 30)),
        )
        assert "firms_hist_20230601_20231130_20260101_000000.csv" in path

    def test_merge_csv_texts_keeps_single_header(self):
        merged = ing.merge_csv_texts(["a,b\n1,2\n", "a,b\n3,4\n", ""])
        assert merged == "a,b\n1,2\n3,4\n"

    def test_merge_csv_texts_empty(self):
        assert ing.merge_csv_texts([]) == ""

    def test_backfill_days_are_clamped(self):
        assert ing.clamp_backfill_days(0) == 1
        assert ing.clamp_backfill_days(500) == ing.MAX_BACKFILL_DAYS


class TestRequests:
    def test_historical_request_uses_dates_not_past_days(self, monkeypatch):
        seen = {}

        def fake(url, params=None, sleep=None):
            seen["params"] = params
            return FakeResponse(200)

        monkeypatch.setattr(ing, "get_with_retry", fake)
        window = (date(2023, 6, 1), date(2023, 6, 3))
        ing.request_open_meteo(ing.AIR_QUALITY_URL, ("pm2_5",), 2, window)
        assert seen["params"]["start_date"] == "2023-06-01"
        assert seen["params"]["end_date"] == "2023-06-03"
        assert "past_days" not in seen["params"]

    def test_live_request_uses_past_days(self, monkeypatch):
        seen = {}

        def fake(url, params=None, sleep=None):
            seen["params"] = params
            return FakeResponse(200)

        monkeypatch.setattr(ing, "get_with_retry", fake)
        ing.request_open_meteo(ing.WEATHER_URL, ("pm2_5",), 2)
        assert seen["params"]["past_days"] == 2
        assert "start_date" not in seen["params"]


class TestRetry:
    def test_retries_server_errors_with_backoff(self, monkeypatch):
        calls = patch_get(monkeypatch, [500, 503, 200])
        delays = []
        response = ing.get_with_retry("http://x", sleep=delays.append)
        assert response.status_code == 200
        assert len(calls) == 3
        assert delays == [5, 15]

    def test_gives_up_after_three_retries(self, monkeypatch):
        calls = patch_get(monkeypatch, [500, 500, 500, 500])
        delays = []
        with pytest.raises(requests.exceptions.HTTPError):
            ing.get_with_retry("http://x", sleep=delays.append)
        assert len(calls) == 4
        assert delays == [5, 15, 45]

    def test_client_error_is_not_retried(self, monkeypatch):
        calls = patch_get(monkeypatch, [404, 200])
        with pytest.raises(requests.exceptions.HTTPError):
            ing.get_with_retry("http://x", sleep=lambda _: None)
        assert len(calls) == 1

    def test_describe_error_hides_url(self):
        error = requests.exceptions.HTTPError(response=FakeResponse(500))
        text = ing.describe_error(error)
        assert "500" in text and "http" not in text


class TestMissingKey:
    def test_firms_without_key_raises(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "")
        with pytest.raises(ing.MissingCredentialError):
            ing.ingest_firms("20260101_000000", "incremental", 30)

    def test_failed_step_returns_none(self, monkeypatch):
        monkeypatch.setenv("FIRMS_MAP_KEY", "")
        result = ing.run_ingestion_step(
            "firms", ing.ingest_firms, "20260101_000000", "incremental", 30
        )
        assert result is None
