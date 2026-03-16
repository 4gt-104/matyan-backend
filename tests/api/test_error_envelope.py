"""Tests for standard error envelope (4xx/5xx responses)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

from .conftest import BASE

URL_RUNS = f"{BASE}/runs"
URL_EXPERIMENTS = f"{BASE}/experiments"


def _assert_envelope(body: dict) -> None:
    assert "error" in body
    assert "detail" in body
    assert body["detail"] == body["error"]["message"]
    assert "code" in body["error"]
    assert "message" in body["error"]


class TestErrorEnvelope404:
    """404 responses use standard envelope."""

    def test_get_nonexistent_run(self, client: TestClient) -> None:
        resp = client.get(f"{URL_RUNS}/nonexistent-run-hash/info/")
        assert resp.status_code == 404
        body = resp.json()
        _assert_envelope(body)
        assert body["error"]["code"] == "not_found"
        assert body["error"]["message"]  # message present (may be default or framework default)

    def test_get_nonexistent_experiment(self, client: TestClient) -> None:
        resp = client.get(f"{URL_EXPERIMENTS}/00000000-0000-0000-0000-000000000000/")
        assert resp.status_code == 404
        body = resp.json()
        _assert_envelope(body)
        assert body["error"]["code"] == "not_found"


class TestErrorEnvelope400:
    """400 responses use standard envelope with custom message."""

    def test_duplicate_experiment_name(self, client: TestClient) -> None:
        client.post(f"{URL_EXPERIMENTS}/", json={"name": "dup"})
        resp = client.post(f"{URL_EXPERIMENTS}/", json={"name": "dup"})
        assert resp.status_code == 400
        body = resp.json()
        _assert_envelope(body)
        assert body["error"]["code"] == "bad_request"
        assert "dup" in body["error"]["message"] or "exist" in body["error"]["message"].lower()


class TestErrorEnvelope422:
    """422 validation errors use standard envelope."""

    def test_invalid_body(self, client: TestClient) -> None:
        resp = client.post(f"{URL_EXPERIMENTS}/", json={"wrong_key": "x"})
        assert resp.status_code == 422
        body = resp.json()
        _assert_envelope(body)
        assert body["error"]["code"] == "validation_error"


class TestErrorEnvelopeDetailBackwardCompat:
    """detail field mirrors error.message for UI compatibility."""

    def test_detail_equals_message(self, client: TestClient) -> None:
        resp = client.get(f"{URL_RUNS}/missing/info/")
        assert resp.status_code == 404
        body = resp.json()
        assert body["detail"] == body["error"]["message"]
