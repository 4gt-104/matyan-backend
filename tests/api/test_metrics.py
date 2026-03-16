"""Tests for the /metrics/ Prometheus endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from matyan_backend.app import app
from matyan_backend.metrics import normalize_path


def _client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


class TestMetricsEndpoint:
    def test_returns_200_with_prometheus_content_type(self) -> None:
        resp = _client().get("/metrics/")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]

    def test_contains_default_process_metrics(self) -> None:
        resp = _client().get("/metrics/")
        body = resp.text
        assert "process_" in body or "python_" in body

    def test_contains_http_request_metrics_after_request(self) -> None:
        client = _client()
        client.get("/health/live")
        resp = client.get("/metrics/")
        body = resp.text
        assert "matyan_http_requests_total" in body
        assert "matyan_http_request_duration_seconds" in body

    def test_path_template_labels_normalized(self) -> None:
        client = _client()
        client.get("/health/live")
        resp = client.get("/metrics/")
        body = resp.text
        assert 'path_template="/health/live/"' in body

    def test_without_trailing_slash(self) -> None:
        resp = _client().get("/metrics")
        assert resp.status_code == 200


class TestPathNormalization:
    def test_hex_run_hash_replaced(self) -> None:

        result = normalize_path("/api/v1/rest/runs/0cf28fd9b5bf2002/info/")
        assert result == "/api/v1/rest/runs/{id}/info/"

    def test_uuid_replaced(self) -> None:

        result = normalize_path("/api/v1/rest/dashboards/550e8400-e29b-41d4-a716-446655440000/")
        assert result == "/api/v1/rest/dashboards/{id}/"

    def test_no_replacement_for_short_paths(self) -> None:

        result = normalize_path("/api/v1/rest/runs/")
        assert result == "/api/v1/rest/runs/"
