"""Tests for the /health/live and /health/ready endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from matyan_backend.app import app

if TYPE_CHECKING:
    from collections.abc import Generator


def _client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ruff: noqa: SLF001


@pytest.fixture
def _mock_ping_ok() -> Generator[None]:
    """Patch FDB ping to return True (healthy). Use via @pytest.mark.usefixtures."""
    with patch("matyan_backend.api.health.ping", return_value=True):
        yield


@pytest.fixture
def _mock_ping_fail() -> Generator[None]:
    """Patch FDB ping to raise (unhealthy). Use via @pytest.mark.usefixtures."""
    with patch(
        "matyan_backend.api.health.ping",
        side_effect=RuntimeError("FDB unavailable"),
    ):
        yield


class TestLiveness:
    def test_returns_200(self) -> None:
        resp = _client().get("/health/live")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_without_trailing_slash(self) -> None:
        resp = _client().get("/health/live")
        assert resp.status_code == 200


class TestReadiness:
    @pytest.mark.usefixtures("_mock_ping_ok")
    @patch("matyan_backend.api.health.get_ingestion_producer")
    @patch("matyan_backend.api.health.get_producer")
    def test_all_healthy(
        self,
        mock_control: MagicMock,
        mock_ingestion: MagicMock,
    ) -> None:
        mock_control.return_value._producer = MagicMock()
        mock_ingestion.return_value._producer = MagicMock()

        resp = _client().get("/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["checks"]["fdb"] == "ok"
        assert body["checks"]["kafka"]["control_producer"] is True
        assert body["checks"]["kafka"]["ingestion_producer"] is True

    @pytest.mark.usefixtures("_mock_ping_fail")
    @patch("matyan_backend.api.health.get_ingestion_producer")
    @patch("matyan_backend.api.health.get_producer")
    def test_fdb_down_returns_503(
        self,
        mock_control: MagicMock,
        mock_ingestion: MagicMock,
    ) -> None:
        mock_control.return_value._producer = MagicMock()
        mock_ingestion.return_value._producer = MagicMock()

        resp = _client().get("/health/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert "FDB unavailable" in body["checks"]["fdb"]

    @pytest.mark.usefixtures("_mock_ping_ok")
    @patch("matyan_backend.api.health.get_ingestion_producer")
    @patch("matyan_backend.api.health.get_producer")
    def test_kafka_not_started_returns_503(
        self,
        mock_control: MagicMock,
        mock_ingestion: MagicMock,
    ) -> None:
        mock_control.return_value._producer = None
        mock_ingestion.return_value._producer = None

        resp = _client().get("/health/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["checks"]["kafka"]["control_producer"] is False
        assert body["checks"]["kafka"]["ingestion_producer"] is False

    @pytest.mark.usefixtures("_mock_ping_ok")
    @patch("matyan_backend.api.health.get_ingestion_producer")
    @patch("matyan_backend.api.health.get_producer")
    def test_partial_kafka_returns_503(
        self,
        mock_control: MagicMock,
        mock_ingestion: MagicMock,
    ) -> None:
        mock_control.return_value._producer = MagicMock()
        mock_ingestion.return_value._producer = None

        resp = _client().get("/health/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["checks"]["kafka"]["control_producer"] is True
        assert body["checks"]["kafka"]["ingestion_producer"] is False

    def test_ready_fdb_integration(self) -> None:
        """With real FDB running, the FDB check should pass (Kafka may not be started)."""
        resp = _client().get("/health/ready")
        body = resp.json()
        assert body["checks"]["fdb"] == "ok"
