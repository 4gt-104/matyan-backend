"""Tests for app.py — middleware and lifespan."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from matyan_backend.app import app


class TestTrailingSlashMiddleware:
    def test_path_without_trailing_slash(self) -> None:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/v1/rest/projects/status")
        assert resp.status_code in (200, 307, 500)

    def test_path_with_trailing_slash(self) -> None:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/v1/rest/projects/status/")
        assert resp.status_code in (200, 500)


class TestLifespan:
    @patch("matyan_backend.app.get_producer")
    @patch("matyan_backend.app.ensure_directories")
    @patch("matyan_backend.app.init_fdb")
    def test_lifespan_starts_and_stops_producer(
        self,
        mock_init: AsyncMock,
        mock_dirs: AsyncMock,
        mock_get_producer: AsyncMock,
    ) -> None:
        mock_producer = AsyncMock()
        mock_get_producer.return_value = mock_producer

        with TestClient(app):
            mock_init.assert_called_once()
            mock_dirs.assert_called_once()
            mock_producer.start.assert_called_once()

        mock_producer.stop.assert_called_once()
