"""API-level tests for the version endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .conftest import BASE

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

URL = f"{BASE}/version"


def test_get_version_returns_200_and_fields(client: TestClient) -> None:
    """GET /api/v1/rest/version/ returns 200 with version, component, fdb_api_version."""
    resp = client.get(f"{URL}/")
    assert resp.status_code == 200
    body = resp.json()
    assert "version" in body
    assert body["component"] == "backend"
    assert "fdb_api_version" in body
    assert isinstance(body["fdb_api_version"], int)
    # Version string typically semver-like
    assert body["version"]
    assert body["version"].replace(".", "").replace("dev", "").replace("0", "").isalnum() or "0" in body["version"]
