"""API-level tests for dashboard, dashboard-app, and report endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

from .conftest import BASE

# ── Dashboards ──────────────────────────────────────────────────────────────

DASH_URL = f"{BASE}/dashboards"


def _create_dashboard(client: TestClient, name: str = "My Dashboard", **kw: Any) -> dict:  # noqa: ANN401
    resp = client.post(f"{DASH_URL}/", json={"name": name, **kw})
    assert resp.status_code == 201
    return resp.json()


class TestDashboardList:
    def test_empty(self, client: TestClient) -> None:
        resp = client.get(f"{DASH_URL}/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_created(self, client: TestClient) -> None:
        _create_dashboard(client, "dash-a")
        _create_dashboard(client, "dash-b")
        resp = client.get(f"{DASH_URL}/")
        names = {d["name"] for d in resp.json()}
        assert names == {"dash-a", "dash-b"}


class TestDashboardCreate:
    def test_create_basic(self, client: TestClient) -> None:
        d = _create_dashboard(client, "new-dash")
        assert d["name"] == "new-dash"
        assert "id" in d
        assert d["created_at"] is not None

    def test_create_with_description(self, client: TestClient) -> None:
        d = _create_dashboard(client, "described", description="a desc")
        assert d["description"] == "a desc"


class TestDashboardGetById:
    def test_get_existing(self, client: TestClient) -> None:
        created = _create_dashboard(client, "get-me")
        resp = client.get(f"{DASH_URL}/{created['id']}/")
        assert resp.status_code == 200
        assert resp.json()["name"] == "get-me"

    def test_get_nonexistent(self, client: TestClient) -> None:
        resp = client.get(f"{DASH_URL}/nonexistent-id/")
        assert resp.status_code == 404


class TestDashboardUpdate:
    def test_update_name(self, client: TestClient) -> None:
        created = _create_dashboard(client, "old-name")
        resp = client.put(f"{DASH_URL}/{created['id']}/", json={"name": "new-name"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "new-name"

    def test_update_nonexistent(self, client: TestClient) -> None:
        resp = client.put(f"{DASH_URL}/bad-id/", json={"name": "x"})
        assert resp.status_code == 404


class TestDashboardDelete:
    def test_delete(self, client: TestClient) -> None:
        created = _create_dashboard(client, "del-me")
        resp = client.delete(f"{DASH_URL}/{created['id']}/")
        assert resp.status_code == 204

        listing = client.get(f"{DASH_URL}/").json()
        assert all(d["id"] != created["id"] for d in listing)

    def test_delete_nonexistent(self, client: TestClient) -> None:
        resp = client.delete(f"{DASH_URL}/bad-id/")
        assert resp.status_code == 404


# ── Dashboard Apps ──────────────────────────────────────────────────────────

APPS_URL = f"{BASE}/apps"


def _create_app(client: TestClient, app_type: str = "metrics", state: dict | None = None) -> dict:
    resp = client.post(f"{APPS_URL}/", json={"type": app_type, "state": state or {}})
    assert resp.status_code == 201
    return resp.json()


class TestDashboardAppList:
    def test_empty(self, client: TestClient) -> None:
        resp = client.get(f"{APPS_URL}/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_created(self, client: TestClient) -> None:
        _create_app(client, "metrics")
        _create_app(client, "images")
        resp = client.get(f"{APPS_URL}/")
        types = {a["type"] for a in resp.json()}
        assert types == {"metrics", "images"}


class TestDashboardAppCreate:
    def test_create(self, client: TestClient) -> None:
        a = _create_app(client, "params", state={"key": "val"})
        assert a["type"] == "params"
        assert a["state"] == {"key": "val"}
        assert "id" in a


class TestDashboardAppGetById:
    def test_get_existing(self, client: TestClient) -> None:
        created = _create_app(client)
        resp = client.get(f"{APPS_URL}/{created['id']}/")
        assert resp.status_code == 200
        assert resp.json()["type"] == "metrics"

    def test_get_nonexistent(self, client: TestClient) -> None:
        resp = client.get(f"{APPS_URL}/bad-id/")
        assert resp.status_code == 404


class TestDashboardAppUpdate:
    def test_update_state(self, client: TestClient) -> None:
        created = _create_app(client, state={"a": 1})
        resp = client.put(f"{APPS_URL}/{created['id']}/", json={"state": {"a": 2}})
        assert resp.status_code == 200
        assert resp.json()["state"] == {"a": 2}

    def test_update_nonexistent(self, client: TestClient) -> None:
        resp = client.put(f"{APPS_URL}/bad-id/", json={"type": "x"})
        assert resp.status_code == 404


class TestDashboardAppDelete:
    def test_delete(self, client: TestClient) -> None:
        created = _create_app(client)
        resp = client.delete(f"{APPS_URL}/{created['id']}/")
        assert resp.status_code == 204

        assert client.get(f"{APPS_URL}/{created['id']}/").status_code == 404

    def test_delete_nonexistent(self, client: TestClient) -> None:
        resp = client.delete(f"{APPS_URL}/bad-id/")
        assert resp.status_code == 404


# ── Reports ─────────────────────────────────────────────────────────────────

REPORTS_URL = f"{BASE}/reports"


def _create_report(client: TestClient, name: str = "Report 1", **kw: Any) -> dict:  # noqa: ANN401
    resp = client.post(f"{REPORTS_URL}/", json={"name": name, **kw})
    assert resp.status_code == 201
    return resp.json()


class TestReportList:
    def test_empty(self, client: TestClient) -> None:
        resp = client.get(f"{REPORTS_URL}/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_created(self, client: TestClient) -> None:
        _create_report(client, "rep-a")
        _create_report(client, "rep-b")
        resp = client.get(f"{REPORTS_URL}/")
        names = {r["name"] for r in resp.json()}
        assert names == {"rep-a", "rep-b"}


class TestReportCreate:
    def test_create_basic(self, client: TestClient) -> None:
        r = _create_report(client, "new-rep")
        assert r["name"] == "new-rep"
        assert "id" in r

    def test_create_with_code_and_desc(self, client: TestClient) -> None:
        r = _create_report(client, "coded", code="print('hi')", description="d")
        assert r["code"] == "print('hi')"
        assert r["description"] == "d"


class TestReportGetById:
    def test_get_existing(self, client: TestClient) -> None:
        created = _create_report(client, "get-me")
        resp = client.get(f"{REPORTS_URL}/{created['id']}/")
        assert resp.status_code == 200
        assert resp.json()["name"] == "get-me"

    def test_get_nonexistent(self, client: TestClient) -> None:
        resp = client.get(f"{REPORTS_URL}/bad-id/")
        assert resp.status_code == 404


class TestReportUpdate:
    def test_update_name(self, client: TestClient) -> None:
        created = _create_report(client, "old")
        resp = client.put(f"{REPORTS_URL}/{created['id']}/", json={"name": "new"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "new"

    def test_update_nonexistent(self, client: TestClient) -> None:
        resp = client.put(f"{REPORTS_URL}/bad-id/", json={"name": "x"})
        assert resp.status_code == 404


class TestReportDelete:
    def test_delete(self, client: TestClient) -> None:
        created = _create_report(client, "del-me")
        resp = client.delete(f"{REPORTS_URL}/{created['id']}/")
        assert resp.status_code == 204

        assert client.get(f"{REPORTS_URL}/{created['id']}/").status_code == 404

    def test_delete_nonexistent(self, client: TestClient) -> None:
        resp = client.delete(f"{REPORTS_URL}/bad-id/")
        assert resp.status_code == 404
