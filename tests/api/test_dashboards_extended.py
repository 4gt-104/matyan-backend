"""Extended tests for dashboards, dashboard apps, and reports edge cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.storage import entities

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database

from .conftest import BASE


class TestDashboardUpdate:
    def test_update_description(self, db: Database, client: TestClient) -> None:
        d = entities.create_dashboard(db, "upd-dash")
        resp = client.put(
            f"{BASE}/dashboards/{d['id']}/",
            json={"description": "new desc"},
        )
        assert resp.status_code == 200
        assert resp.json()["description"] == "new desc"

    def test_update_nonexistent(self, client: TestClient) -> None:
        resp = client.put(
            f"{BASE}/dashboards/nonexistent/",
            json={"name": "x"},
        )
        assert resp.status_code == 404


class TestDashboardAppUpdate:
    def test_update_type(self, db: Database, client: TestClient) -> None:
        a = entities.create_dashboard_app(db, "metric_explorer", state={"key": "val"})
        resp = client.put(
            f"{BASE}/apps/{a['id']}/",
            json={"type": "params_explorer"},
        )
        assert resp.status_code == 200

    def test_update_state(self, db: Database, client: TestClient) -> None:
        a = entities.create_dashboard_app(db, "metric_explorer", state={})
        resp = client.put(
            f"{BASE}/apps/{a['id']}/",
            json={"state": {"new": "state"}},
        )
        assert resp.status_code == 200

    def test_update_nonexistent(self, client: TestClient) -> None:
        resp = client.put(
            f"{BASE}/apps/nonexistent/",
            json={"type": "x"},
        )
        assert resp.status_code == 404

    def test_delete_nonexistent(self, client: TestClient) -> None:
        resp = client.delete(f"{BASE}/apps/nonexistent/")
        assert resp.status_code == 404


class TestReportUpdate:
    def test_update_code(self, db: Database, client: TestClient) -> None:
        r = entities.create_report(db, "test-report")
        resp = client.put(
            f"{BASE}/reports/{r['id']}/",
            json={"code": "new code"},
        )
        assert resp.status_code == 200

    def test_update_description(self, db: Database, client: TestClient) -> None:
        r = entities.create_report(db, "test-report2")
        resp = client.put(
            f"{BASE}/reports/{r['id']}/",
            json={"description": "new desc"},
        )
        assert resp.status_code == 200

    def test_update_nonexistent(self, client: TestClient) -> None:
        resp = client.put(
            f"{BASE}/reports/nonexistent/",
            json={"name": "x"},
        )
        assert resp.status_code == 404

    def test_delete_nonexistent(self, client: TestClient) -> None:
        resp = client.delete(f"{BASE}/reports/nonexistent/")
        assert resp.status_code == 404
