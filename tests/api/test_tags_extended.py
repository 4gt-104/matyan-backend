"""Extended tag API tests — creation errors, description/archived updates, tagged runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.storage import entities, runs

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database

from .conftest import BASE


class TestTagCreateEdgeCases:
    def test_create_duplicate_name(self, db: Database, client: TestClient) -> None:
        entities.create_tag(db, "dup-tag")
        resp = client.post(f"{BASE}/tags/", json={"name": "dup-tag"})
        assert resp.status_code == 400


class TestTagUpdateEdgeCases:
    def test_update_description(self, db: Database, client: TestClient) -> None:
        tag = entities.create_tag(db, "desc-tag")
        resp = client.put(
            f"{BASE}/tags/{tag['id']}/",
            json={"description": "A description"},
        )
        assert resp.status_code == 200

    def test_update_archived(self, db: Database, client: TestClient) -> None:
        tag = entities.create_tag(db, "arch-tag")
        resp = client.put(
            f"{BASE}/tags/{tag['id']}/",
            json={"archived": True},
        )
        assert resp.status_code == 200

    def test_update_nonexistent(self, client: TestClient) -> None:
        resp = client.put(
            f"{BASE}/tags/nonexistent/",
            json={"name": "x"},
        )
        assert resp.status_code == 404


class TestTaggedRuns:
    def test_runs_with_experiment(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "tag-run-exp")
        runs.create_run(db, "tr1", experiment_id=exp["id"])
        tag = entities.create_tag(db, "tag-for-runs")
        entities.add_tag_to_run(db, "tr1", tag["id"])

        resp = client.get(f"{BASE}/tags/{tag['id']}/runs/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["runs"][0]["experiment"] == "tag-run-exp"

    def test_runs_nonexistent_tag(self, client: TestClient) -> None:
        resp = client.get(f"{BASE}/tags/nonexistent/runs/")
        assert resp.status_code == 404
