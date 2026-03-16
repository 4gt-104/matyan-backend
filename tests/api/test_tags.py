"""API-level tests for tag endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matyan_backend.storage import entities, runs

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database

from .conftest import BASE

URL = f"{BASE}/tags"


def _create_tag(client: TestClient, name: str = "my-tag", **kw: Any) -> dict:  # noqa: ANN401
    resp = client.post(f"{URL}/", json={"name": name, **kw})
    assert resp.status_code == 200
    return resp.json()


class TestTagList:
    def test_empty(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_created_tags(self, client: TestClient) -> None:
        _create_tag(client, "alpha")
        _create_tag(client, "beta")
        resp = client.get(f"{URL}/")
        names = {t["name"] for t in resp.json()}
        assert names == {"alpha", "beta"}

    def test_run_count_is_populated(self, client: TestClient, db: Database) -> None:
        tag = _create_tag(client, "counted")
        tag_id = tag["id"]
        runs.create_run(db, "run1")
        entities.add_tag_to_run(db, "run1", tag_id)

        resp = client.get(f"{URL}/")
        tag_out = next(t for t in resp.json() if t["id"] == tag_id)
        assert tag_out["run_count"] == 1


class TestTagSearch:
    def test_search_no_match(self, client: TestClient) -> None:
        _create_tag(client, "visible")
        resp = client.get(f"{URL}/search/", params={"q": "nope"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_search_by_substring(self, client: TestClient) -> None:
        _create_tag(client, "alpha-release")
        _create_tag(client, "beta-release")
        _create_tag(client, "gamma")
        resp = client.get(f"{URL}/search/", params={"q": "release"})
        names = {t["name"] for t in resp.json()}
        assert names == {"alpha-release", "beta-release"}


class TestTagCreate:
    def test_create_basic(self, client: TestClient) -> None:
        data = _create_tag(client, "new-tag")
        assert data["status"] == "OK"
        assert "id" in data

    def test_create_with_color_and_desc(self, client: TestClient) -> None:
        _create_tag(client, "styled", color="#ff0000", description="red tag")
        resp = client.get(f"{URL}/")
        tag = resp.json()[0]
        assert tag["color"] == "#ff0000"
        assert tag["description"] == "red tag"


class TestTagGetById:
    def test_get_existing(self, client: TestClient) -> None:
        created = _create_tag(client, "findme")
        resp = client.get(f"{URL}/{created['id']}/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "findme"
        assert body["archived"] is False

    def test_get_nonexistent(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/00000000-0000-0000-0000-000000000000/")
        assert resp.status_code == 404


class TestTagUpdate:
    def test_update_name(self, client: TestClient) -> None:
        created = _create_tag(client, "old-name")
        resp = client.put(f"{URL}/{created['id']}/", json={"name": "new-name"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "OK"

        fetched = client.get(f"{URL}/{created['id']}/").json()
        assert fetched["name"] == "new-name"

    def test_update_archived(self, client: TestClient) -> None:
        created = _create_tag(client, "to-archive")
        client.put(f"{URL}/{created['id']}/", json={"archived": True})
        fetched = client.get(f"{URL}/{created['id']}/").json()
        assert fetched["archived"] is True

    def test_update_nonexistent(self, client: TestClient) -> None:
        resp = client.put(
            f"{URL}/00000000-0000-0000-0000-000000000000/",
            json={"name": "x"},
        )
        assert resp.status_code == 404


class TestTagDelete:
    def test_delete_existing(self, client: TestClient) -> None:
        created = _create_tag(client, "delete-me")
        resp = client.delete(f"{URL}/{created['id']}/")
        assert resp.status_code == 200

        resp = client.get(f"{URL}/{created['id']}/")
        assert resp.status_code == 404

    def test_delete_nonexistent(self, client: TestClient) -> None:
        resp = client.delete(f"{URL}/00000000-0000-0000-0000-000000000000/")
        assert resp.status_code == 404


class TestTaggedRuns:
    def test_get_runs_for_tag(self, client: TestClient, db: Database) -> None:
        created = _create_tag(client, "with-runs")
        tag_id = created["id"]

        runs.create_run(db, "r1", name="Run One")
        runs.create_run(db, "r2", name="Run Two")
        entities.add_tag_to_run(db, "r1", tag_id)
        entities.add_tag_to_run(db, "r2", tag_id)

        resp = client.get(f"{URL}/{tag_id}/runs/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == tag_id
        run_ids = {r["run_id"] for r in body["runs"]}
        assert run_ids == {"r1", "r2"}

    def test_get_runs_for_nonexistent_tag(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/00000000-0000-0000-0000-000000000000/runs/")
        assert resp.status_code == 404
