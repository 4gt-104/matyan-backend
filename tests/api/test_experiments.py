"""API-level tests for experiment endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.storage import entities, runs

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database

from .conftest import BASE

URL = f"{BASE}/experiments"


def _create_experiment(client: TestClient, name: str = "exp1") -> dict:
    resp = client.post(f"{URL}/", json={"name": name})
    assert resp.status_code == 200
    return resp.json()


class TestExperimentList:
    def test_empty(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_created(self, client: TestClient) -> None:
        _create_experiment(client, "exp-a")
        _create_experiment(client, "exp-b")
        resp = client.get(f"{URL}/")
        names = {e["name"] for e in resp.json()}
        assert names == {"exp-a", "exp-b"}


class TestExperimentSearch:
    def test_search_by_name(self, client: TestClient) -> None:
        _create_experiment(client, "search-target")
        _create_experiment(client, "other")
        resp = client.get(f"{URL}/search/", params={"q": "target"})
        assert len(resp.json()) == 1
        assert resp.json()[0]["name"] == "search-target"

    def test_search_empty_query(self, client: TestClient) -> None:
        _create_experiment(client, "all")
        resp = client.get(f"{URL}/search/")
        assert len(resp.json()) >= 1


class TestExperimentCreate:
    def test_create(self, client: TestClient) -> None:
        data = _create_experiment(client, "new-exp")
        assert data["status"] == "OK"
        assert "id" in data

    def test_duplicate_name_fails(self, client: TestClient) -> None:
        _create_experiment(client, "dup-exp")
        resp = client.post(f"{URL}/", json={"name": "dup-exp"})
        assert resp.status_code == 400


class TestExperimentGetById:
    def test_get_existing(self, client: TestClient) -> None:
        created = _create_experiment(client, "get-me")
        resp = client.get(f"{URL}/{created['id']}/")
        assert resp.status_code == 200
        assert resp.json()["name"] == "get-me"

    def test_get_nonexistent(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/00000000-0000-0000-0000-000000000000/")
        assert resp.status_code == 404


class TestExperimentUpdate:
    def test_update_name(self, client: TestClient) -> None:
        created = _create_experiment(client, "old")
        resp = client.put(f"{URL}/{created['id']}/", json={"name": "new"})
        assert resp.status_code == 200
        fetched = client.get(f"{URL}/{created['id']}/").json()
        assert fetched["name"] == "new"

    def test_update_archived(self, client: TestClient) -> None:
        created = _create_experiment(client, "to-archive")
        client.put(f"{URL}/{created['id']}/", json={"archived": True})
        fetched = client.get(f"{URL}/{created['id']}/").json()
        assert fetched["archived"] is True

    def test_update_nonexistent(self, client: TestClient) -> None:
        resp = client.put(
            f"{URL}/00000000-0000-0000-0000-000000000000/",
            json={"name": "x"},
        )
        assert resp.status_code == 404


class TestExperimentDelete:
    def test_delete(self, client: TestClient) -> None:
        created = _create_experiment(client, "del-me")
        resp = client.delete(f"{URL}/{created['id']}/")
        assert resp.status_code == 200
        assert client.get(f"{URL}/{created['id']}/").status_code == 404

    def test_delete_nonexistent(self, client: TestClient) -> None:
        resp = client.delete(f"{URL}/00000000-0000-0000-0000-000000000000/")
        assert resp.status_code == 404


class TestExperimentRuns:
    def test_get_runs(self, client: TestClient, db: Database) -> None:
        created = _create_experiment(client, "with-runs")
        exp_id = created["id"]
        runs.create_run(db, "run-a", name="A", experiment_id=exp_id)
        runs.create_run(db, "run-b", name="B", experiment_id=exp_id)
        entities.set_run_experiment(db, "run-a", exp_id)
        entities.set_run_experiment(db, "run-b", exp_id)

        resp = client.get(f"{URL}/{exp_id}/runs/")
        assert resp.status_code == 200
        body = resp.json()
        run_ids = {r["run_id"] for r in body["runs"]}
        assert run_ids == {"run-a", "run-b"}

    def test_get_runs_nonexistent_experiment(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/00000000-0000-0000-0000-000000000000/runs/")
        assert resp.status_code == 404


class TestExperimentNotes:
    def test_note_crud(self, client: TestClient) -> None:
        created = _create_experiment(client, "noted")
        exp_id = created["id"]

        create_resp = client.post(f"{URL}/{exp_id}/note/", json={"content": "hello"})
        assert create_resp.status_code == 201
        note_id = create_resp.json()["id"]

        get_resp = client.get(f"{URL}/{exp_id}/note/{note_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["content"] == "hello"

        client.put(f"{URL}/{exp_id}/note/{note_id}", json={"content": "updated"})
        updated = client.get(f"{URL}/{exp_id}/note/{note_id}").json()
        assert updated["content"] == "updated"

        del_resp = client.delete(f"{URL}/{exp_id}/note/{note_id}")
        assert del_resp.status_code == 200

        assert client.get(f"{URL}/{exp_id}/note/{note_id}").status_code == 404


class TestExperimentActivity:
    def test_activity(self, client: TestClient, db: Database) -> None:
        created = _create_experiment(client, "activity-exp")
        exp_id = created["id"]
        runs.create_run(db, "act-r1", experiment_id=exp_id)
        entities.set_run_experiment(db, "act-r1", exp_id)

        resp = client.get(f"{URL}/{exp_id}/activity/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["num_runs"] == 1
        assert body["num_active_runs"] == 1
