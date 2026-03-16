"""Extended experiment API tests — notes, activity, edge cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.storage import entities, runs

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database

from .conftest import BASE


class TestExperimentNotes:
    def test_list_notes(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "note-list-exp")
        entities.create_note(db, "first", experiment_id=exp["id"])

        resp = client.get(f"{BASE}/experiments/{exp['id']}/note/")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_create_note(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "note-create-exp")
        resp = client.post(
            f"{BASE}/experiments/{exp['id']}/note/",
            json={"content": "hello note"},
        )
        assert resp.status_code == 201
        assert "id" in resp.json()

    def test_create_note_nonexistent_exp(self, db: Database, client: TestClient) -> None:  # noqa: ARG002
        resp = client.post(f"{BASE}/experiments/nonexistent/note/", json={"content": "x"})
        assert resp.status_code == 404

    def test_get_note(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "note-get-exp")
        resp = client.post(
            f"{BASE}/experiments/{exp['id']}/note/",
            json={"content": "detail note"},
        )
        note_id = resp.json()["id"]

        resp = client.get(f"{BASE}/experiments/{exp['id']}/note/{note_id}/")
        assert resp.status_code == 200
        assert resp.json()["content"] == "detail note"

    def test_get_note_nonexistent(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "note-get-exp2")
        resp = client.get(f"{BASE}/experiments/{exp['id']}/note/fake-id/")
        assert resp.status_code == 404

    def test_update_note(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "note-upd-exp")
        resp = client.post(f"{BASE}/experiments/{exp['id']}/note/", json={"content": "old"})
        note_id = resp.json()["id"]

        resp = client.put(
            f"{BASE}/experiments/{exp['id']}/note/{note_id}/",
            json={"content": "new"},
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == "new"

    def test_update_note_nonexistent(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "note-upd-exp2")
        resp = client.put(
            f"{BASE}/experiments/{exp['id']}/note/fake-id/",
            json={"content": "x"},
        )
        assert resp.status_code == 404

    def test_delete_note(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "note-del-exp")
        resp = client.post(f"{BASE}/experiments/{exp['id']}/note/", json={"content": "delete"})
        note_id = resp.json()["id"]

        resp = client.delete(f"{BASE}/experiments/{exp['id']}/note/{note_id}/")
        assert resp.status_code == 200

    def test_delete_note_nonexistent(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "note-del-exp2")
        resp = client.delete(f"{BASE}/experiments/{exp['id']}/note/fake-id/")
        assert resp.status_code == 404

    def test_list_notes_nonexistent_exp(self, client: TestClient) -> None:
        resp = client.get(f"{BASE}/experiments/nonexistent/note/")
        assert resp.status_code == 404


class TestExperimentActivity:
    def test_activity(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "activity-exp")
        runs.create_run(db, "eact1", experiment_id=exp["id"])
        entities.set_run_experiment(db, "eact1", exp["id"])

        resp = client.get(f"{BASE}/experiments/{exp['id']}/activity/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_runs"] == 1

    def test_activity_nonexistent(self, client: TestClient) -> None:
        resp = client.get(f"{BASE}/experiments/nonexistent/activity/")
        assert resp.status_code == 404

    def test_activity_archived_run(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "activity-exp2")
        runs.create_run(db, "eact2", experiment_id=exp["id"])
        entities.set_run_experiment(db, "eact2", exp["id"])
        runs.update_run_meta(db, "eact2", is_archived=True)

        resp = client.get(f"{BASE}/experiments/{exp['id']}/activity/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_archived_runs"] == 1


class TestExperimentRuns:
    def test_runs_with_offset(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "runs-off-exp")
        for i in range(3):
            rh = f"eroff{i}"
            runs.create_run(db, rh, experiment_id=exp["id"])
            entities.set_run_experiment(db, rh, exp["id"])

        run_hashes = entities.get_runs_for_experiment(db, exp["id"])
        if len(run_hashes) >= 2:
            offset_hash = run_hashes[0]
            resp = client.get(
                f"{BASE}/experiments/{exp['id']}/runs/",
                params={"offset": offset_hash},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert all(r["run_id"] != offset_hash for r in data["runs"])

    def test_runs_nonexistent_exp(self, client: TestClient) -> None:
        resp = client.get(f"{BASE}/experiments/nonexistent/runs/")
        assert resp.status_code == 404
