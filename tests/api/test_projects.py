"""API-level tests for project endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.storage import runs
from matyan_backend.storage.indexes import deindex_run

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database

from .conftest import BASE

URL = f"{BASE}/projects"


class TestProjectInfo:
    def test_get_project(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/")
        assert resp.status_code == 200
        body = resp.json()
        assert "name" in body
        assert "description" in body
        assert "telemetry_enabled" in body


class TestProjectActivity:
    def test_empty(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/activity/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["num_runs"] == 0
        assert body["num_experiments"] == 0

    def test_with_runs(self, client: TestClient, db: Database) -> None:
        runs.create_run(db, "proj-r1")
        runs.create_run(db, "proj-r2")
        resp = client.get(f"{URL}/activity/")
        body = resp.json()
        assert body["num_runs"] == 2
        assert body["num_active_runs"] == 2

    def test_pending_deletion_excluded(self, client: TestClient, db: Database) -> None:
        runs.create_run(db, "pd-act-1")
        runs.create_run(db, "pd-act-2")
        runs.mark_pending_deletion(db, "pd-act-2")
        deindex_run(db, "pd-act-2")
        resp = client.get(f"{URL}/activity/")
        body = resp.json()
        assert body["num_runs"] == 1


class TestProjectParams:
    def test_empty(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/params/")
        assert resp.status_code == 200

    def test_exclude_params(self, client: TestClient, db: Database) -> None:
        runs.create_run(db, "pp-ex1")
        runs.set_run_attrs(db, "pp-ex1", (), {"lr": 0.01})
        runs.set_context(db, "pp-ex1", 0, {})
        runs.set_trace_info(db, "pp-ex1", 0, "loss", dtype="float", last=0.1)

        resp = client.get(f"{URL}/params/", params={"exclude_params": "true"})
        assert resp.status_code == 200
        body = resp.json()
        assert "params" not in body
        assert "loss" in body.get("metric", {})


class TestProjectPinnedSequences:
    def test_empty(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/pinned-sequences/")
        assert resp.status_code == 200
        assert resp.json()["sequences"] == []

    def test_set_and_get(self, client: TestClient) -> None:
        seqs = [{"name": "loss", "context": {"subset": "train"}}]
        post_resp = client.post(f"{URL}/pinned-sequences/", json={"sequences": seqs})
        assert post_resp.status_code == 200
        assert len(post_resp.json()["sequences"]) == 1

        get_resp = client.get(f"{URL}/pinned-sequences/")
        body = get_resp.json()
        assert len(body["sequences"]) == 1
        assert body["sequences"][0]["name"] == "loss"


class TestProjectStatus:
    def test_status(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/status/")
        assert resp.status_code == 200
