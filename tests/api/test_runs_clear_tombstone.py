"""API-level tests for POST /runs/{run_id}/clear-tombstone/."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.storage import indexes, runs

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database

from .conftest import BASE

URL = f"{BASE}/runs"


class TestClearRunTombstone:
    def test_clear_existing_tombstone(self, client: TestClient, db: Database) -> None:
        runs.create_run(db, "tomb_api_1")
        runs.delete_run(db, "tomb_api_1")
        assert indexes.is_run_deleted(db, "tomb_api_1") is True

        resp = client.post(f"{URL}/tomb_api_1/clear-tombstone/")
        assert resp.status_code == 204

        assert indexes.is_run_deleted(db, "tomb_api_1") is False

    def test_clear_nonexistent_tombstone_is_idempotent(self, client: TestClient, db: Database) -> None:
        assert indexes.is_run_deleted(db, "tomb_api_no_exist") is False
        resp = client.post(f"{URL}/tomb_api_no_exist/clear-tombstone/")
        assert resp.status_code == 204
