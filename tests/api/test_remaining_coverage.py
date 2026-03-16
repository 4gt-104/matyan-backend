"""Targeted tests to cover remaining gaps across multiple modules.

Covers: custom object get-batch/get-step with non-wrap types, blob batch,
experiment runs with offset/limit, dashboard with app_id, tags edge cases,
alignment with matching traces, run log edge cases, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from matyan_backend.api.runs._blob_uri import generate_uri
from matyan_backend.storage import entities, runs, sequences

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database

from .conftest import BASE

# --- Distribution get-batch (non-wrap type) ---


class TestDistributionGetBatch:
    def test_get_batch(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "dgb1")
        runs.set_context(db, "dgb1", 0, {})
        runs.set_trace_info(db, "dgb1", 0, "distributions", dtype="distribution", last=0.0, last_step=1)
        for i in range(2):
            sequences.write_sequence_step(
                db,
                "dgb1",
                0,
                "distributions",
                i,
                {
                    "data": [1.0, 2.0],
                    "bin_count": 2,
                    "range": [0, 5],
                },
            )

        resp = client.post(
            f"{BASE}/runs/dgb1/distributions/get-batch/",
            json=[{"name": "distributions", "context": {}}],
        )
        assert resp.status_code == 200

    def test_get_step(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "dgs1")
        runs.set_context(db, "dgs1", 0, {})
        runs.set_trace_info(db, "dgs1", 0, "distributions", dtype="distribution", last=0.0, last_step=0)
        sequences.write_sequence_step(
            db,
            "dgs1",
            0,
            "distributions",
            0,
            {
                "data": [1.0],
                "bin_count": 1,
                "range": [0, 1],
            },
        )

        resp = client.post(
            f"{BASE}/runs/dgs1/distributions/get-step/",
            json=[{"name": "distributions", "context": {}}],
            params={"record_step": 0},
        )
        assert resp.status_code == 200


# --- Figure get-batch and get-step ---


class TestFigureGetBatch:
    def test_get_batch(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "fgb1")
        runs.set_context(db, "fgb1", 0, {})
        runs.set_trace_info(db, "fgb1", 0, "figures", dtype="figure", last=0.0, last_step=0)
        sequences.write_sequence_step(
            db,
            "fgb1",
            0,
            "figures",
            0,
            {
                "data": {"x": [1], "y": [2]},
            },
        )

        resp = client.post(
            f"{BASE}/runs/fgb1/figures/get-batch/",
            json=[{"name": "figures", "context": {}}],
        )
        assert resp.status_code == 200


# --- Image search (uses use_list=True + blob URI) ---


class TestImageSearch:
    def test_search(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "ims1")
        runs.set_context(db, "ims1", 0, {})
        runs.set_trace_info(db, "ims1", 0, "images", dtype="image", last=0.0, last_step=0)
        sequences.write_sequence_step(
            db,
            "ims1",
            0,
            "images",
            0,
            [
                {"caption": "cat", "width": 64, "height": 64},
            ],
        )

        resp = client.get(f"{BASE}/runs/search/images/")
        assert resp.status_code == 200

    def test_get_batch(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "imgb1")
        runs.set_context(db, "imgb1", 0, {})
        runs.set_trace_info(db, "imgb1", 0, "images", dtype="image", last=0.0, last_step=0)
        sequences.write_sequence_step(
            db,
            "imgb1",
            0,
            "images",
            0,
            [
                {"caption": "test", "width": 32, "height": 32},
            ],
        )

        resp = client.post(
            f"{BASE}/runs/imgb1/images/get-batch/",
            json=[{"name": "images", "context": {}}],
        )
        assert resp.status_code == 200

    def test_get_step(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "imgs1")
        runs.set_context(db, "imgs1", 0, {})
        runs.set_trace_info(db, "imgs1", 0, "images", dtype="image", last=0.0, last_step=0)
        sequences.write_sequence_step(
            db,
            "imgs1",
            0,
            "images",
            0,
            [
                {"caption": "test", "width": 32, "height": 32},
            ],
        )

        resp = client.post(
            f"{BASE}/runs/imgs1/images/get-step/",
            json=[{"name": "images", "context": {}}],
            params={"record_step": 0},
        )
        assert resp.status_code == 200


# --- Audio search ---


class TestAudioSearch:
    def test_search(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "aus1")
        runs.set_context(db, "aus1", 0, {})
        runs.set_trace_info(db, "aus1", 0, "audios", dtype="audio", last=0.0, last_step=0)
        sequences.write_sequence_step(
            db,
            "aus1",
            0,
            "audios",
            0,
            [
                {"caption": "clip", "format": "wav"},
            ],
        )

        resp = client.get(f"{BASE}/runs/search/audios/")
        assert resp.status_code == 200


# --- Blob batch endpoint (images/audios) ---


class TestBlobBatch:
    @patch("matyan_backend.api.runs._custom_objects._fetch_blob_from_s3", return_value=b"fake-blob")
    def test_images_blob_batch(self, mock_fetch: MagicMock, db: Database, client: TestClient) -> None:  # noqa: ARG002
        runs.create_run(db, "bb1")
        uri = generate_uri("bb1", 0, "images", 0, 0)
        resp = client.post(f"{BASE}/runs/images/get-batch/", json=[uri])
        assert resp.status_code == 200

    def test_blob_batch_invalid_uri(self, client: TestClient) -> None:
        resp = client.post(f"{BASE}/runs/images/get-batch/", json=["invalid-uri"])
        assert resp.status_code == 200


# --- Experiment runs with limit/offset ---


class TestExperimentRunsExtended:
    def test_with_limit(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "lim-exp")
        for i in range(3):
            rh = f"limrun{i}"
            runs.create_run(db, rh, experiment_id=exp["id"])
            entities.set_run_experiment(db, rh, exp["id"])

        resp = client.get(f"{BASE}/experiments/{exp['id']}/runs/", params={"limit": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["runs"]) == 2

    def test_experiment_note_not_found(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "note-exp")
        resp = client.get(f"{BASE}/experiments/{exp['id']}/note/missing-note/")
        assert resp.status_code == 404

    def test_experiment_update_note(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "upnote-exp")
        resp = client.post(
            f"{BASE}/experiments/{exp['id']}/note/",
            json={"content": "initial"},
        )
        assert resp.status_code == 201
        note_id = resp.json()["id"]

        resp = client.put(
            f"{BASE}/experiments/{exp['id']}/note/{note_id}/",
            json={"content": "updated"},
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == "updated"

    def test_experiment_delete_note(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "delnote-exp")
        resp = client.post(
            f"{BASE}/experiments/{exp['id']}/note/",
            json={"content": "delete me"},
        )
        note_id = resp.json()["id"]

        resp = client.delete(f"{BASE}/experiments/{exp['id']}/note/{note_id}/")
        assert resp.status_code == 200


# --- Dashboard with app_id ---


class TestDashboardWithApp:
    def test_dashboard_with_app_type(self, db: Database, client: TestClient) -> None:  # noqa: ARG002
        app_resp = client.post(
            f"{BASE}/apps/",
            json={"type": "metric_explorer", "state": {}},
        )
        app_id = app_resp.json()["id"]

        resp = client.post(
            f"{BASE}/dashboards/",
            json={"name": "My Dashboard", "app_id": app_id},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["app_id"] == app_id


# --- Tags search with match ---


class TestTagSearchMatch:
    def test_search_match(self, db: Database, client: TestClient) -> None:
        entities.create_tag(db, "production")
        resp = client.get(f"{BASE}/tags/search/", params={"q": "prod"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    def test_tag_update_color(self, db: Database, client: TestClient) -> None:
        tag = entities.create_tag(db, "colorful")
        resp = client.put(
            f"{BASE}/tags/{tag['id']}/",
            json={"color": "#ff0000"},
        )
        assert resp.status_code == 200


# --- Run note edge cases ---


class TestRunNoteEdgeCases:
    def test_get_note_wrong_run(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "rn1")
        runs.create_run(db, "rn2")
        resp = client.post(f"{BASE}/runs/rn1/note/", json={"content": "note"})
        note_id = resp.json()["id"]

        resp = client.get(f"{BASE}/runs/rn2/note/{note_id}/")
        assert resp.status_code == 404

    def test_delete_note_wrong_run(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "rn3")
        runs.create_run(db, "rn4")
        resp = client.post(f"{BASE}/runs/rn3/note/", json={"content": "note"})
        note_id = resp.json()["id"]

        resp = client.delete(f"{BASE}/runs/rn4/note/{note_id}/")
        assert resp.status_code == 404


# --- Project params with exclude ---


class TestProjectParamsExtended:
    def test_exclude_params(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "ppe1")
        runs.set_run_attrs(db, "ppe1", (), {"hparams": {"lr": 0.01}})

        resp = client.get(f"{BASE}/projects/params/", params={"exclude_params": "true"})
        assert resp.status_code == 200


# --- Custom object get-batch edge cases ---


class TestCustomObjectGetBatchEdgeCases:
    def test_get_batch_context_mismatch(self, db: Database, client: TestClient) -> None:
        """When context doesn't match, the trace is skipped (line 248)."""
        runs.create_run(db, "cbce1")
        runs.set_context(db, "cbce1", 0, {})
        runs.set_trace_info(db, "cbce1", 0, "texts", dtype="text", last=0.0, last_step=0)
        sequences.write_sequence_step(db, "cbce1", 0, "texts", 0, {"data": "hi"})

        resp = client.post(
            f"{BASE}/runs/cbce1/texts/get-batch/",
            json=[{"name": "texts", "context": {"nonexistent": True}}],
        )
        assert resp.status_code == 200

    def test_get_step_step_not_found(self, db: Database, client: TestClient) -> None:
        """When the requested step doesn't exist (line 348-349)."""
        runs.create_run(db, "cbce2")
        runs.set_context(db, "cbce2", 0, {})
        runs.set_trace_info(db, "cbce2", 0, "texts", dtype="text", last=0.0, last_step=0)
        sequences.write_sequence_step(db, "cbce2", 0, "texts", 0, {"data": "hi"})

        resp = client.post(
            f"{BASE}/runs/cbce2/texts/get-step/",
            json=[{"name": "texts", "context": {}}],
            params={"record_step": 999},
        )
        assert resp.status_code == 200

    def test_get_step_negative_step(self, db: Database, client: TestClient) -> None:
        """When step is negative but not -1 (line 352-353)."""
        runs.create_run(db, "cbce3")
        runs.set_context(db, "cbce3", 0, {})
        runs.set_trace_info(db, "cbce3", 0, "texts", dtype="text", last=0.0, last_step=0)
        sequences.write_sequence_step(db, "cbce3", 0, "texts", 0, {"data": "hi"})

        resp = client.post(
            f"{BASE}/runs/cbce3/texts/get-step/",
            json=[{"name": "texts", "context": {}}],
            params={"record_step": -5},
        )
        assert resp.status_code == 200

    def test_image_get_batch_with_index_range(self, db: Database, client: TestClient) -> None:
        """Images use_list=True: test with index_range (lines 358-361)."""
        runs.create_run(db, "cbce4")
        runs.set_context(db, "cbce4", 0, {})
        runs.set_trace_info(db, "cbce4", 0, "images", dtype="image", last=0.0, last_step=0)
        sequences.write_sequence_step(
            db,
            "cbce4",
            0,
            "images",
            0,
            [{"caption": f"img{i}", "width": 32, "height": 32} for i in range(10)],
        )

        resp = client.post(
            f"{BASE}/runs/cbce4/images/get-batch/",
            json=[{"name": "images", "context": {}}],
            params={"index_range": "2,5", "index_density": 2},
        )
        assert resp.status_code == 200

    def test_image_get_step_with_index_range(self, db: Database, client: TestClient) -> None:
        """Images use_list=True get-step: index_range (lines 358-361 in get-step)."""
        runs.create_run(db, "cbce5")
        runs.set_context(db, "cbce5", 0, {})
        runs.set_trace_info(db, "cbce5", 0, "images", dtype="image", last=0.0, last_step=0)
        sequences.write_sequence_step(
            db,
            "cbce5",
            0,
            "images",
            0,
            [{"caption": f"img{i}", "width": 32, "height": 32} for i in range(10)],
        )

        resp = client.post(
            f"{BASE}/runs/cbce5/images/get-step/",
            json=[{"name": "images", "context": {}}],
            params={"record_step": 0, "index_range": "1,4", "index_density": 2},
        )
        assert resp.status_code == 200


# --- Metric alignment with real matching context ---


class TestAlignmentMatching:
    def test_alignment_real_data(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "alm1")
        runs.set_context(db, "alm1", 0, {})
        runs.set_trace_info(db, "alm1", 0, "loss", dtype="float", last=0.1, last_step=4)
        runs.set_trace_info(db, "alm1", 0, "acc", dtype="float", last=0.9, last_step=4)
        for i in range(5):
            sequences.write_sequence_step(db, "alm1", 0, "loss", i, 1.0 - i * 0.2)
            sequences.write_sequence_step(db, "alm1", 0, "acc", i, i * 0.2)

        resp = client.post(
            f"{BASE}/runs/search/metric/align/",
            json={
                "align_by": "acc",
                "runs": [
                    {
                        "run_id": "alm1",
                        "traces": [{"name": "loss", "context": {}, "slice": [0, 5, 1]}],
                    },
                ],
            },
        )
        assert resp.status_code == 200
        assert len(resp.content) > 0
