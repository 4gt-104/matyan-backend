"""Extended run API tests — info with experiment/artifacts, log records, update experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.storage import entities, runs, sequences

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database

from .conftest import BASE


class TestRunInfoExtended:
    def test_info_with_experiment(self, db: Database, client: TestClient) -> None:
        exp = entities.create_experiment(db, "info-exp")
        runs.create_run(db, "rie1", experiment_id=exp["id"])

        resp = client.get(f"{BASE}/runs/rie1/info/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["props"]["experiment"]["name"] == "info-exp"

    def test_info_with_tags(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "rie2")
        tag = entities.create_tag(db, "info-tag")
        entities.add_tag_to_run(db, "rie2", tag["id"])

        resp = client.get(f"{BASE}/runs/rie2/info/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["props"]["tags"]) == 1

    def test_info_skip_system(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "rie3")
        runs.set_run_attrs(db, "rie3", (), {"hparams": {"lr": 0.01}, "__system_params": {"gpu": "a100"}})

        resp = client.get(f"{BASE}/runs/rie3/info/", params={"skip_system": "true"})
        assert resp.status_code == 200
        data = resp.json()
        assert "__system_params" not in data["params"]
        assert "hparams" in data["params"]

    def test_info_sequence_filter(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "rie4")
        runs.set_context(db, "rie4", 0, {})
        runs.set_trace_info(db, "rie4", 0, "loss", dtype="float", last=0.1)
        runs.set_trace_info(db, "rie4", 0, "train_images", dtype="image")

        resp = client.get(f"{BASE}/runs/rie4/info/", params={"sequence": "metric"})
        assert resp.status_code == 200
        data = resp.json()
        assert "metric" in data["traces"]

    def test_info_with_artifacts(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "rie5")
        runs.set_run_attrs(
            db,
            "rie5",
            ("__blobs__", "model.pt"),
            {
                "s3_key": "rie5/model.pt",
                "content_type": "application/octet-stream",
            },
        )

        resp = client.get(f"{BASE}/runs/rie5/info/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["artifacts"]) == 1
        assert "model.pt" in data["artifacts"][0]["name"]

    def test_info_hidden_dtypes_excluded(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "rie6")
        runs.set_context(db, "rie6", 0, {})
        runs.set_trace_info(db, "rie6", 0, "loss", dtype="float", last=0.1)
        runs.set_trace_info(db, "rie6", 0, "logs", dtype="logs", last=0.0)
        runs.set_trace_info(db, "rie6", 0, "__log_records", dtype="log_records", last=0.0)

        resp = client.get(f"{BASE}/runs/rie6/info/")
        assert resp.status_code == 200
        data = resp.json()
        metric_names = [t["name"] for t in data["traces"].get("metric", [])]
        assert "loss" in metric_names
        assert "logs" not in metric_names
        assert "__log_records" not in metric_names


class TestRunLogRecordsExtended:
    def test_log_record_levels(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "lre1")
        for i, level in enumerate([10, 20, 30, 40, 50]):
            sequences.write_sequence_step(
                db,
                "lre1",
                0,
                "__log_records",
                i,
                {
                    "message": f"msg {i}",
                    "log_level": level,
                    "timestamp": 1000.0 + i,
                },
            )

        resp = client.get(f"{BASE}/runs/lre1/log-records/")
        assert resp.status_code == 200

    def test_log_record_non_dict(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "lre2")
        sequences.write_sequence_step(db, "lre2", 0, "__log_records", 0, "plain string message")

        resp = client.get(f"{BASE}/runs/lre2/log-records/")
        assert resp.status_code == 200


class TestUpdateRunExperiment:
    def test_update_experiment(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "ure1")
        entities.create_experiment(db, "target-exp")

        resp = client.put(
            f"{BASE}/runs/ure1/",
            json={"experiment": "target-exp"},
        )
        assert resp.status_code == 200

    def test_update_description(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "ure2")
        resp = client.put(
            f"{BASE}/runs/ure2/",
            json={"description": "A new description"},
        )
        assert resp.status_code == 200
        meta = runs.get_run_meta(db, "ure2")
        assert meta["description"] == "A new description"


class TestRunLogsExtended:
    def test_logs_with_explicit_range(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "rle1")
        for i in range(10):
            sequences.write_sequence_step(db, "rle1", 0, "logs", i, f"line {i}")

        resp = client.get(f"{BASE}/runs/rle1/logs/", params={"record_range": "2,5"})
        assert resp.status_code == 200

    def test_logs_with_start_only(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "rle2")
        for i in range(10):
            sequences.write_sequence_step(db, "rle2", 0, "logs", i, f"line {i}")

        resp = client.get(f"{BASE}/runs/rle2/logs/", params={"record_range": "5,"})
        assert resp.status_code == 200

    def test_logs_with_end_only(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "rle3")
        for i in range(10):
            sequences.write_sequence_step(db, "rle3", 0, "logs", i, f"line {i}")

        resp = client.get(f"{BASE}/runs/rle3/logs/", params={"record_range": ",5"})
        assert resp.status_code == 200

    def test_log_records_with_start_only(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "rle4")
        for i in range(5):
            sequences.write_sequence_step(
                db,
                "rle4",
                0,
                "__log_records",
                i,
                {
                    "message": f"msg {i}",
                    "log_level": 20,
                    "timestamp": 1000.0 + i,
                },
            )

        resp = client.get(f"{BASE}/runs/rle4/log-records/", params={"record_range": "2,"})
        assert resp.status_code == 200

    def test_log_records_with_end_only(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "rle5")
        for i in range(5):
            sequences.write_sequence_step(
                db,
                "rle5",
                0,
                "__log_records",
                i,
                {
                    "message": f"msg {i}",
                    "log_level": 20,
                    "timestamp": 1000.0 + i,
                },
            )

        resp = client.get(f"{BASE}/runs/rle5/log-records/", params={"record_range": ",3"})
        assert resp.status_code == 200


class TestRemoveTagNotFound:
    def test_remove_tag_nonexistent_run(self, client: TestClient) -> None:
        resp = client.delete(f"{BASE}/runs/missing/tags/tag-id/")
        assert resp.status_code == 404


class TestRunInfoDuration:
    def test_info_with_duration(self, db: Database, client: TestClient) -> None:
        runs.create_run(db, "ridur1")
        runs.update_run_meta(db, "ridur1", duration=120.5)

        resp = client.get(f"{BASE}/runs/ridur1/info/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["props"]["end_time"] is not None
