"""Tests for backup export and direct restore round-trip."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from matyan_api_models.backup import BackupManifest

from matyan_backend.backup.export_entities import export_entities
from matyan_backend.backup.export_run import export_run
from matyan_backend.backup.restore_direct import _restore_entities, _restore_run
from matyan_backend.storage import entities, indexes, runs, sequences

if TYPE_CHECKING:
    from pathlib import Path

    from matyan_backend.fdb_types import Database


def _seed_run(db: Database, run_hash: str, *, experiment_name: str | None = None) -> None:
    """Create a run with some representative data for testing."""
    exp_id = None
    if experiment_name:
        try:
            exp = entities.create_experiment(db, experiment_name)
            exp_id = exp["id"]
        except ValueError:
            exp = entities.get_experiment_by_name(db, experiment_name)
            exp_id = exp["id"] if exp else None

    runs.create_run(db, run_hash, name=f"Test {run_hash}", experiment_id=exp_id)
    runs.set_run_attrs(db, run_hash, ("hparams",), {"lr": 0.01, "batch_size": 32})
    runs.set_run_attrs(db, run_hash, ("custom_key",), "custom_value")

    ctx_id = 0
    runs.set_context(db, run_hash, ctx_id, {})
    for step in range(5):
        sequences.write_sequence_step(db, run_hash, ctx_id, "loss", step, 1.0 - step * 0.1)
        sequences.write_sequence_step(db, run_hash, ctx_id, "accuracy", step, step * 0.2)
    runs.set_trace_info(db, run_hash, ctx_id, "loss", dtype="float", last=0.6, last_step=4)
    runs.set_trace_info(db, run_hash, ctx_id, "accuracy", dtype="float", last=0.8, last_step=4)


class TestExportRun:
    def test_export_creates_expected_files(self, db: Database, tmp_path: Path) -> None:
        _seed_run(db, "exp_r1")
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        (backup_dir / "runs").mkdir()

        seq_count = export_run(db, "exp_r1", backup_dir)

        run_dir = backup_dir / "runs" / "exp_r1"
        assert run_dir.is_dir()
        assert (run_dir / "run.json").exists()
        assert (run_dir / "attrs.json").exists()
        assert (run_dir / "traces.json").exists()
        assert (run_dir / "contexts.json").exists()
        assert (run_dir / "sequences.jsonl").exists()
        assert seq_count == 10  # 5 steps * 2 metrics

    def test_exported_metadata_is_correct(self, db: Database, tmp_path: Path) -> None:
        _seed_run(db, "exp_r2")
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        (backup_dir / "runs").mkdir()
        export_run(db, "exp_r2", backup_dir)

        meta = json.loads((backup_dir / "runs" / "exp_r2" / "run.json").read_text())
        assert meta["name"] == "Test exp_r2"
        assert meta["active"] is True

        attrs = json.loads((backup_dir / "runs" / "exp_r2" / "attrs.json").read_text())
        assert attrs["hparams"]["lr"] == 0.01
        assert attrs["custom_key"] == "custom_value"

    def test_exported_sequences_content(self, db: Database, tmp_path: Path) -> None:
        _seed_run(db, "exp_r3")
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        (backup_dir / "runs").mkdir()
        export_run(db, "exp_r3", backup_dir)

        with (backup_dir / "runs" / "exp_r3" / "sequences.jsonl").open() as f:
            records = [json.loads(line) for line in f]
        loss_records = [r for r in records if r["name"] == "loss"]
        assert len(loss_records) == 5
        assert loss_records[0]["step"] == 0
        assert loss_records[0]["value"] == 1.0


class TestExportEntities:
    def test_export_entities(self, db: Database, tmp_path: Path) -> None:
        entities.create_experiment(db, "test-exp", description="desc")
        entities.create_tag(db, "test-tag", color="red")
        entities.create_dashboard(db, "test-dash")

        counts = export_entities(db, tmp_path)
        assert counts["experiment"] >= 1
        assert counts["tag"] >= 1
        assert counts["dashboard"] >= 1

        with (tmp_path / "entities.jsonl").open() as f:
            records = [json.loads(line) for line in f]
        exp_records = [r for r in records if r["entity_type"] == "experiment"]
        assert any(r["data"]["name"] == "test-exp" for r in exp_records)


class TestDirectRestoreRoundTrip:
    def test_export_and_restore_run(self, db: Database, tmp_path: Path) -> None:
        _seed_run(db, "rt_r1", experiment_name="rt-exp")

        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        (backup_dir / "runs").mkdir()
        export_run(db, "rt_r1", backup_dir)
        export_entities(db, backup_dir)

        runs.delete_run(db, "rt_r1")
        assert runs.get_run(db, "rt_r1") is None

        _restore_entities(db, backup_dir)
        run_dir = backup_dir / "runs" / "rt_r1"
        seq_count = _restore_run(db, "rt_r1", run_dir)

        restored = runs.get_run(db, "rt_r1")
        assert restored is not None
        assert restored["name"] == "Test rt_r1"

        attrs = runs.get_run_attrs(db, "rt_r1")
        assert attrs["hparams"]["lr"] == 0.01
        assert attrs["custom_key"] == "custom_value"

        traces = runs.get_run_traces_info(db, "rt_r1")
        trace_names = {t["name"] for t in traces}
        assert "loss" in trace_names
        assert "accuracy" in trace_names

        loss_data = sequences.read_sequence(db, "rt_r1", 0, "loss")
        assert len(loss_data["steps"]) == 5
        assert loss_data["val"][0] == 1.0

        assert seq_count == 10

    def test_manifest_round_trip(self, db: Database, tmp_path: Path) -> None:
        _seed_run(db, "mrt_r1")
        _seed_run(db, "mrt_r2")

        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        (backup_dir / "runs").mkdir()
        export_run(db, "mrt_r1", backup_dir)
        export_run(db, "mrt_r2", backup_dir)

        manifest = BackupManifest(
            run_count=2,
            run_hashes=["mrt_r1", "mrt_r2"],
        )
        manifest.write(backup_dir)
        loaded = BackupManifest.read(backup_dir)
        assert loaded.run_count == 2
        assert loaded.run_hashes == ["mrt_r1", "mrt_r2"]
        assert loaded.validate(backup_dir) == []


class TestSelectiveBackup:
    def test_lookup_by_experiment(self, db: Database) -> None:
        _seed_run(db, "sel_r1", experiment_name="sel-exp")
        _seed_run(db, "sel_r2", experiment_name="sel-exp")
        _seed_run(db, "sel_r3", experiment_name="other-exp")

        hashes = indexes.lookup_by_experiment(db, "sel-exp")
        assert "sel_r1" in hashes
        assert "sel_r2" in hashes
        assert "sel_r3" not in hashes

    def test_lookup_by_created_at(self, db: Database) -> None:
        _seed_run(db, "sel_r4")
        hashes = indexes.lookup_by_created_at(db, start=0.0)
        assert "sel_r4" in hashes
