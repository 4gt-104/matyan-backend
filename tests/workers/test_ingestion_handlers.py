"""Tests for workers/ingestion.py handler functions.

Tests call the handler functions directly with a real FDB database,
bypassing Kafka entirely.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from matyan_api_models.context import context_to_id
from matyan_api_models.kafka import IngestionMessage

from matyan_backend.storage import entities, indexes, runs, sequences
from matyan_backend.storage.indexes import is_run_deleted
from matyan_backend.workers.ingestion import (
    IngestionWorker,
    _handle_add_tag,
    _handle_blob_ref,
    _handle_create_run,
    _handle_delete_run,
    _handle_finish_run,
    _handle_log_custom_object,
    _handle_log_hparams,
    _handle_log_metric,
    _handle_log_record,
    _handle_log_terminal_line,
    _handle_remove_tag,
    _handle_set_run_property,
    _next_step,
    _parse_client_ts,
)

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database

# ruff: noqa: SLF001


def _msg(msg_type: str, run_id: str, **payload: object) -> IngestionMessage:
    return IngestionMessage(
        type=msg_type,
        run_id=run_id,
        timestamp=datetime.now(UTC),
        payload=dict(payload),
    )


class TestParseClientTs:
    def test_valid_iso(self) -> None:
        ts = _parse_client_ts({"client_datetime": "2025-01-15T12:00:00+00:00"})
        assert ts is not None
        assert isinstance(ts, float)

    def test_missing_key(self) -> None:
        assert _parse_client_ts({}) is None

    def test_invalid_format(self) -> None:
        assert _parse_client_ts({"client_datetime": "not-a-date"}) is None

    def test_empty_string(self) -> None:
        assert _parse_client_ts({"client_datetime": ""}) is None


class TestNextStep:
    def test_empty_sequence_returns_zero(self, db: Database) -> None:
        runs.create_run(db, "ns1")
        assert _next_step(db, "ns1", 0, "loss") == 0

    def test_after_writes(self, db: Database) -> None:
        runs.create_run(db, "ns2")
        sequences.write_sequence_step(db, "ns2", 0, "loss", 0, 1.0)
        sequences.write_sequence_step(db, "ns2", 0, "loss", 1, 2.0)
        assert _next_step(db, "ns2", 0, "loss") == 2


class TestHandleCreateRun:
    def test_new_run(self, db: Database) -> None:
        _handle_create_run(db, _msg("create_run", "hcr1"))
        run = runs.get_run(db, "hcr1")
        assert run is not None
        assert run["active"] is True

    def test_force_resume_existing(self, db: Database) -> None:
        runs.create_run(db, "hcr2")
        runs.update_run_meta(db, "hcr2", active=False)

        _handle_create_run(db, _msg("create_run", "hcr2", force_resume=True))
        run = runs.get_run(db, "hcr2")
        assert run is not None
        assert run["active"] is True

    def test_force_resume_nonexistent_creates(self, db: Database) -> None:
        _handle_create_run(db, _msg("create_run", "hcr3", force_resume=True))
        assert runs.get_run(db, "hcr3") is not None

    def test_with_client_datetime(self, db: Database) -> None:
        _handle_create_run(
            db,
            _msg("create_run", "hcr4", client_datetime="2025-06-01T10:00:00+00:00"),
        )
        meta = runs.get_run_meta(db, "hcr4")
        assert meta.get("client_start_ts") is not None


class TestHandleLogMetric:
    def test_basic(self, db: Database) -> None:
        runs.create_run(db, "hlm1")
        _handle_log_metric(
            db,
            _msg("log_metric", "hlm1", name="loss", value=0.5, step=0, context={}),
        )
        data = sequences.read_sequence(db, "hlm1", 0, "loss")
        assert data["val"] == [0.5]

    def test_auto_step(self, db: Database) -> None:
        runs.create_run(db, "hlm2")
        _handle_log_metric(db, _msg("log_metric", "hlm2", name="acc", value=0.9, context={}))
        _handle_log_metric(db, _msg("log_metric", "hlm2", name="acc", value=0.95, context={}))
        data = sequences.read_sequence(db, "hlm2", 0, "acc")
        assert len(data["val"]) == 2
        assert data["steps"] == [0, 1]

    def test_with_context(self, db: Database) -> None:
        runs.create_run(db, "hlm3")
        ctx = {"subset": "val"}
        _handle_log_metric(
            db,
            _msg("log_metric", "hlm3", name="loss", value=0.3, step=0, context=ctx),
        )
        ctx_id = context_to_id(ctx)
        data = sequences.read_sequence(db, "hlm3", ctx_id, "loss")
        assert data["val"] == [0.3]

    def test_sets_trace_info(self, db: Database) -> None:
        runs.create_run(db, "hlm4")
        _handle_log_metric(
            db,
            _msg("log_metric", "hlm4", name="loss", value=0.1, step=5, context={}, dtype="float"),
        )
        traces = runs.get_run_traces_info(db, "hlm4")
        assert len(traces) == 1
        assert traces[0]["name"] == "loss"
        assert traces[0]["last"] == 0.1
        assert traces[0]["last_step"] == 5


class TestHandleLogHparams:
    def test_flat(self, db: Database) -> None:
        runs.create_run(db, "hlh1")
        _handle_log_hparams(
            db,
            _msg("log_hparams", "hlh1", value={"hparams": {"lr": 0.01, "bs": 32}}),
        )
        attrs = runs.get_run_attrs(db, "hlh1", ("hparams",))
        assert attrs == {"lr": 0.01, "bs": 32}

    def test_nested_unwrapping(self, db: Database) -> None:
        runs.create_run(db, "hlh2")
        _handle_log_hparams(
            db,
            _msg("log_hparams", "hlh2", value={"hparams": {"model": {"type": "cnn"}}}),
        )
        val = runs.get_run_attrs(db, "hlh2", ("hparams", "model"))
        assert val == {"type": "cnn"}

    def test_empty_value_ignored(self, db: Database) -> None:
        runs.create_run(db, "hlh3")
        _handle_log_hparams(db, _msg("log_hparams", "hlh3", value={}))
        assert runs.get_run_attrs(db, "hlh3") is None


class TestHandleFinishRun:
    def test_basic(self, db: Database) -> None:
        runs.create_run(db, "hfr1")
        _handle_finish_run(db, _msg("finish_run", "hfr1"))
        meta = runs.get_run_meta(db, "hfr1")
        assert meta["active"] is False
        assert meta.get("finalized_at") is not None

    def test_with_client_timestamps(self, db: Database) -> None:
        runs.create_run(db, "hfr2")
        runs.update_run_meta(db, "hfr2", client_start_ts=1000.0)
        _handle_finish_run(
            db,
            _msg("finish_run", "hfr2", client_datetime="2025-06-01T10:00:10+00:00"),
        )
        meta = runs.get_run_meta(db, "hfr2")
        assert meta["active"] is False
        assert meta.get("duration") is not None
        assert meta["duration"] >= 0


class TestHandleLogCustomObject:
    def test_basic(self, db: Database) -> None:
        runs.create_run(db, "hlco1")
        _handle_log_custom_object(
            db,
            _msg(
                "log_custom_object",
                "hlco1",
                name="images",
                value={"data": "base64data", "width": 100, "height": 100},
                step=0,
                dtype="image",
                context={},
            ),
        )
        data = sequences.read_sequence(db, "hlco1", 0, "images")
        assert len(data["val"]) == 1
        assert data["val"][0]["width"] == 100


class TestHandleBlobRef:
    def test_stores_blob_metadata(self, db: Database) -> None:
        runs.create_run(db, "hbr1")
        _handle_blob_ref(
            db,
            _msg(
                "blob_ref",
                "hbr1",
                artifact_path="models/weights.pt",
                s3_key="hbr1/weights.pt",
                content_type="application/octet-stream",
            ),
        )
        blobs = runs.get_run_attrs(db, "hbr1", ("__blobs__", "models/weights.pt"))
        assert blobs["s3_key"] == "hbr1/weights.pt"


class TestHandleSetRunProperty:
    def test_set_name(self, db: Database) -> None:
        runs.create_run(db, "hsrp1")
        _handle_set_run_property(db, _msg("set_run_property", "hsrp1", name="My Run"))
        meta = runs.get_run_meta(db, "hsrp1")
        assert meta["name"] == "My Run"

    def test_set_experiment_creates_if_needed(self, db: Database) -> None:
        runs.create_run(db, "hsrp2")
        _handle_set_run_property(db, _msg("set_run_property", "hsrp2", experiment="new-exp"))
        meta = runs.get_run_meta(db, "hsrp2")
        assert meta.get("experiment_id") is not None

    def test_auto_creates_run_if_missing(self, db: Database) -> None:
        _handle_set_run_property(db, _msg("set_run_property", "hsrp3", name="Auto"))
        run = runs.get_run(db, "hsrp3")
        assert run is not None

    def test_set_archived(self, db: Database) -> None:
        runs.create_run(db, "hsrp4")
        _handle_set_run_property(db, _msg("set_run_property", "hsrp4", archived=True))
        meta = runs.get_run_meta(db, "hsrp4")
        assert meta["is_archived"] is True


class TestHandleAddTag:
    def test_add_new_tag(self, db: Database) -> None:
        runs.create_run(db, "hat1")
        _handle_add_tag(db, _msg("add_tag", "hat1", tag_name="prod"))
        tags = entities.get_tags_for_run(db, "hat1")
        assert any(t["name"] == "prod" for t in tags)

    def test_add_tag_auto_creates_run(self, db: Database) -> None:
        _handle_add_tag(db, _msg("add_tag", "hat2", tag_name="dev"))
        assert runs.get_run(db, "hat2") is not None


class TestHandleRemoveTag:
    def test_remove_existing(self, db: Database) -> None:
        runs.create_run(db, "hrt1")
        tag = entities.create_tag(db, "removeme")
        entities.add_tag_to_run(db, "hrt1", tag["id"])
        runs.add_tag_to_run(db, "hrt1", tag["id"])

        _handle_remove_tag(db, _msg("remove_tag", "hrt1", tag_name="removeme"))
        tags = entities.get_tags_for_run(db, "hrt1")
        assert not any(t["name"] == "removeme" for t in tags)

    def test_remove_nonexistent_tag(self, db: Database) -> None:
        runs.create_run(db, "hrt2")
        _handle_remove_tag(db, _msg("remove_tag", "hrt2", tag_name="ghost"))


class TestHandleLogTerminalLine:
    def test_stores_line(self, db: Database) -> None:
        runs.create_run(db, "htl1")
        _handle_log_terminal_line(db, _msg("log_terminal_line", "htl1", line="hello world", step=0))
        data = sequences.read_sequence(db, "htl1", 0, "logs")
        assert data["val"] == ["hello world"]


class TestHandleLogRecord:
    def test_stores_record(self, db: Database) -> None:
        runs.create_run(db, "hlr1")
        _handle_log_record(
            db,
            _msg(
                "log_record",
                "hlr1",
                message="Training started",
                level=20,
                timestamp=1000.0,
            ),
        )
        data = sequences.read_sequence(db, "hlr1", 0, "__log_records")
        assert len(data["val"]) == 1
        assert data["val"][0]["message"] == "Training started"
        assert data["val"][0]["log_level"] == 20


class TestHandleCreateRunResumeClientTs:
    def test_force_resume_with_client_ts(self, db: Database) -> None:
        runs.create_run(db, "hcrct1")
        runs.update_run_meta(db, "hcrct1", active=False)

        _handle_create_run(
            db,
            _msg("create_run", "hcrct1", force_resume=True, client_datetime="2025-06-01T10:00:00+00:00"),
        )
        meta = runs.get_run_meta(db, "hcrct1")
        assert meta["active"] is True
        assert meta.get("client_start_ts") is not None


class TestHandleLogCustomObjectAutoStep:
    def test_auto_step(self, db: Database) -> None:
        runs.create_run(db, "hlcoas1")
        _handle_log_custom_object(
            db,
            _msg("log_custom_object", "hlcoas1", name="figures", value={"data": "{}"}, dtype="figure", context={}),
        )
        _handle_log_custom_object(
            db,
            _msg("log_custom_object", "hlcoas1", name="figures", value={"data": "{}"}, dtype="figure", context={}),
        )
        data = sequences.read_sequence(db, "hlcoas1", 0, "figures")
        assert len(data["val"]) == 2
        assert data["steps"] == [0, 1]


class TestHandleSetRunPropertyDescription:
    def test_set_description(self, db: Database) -> None:
        runs.create_run(db, "hsrpd1")
        _handle_set_run_property(db, _msg("set_run_property", "hsrpd1", description="A desc"))
        meta = runs.get_run_meta(db, "hsrpd1")
        assert meta["description"] == "A desc"


class TestHandleDeleteRun:
    def test_deletes_run_and_writes_tombstone(self, db: Database) -> None:
        runs.create_run(db, "hdr1")
        assert runs.get_run(db, "hdr1") is not None

        _handle_delete_run(db, _msg("delete_run", "hdr1"))

        assert runs.get_run(db, "hdr1") is None
        assert is_run_deleted(db, "hdr1") is True

    def test_delete_clears_sequences(self, db: Database) -> None:
        runs.create_run(db, "hdr2")
        _handle_log_metric(db, _msg("log_metric", "hdr2", name="loss", value=0.5, step=0, context={}))
        data = sequences.read_sequence(db, "hdr2", 0, "loss")
        assert len(data["val"]) == 1

        _handle_delete_run(db, _msg("delete_run", "hdr2"))
        data = sequences.read_sequence(db, "hdr2", 0, "loss")
        assert data["val"] == []

    def test_delete_nonexistent_run_no_error(self, db: Database) -> None:
        _handle_delete_run(db, _msg("delete_run", "hdr_ghost"))
        assert is_run_deleted(db, "hdr_ghost") is True


class TestTombstoneSkip:
    def test_late_message_skipped_after_delete(self, db: Database) -> None:
        """After delete_run, subsequent messages for the same run are skipped."""
        runs.create_run(db, "ts1")
        _handle_delete_run(db, _msg("delete_run", "ts1"))

        worker = IngestionWorker()
        worker._db = db
        msg = _msg("log_metric", "ts1", name="loss", value=0.1, step=0, context={})
        worker._handle_run_group("ts1", [msg])

        assert runs.get_run(db, "ts1") is None
        data = sequences.read_sequence(db, "ts1", 0, "loss")
        assert data["val"] == []

    def test_create_run_skipped_after_tombstone(self, db: Database) -> None:
        runs.create_run(db, "ts2")
        _handle_delete_run(db, _msg("delete_run", "ts2"))

        worker = IngestionWorker()
        worker._db = db
        worker._handle_run_group("ts2", [_msg("create_run", "ts2")])

        assert runs.get_run(db, "ts2") is None


class TestMessageOrdering:
    def test_create_log_delete_log_sequence(self, db: Database) -> None:
        """Simulate Kafka ordering: create -> log -> delete -> late log."""
        worker = IngestionWorker()
        worker._db = db

        worker._handle_run_group("mo1", [_msg("create_run", "mo1")])
        assert runs.get_run(db, "mo1") is not None

        worker._handle_run_group(
            "mo1",
            [_msg("log_metric", "mo1", name="loss", value=0.5, step=0, context={})],
        )
        data = sequences.read_sequence(db, "mo1", 0, "loss")
        assert data["val"] == [0.5]

        worker._handle_delete(_msg("delete_run", "mo1"))
        assert runs.get_run(db, "mo1") is None
        assert is_run_deleted(db, "mo1") is True

        worker._handle_run_group(
            "mo1",
            [_msg("log_metric", "mo1", name="loss", value=0.1, step=1, context={})],
        )
        assert runs.get_run(db, "mo1") is None
        data = sequences.read_sequence(db, "mo1", 0, "loss")
        assert data["val"] == []


class TestUnknownMessageType:
    def test_unknown_type_logged(self, db: Database) -> None:
        worker = IngestionWorker()
        worker._db = db
        msg = _msg("totally_unknown", "uk1")
        worker._handle_run_group("uk1", [msg])

    def test_known_type_dispatches(self, db: Database) -> None:
        worker = IngestionWorker()
        worker._db = db
        msg = _msg("create_run", "uk2")
        worker._handle_run_group("uk2", [msg])
        assert runs.get_run(db, "uk2") is not None


class TestLogMetricCreatesTraceIndex:
    def test_log_metric_creates_trace_index(self, db: Database) -> None:
        """_handle_log_metric should write a trace-name index entry."""
        runs.create_run(db, "tidx1")
        _handle_log_metric(
            db,
            _msg("log_metric", "tidx1", name="loss", value=0.5, step=0, context={}),
        )
        assert "tidx1" in indexes.lookup_by_trace_name(db, "loss")

    def test_multiple_metrics_indexed(self, db: Database) -> None:
        runs.create_run(db, "tidx2")
        _handle_log_metric(
            db,
            _msg("log_metric", "tidx2", name="loss", value=0.5, step=0, context={}),
        )
        _handle_log_metric(
            db,
            _msg("log_metric", "tidx2", name="acc", value=0.9, step=0, context={}),
        )
        assert "tidx2" in indexes.lookup_by_trace_name(db, "loss")
        assert "tidx2" in indexes.lookup_by_trace_name(db, "acc")
