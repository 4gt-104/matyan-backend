"""API-level tests for run endpoints (CRUD, tags, notes, streaming)."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

from matyan_backend.storage import runs
from matyan_backend.storage.sequences import write_sequence_step

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database

from .conftest import BASE

URL = f"{BASE}/runs"


# ── Binary stream helpers ───────────────────────────────────────────────────


def _decode_stream_pairs(data: bytes) -> list[tuple[bytes, bytes]]:
    """Parse the binary stream format into (key, value) pairs."""
    pairs: list[tuple[bytes, bytes]] = []
    off = 0
    while off < len(data):
        (key_len,) = struct.unpack_from("<I", data, off)
        off += 4
        key = data[off : off + key_len]
        off += key_len
        (val_len,) = struct.unpack_from("<I", data, off)
        off += 4
        val = data[off : off + val_len]
        off += val_len
        pairs.append((key, val))
    return pairs


def _decode_value(raw: bytes) -> bool | Any | str | bytes | None:  # noqa: ANN401, PLR0911
    """Minimal decoder for the streaming value format."""
    if not raw:
        return None
    tag = raw[0]
    payload = raw[1:]
    if tag == 0:
        return None
    if tag == 1:
        return bool(payload[0])
    if tag == 2:
        return struct.unpack("<q", payload)[0]
    if tag == 3:
        return struct.unpack("<d", payload)[0]
    if tag == 4:
        return payload.decode("utf-8")
    if tag == 5:
        return payload
    if tag == 6:
        return "<ARRAY>"
    if tag == 7:
        return "<OBJECT>"
    return raw


# ── Fixtures ────────────────────────────────────────────────────────────────


def _seed_run(db: Database, run_hash: str = "test-run", **kw: Any) -> dict:  # noqa: ANN401
    return runs.create_run(db, run_hash, **kw)


# ── Run Info ────────────────────────────────────────────────────────────────


class TestRunInfo:
    def test_get_info(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "info-run", name="Info Run")
        resp = client.get(f"{URL}/info-run/info/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["props"]["name"] == "Info Run"
        assert "params" in body
        assert "traces" in body

    def test_get_info_nonexistent(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/no-such-run/info/")
        assert resp.status_code == 404


# ── Run Update ──────────────────────────────────────────────────────────────


class TestRunUpdate:
    def test_update_name(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "upd-run")
        resp = client.put(f"{URL}/upd-run/", json={"name": "Renamed"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "OK"

        info = client.get(f"{URL}/upd-run/info/").json()
        assert info["props"]["name"] == "Renamed"

    def test_update_archived(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "arc-run")
        client.put(f"{URL}/arc-run/", json={"archived": True})
        info = client.get(f"{URL}/arc-run/info/").json()
        assert info["props"]["archived"] is True

    def test_update_nonexistent(self, client: TestClient) -> None:
        resp = client.put(f"{URL}/no-run/", json={"name": "x"})
        assert resp.status_code == 404

    def test_finish_active_run(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "active-run")
        info = client.get(f"{URL}/active-run/info/").json()
        assert info["props"]["active"] is True
        assert info["props"]["end_time"] is None

        resp = client.put(f"{URL}/active-run/", json={"active": False})
        assert resp.status_code == 200

        info = client.get(f"{URL}/active-run/info/").json()
        assert info["props"]["active"] is False
        assert info["props"]["end_time"] is not None

    def test_finish_already_finalized_preserves_end_time(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "fin-run")
        runs.update_run_meta(db, "fin-run", active=False, finalized_at=1000.0)
        info = client.get(f"{URL}/fin-run/info/").json()
        original_end = info["props"]["end_time"]

        client.put(f"{URL}/fin-run/", json={"active": False})
        info = client.get(f"{URL}/fin-run/info/").json()
        assert info["props"]["end_time"] == original_end

    def test_reactivate_run(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "react-run")
        runs.update_run_meta(db, "react-run", active=False, finalized_at=1000.0)

        resp = client.put(f"{URL}/react-run/", json={"active": True})
        assert resp.status_code == 200

        info = client.get(f"{URL}/react-run/info/").json()
        assert info["props"]["active"] is True
        assert info["props"]["end_time"] is None


# ── Run Delete ──────────────────────────────────────────────────────────────


class TestRunDelete:
    def test_delete_single(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "del-run")
        resp = client.delete(f"{URL}/del-run/")
        assert resp.status_code == 200
        assert runs.is_pending_deletion(db, "del-run") is True

    def test_delete_nonexistent(self, client: TestClient) -> None:
        resp = client.delete(f"{URL}/no-run/")
        assert resp.status_code == 404

    def test_delete_batch(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "batch-a")
        _seed_run(db, "batch-b")
        resp = client.post(f"{URL}/delete-batch/", json=["batch-a", "batch-b"])
        assert resp.status_code == 200
        assert runs.is_pending_deletion(db, "batch-a") is True
        assert runs.is_pending_deletion(db, "batch-b") is True

    def test_pending_deletion_run_hidden_from_search(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "hidden-run")
        runs.mark_pending_deletion(db, "hidden-run")
        resp = client.get(f"{URL}/search/run/")
        pairs = _decode_stream_pairs(resp.content)
        run_hashes = [
            _decode_value(v)
            for k, v in pairs
            if k.endswith(b"\xfehash")
        ]
        assert "hidden-run" not in run_hashes


class TestRunArchiveBatch:
    def test_archive_batch(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "arc-a")
        _seed_run(db, "arc-b")
        resp = client.post(f"{URL}/archive-batch/", json=["arc-a", "arc-b"], params={"archive": True})
        assert resp.status_code == 200

        info_a = client.get(f"{URL}/arc-a/info/").json()
        info_b = client.get(f"{URL}/arc-b/info/").json()
        assert info_a["props"]["archived"] is True
        assert info_b["props"]["archived"] is True

    def test_unarchive_batch(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "unarc")
        runs.update_run_meta(db, "unarc", is_archived=True)
        client.post(f"{URL}/archive-batch/", json=["unarc"], params={"archive": False})
        info = client.get(f"{URL}/unarc/info/").json()
        assert info["props"]["archived"] is False


class TestRunFinishBatch:
    def test_finish_batch(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "fin-a")
        _seed_run(db, "fin-b")
        resp = client.post(f"{URL}/finish-batch/", json=["fin-a", "fin-b"])
        assert resp.status_code == 200

        info_a = client.get(f"{URL}/fin-a/info/").json()
        info_b = client.get(f"{URL}/fin-b/info/").json()
        assert info_a["props"]["active"] is False
        assert info_a["props"]["end_time"] is not None
        assert info_b["props"]["active"] is False
        assert info_b["props"]["end_time"] is not None

    def test_finish_batch_preserves_existing_finalized_at(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "fin-pre")
        runs.update_run_meta(db, "fin-pre", active=False, finalized_at=1000.0)
        info_before = client.get(f"{URL}/fin-pre/info/").json()
        end_before = info_before["props"]["end_time"]

        client.post(f"{URL}/finish-batch/", json=["fin-pre"])
        info_after = client.get(f"{URL}/fin-pre/info/").json()
        assert info_after["props"]["end_time"] == end_before

    def test_finish_batch_empty(self, client: TestClient) -> None:
        resp = client.post(f"{URL}/finish-batch/", json=[])
        assert resp.status_code == 200


# ── Run ↔ Tags ──────────────────────────────────────────────────────────────


class TestRunTags:
    def test_add_and_remove_tag(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "tagged-run")
        resp = client.post(f"{URL}/tagged-run/tags/new/", json={"tag_name": "important"})
        assert resp.status_code == 200
        tag_id = resp.json()["tag_id"]

        info = client.get(f"{URL}/tagged-run/info/").json()
        tag_names = {t["name"] for t in info["props"]["tags"]}
        assert "important" in tag_names

        del_resp = client.delete(f"{URL}/tagged-run/tags/{tag_id}/")
        assert del_resp.status_code == 200
        assert del_resp.json()["removed"] is True

    def test_add_tag_nonexistent_run(self, client: TestClient) -> None:
        resp = client.post(f"{URL}/no-run/tags/new/", json={"tag_name": "x"})
        assert resp.status_code == 404


# ── Notes ───────────────────────────────────────────────────────────────────


class TestRunNotes:
    def test_note_crud(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "noted-run")

        create_resp = client.post(f"{URL}/noted-run/note/", json={"content": "my note"})
        assert create_resp.status_code == 201
        note_id = create_resp.json()["id"]

        get_resp = client.get(f"{URL}/noted-run/note/{note_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["content"] == "my note"

        client.put(f"{URL}/noted-run/note/{note_id}", json={"content": "updated"})
        updated = client.get(f"{URL}/noted-run/note/{note_id}").json()
        assert updated["content"] == "updated"

        del_resp = client.delete(f"{URL}/noted-run/note/{note_id}")
        assert del_resp.status_code == 200
        assert client.get(f"{URL}/noted-run/note/{note_id}").status_code == 404

    def test_list_notes(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "multi-note")
        client.post(f"{URL}/multi-note/note/", json={"content": "one"})
        client.post(f"{URL}/multi-note/note/", json={"content": "two"})
        resp = client.get(f"{URL}/multi-note/note/")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


# ── Logs ──────────────────────────────────────────────────────────────────


def _seed_logs(db: Database, run_hash: str) -> None:
    runs.set_context(db, run_hash, 0, {})
    for step in range(10):
        write_sequence_step(db, run_hash, 0, "logs", step, f"line {step}", timestamp=1000.0 + step)
    runs.set_trace_info(db, run_hash, 0, "logs", dtype="logs", last=0.0, last_step=9)


def _seed_log_records(db: Database, run_hash: str) -> None:
    runs.set_context(db, run_hash, 0, {})
    levels = [20, 20, 30, 40, 20, 20, 30, 20, 20, 20]
    for step in range(10):
        record = {
            "message": f"Record {step}",
            "log_level": levels[step],
            "timestamp": 2000.0 + step,
            "args": {"key": step} if step == 3 else None,
            "logger_info": ["test.py", 10 + step],
        }
        write_sequence_step(db, run_hash, 0, "__log_records", step, record, timestamp=2000.0 + step)
    runs.set_trace_info(db, run_hash, 0, "__log_records", dtype="log_records", last=0.0, last_step=9)


class TestRunLogs:
    def test_logs_returns_stream(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "log-run")
        _seed_logs(db, "log-run")
        resp = client.get(f"{URL}/log-run/logs/")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        pairs = _decode_stream_pairs(resp.content)
        assert len(pairs) > 0

    def test_logs_empty_run(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "nolog-run")
        resp = client.get(f"{URL}/nolog-run/logs/")
        assert resp.status_code == 200
        assert resp.content == b""

    def test_logs_record_range(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "rng-log")
        _seed_logs(db, "rng-log")
        resp = client.get(f"{URL}/rng-log/logs/", params={"record_range": "2:5"})
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        assert len(pairs) == 3

    def test_logs_default_tail(self, client: TestClient, db: Database) -> None:
        """With 10 lines and no range, all 10 should be returned (< DEFAULT_TAIL of 200)."""
        _seed_run(db, "tail-log")
        _seed_logs(db, "tail-log")
        resp = client.get(f"{URL}/tail-log/logs/")
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        assert len(pairs) == 10

    def test_logs_values_contain_strings(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "val-log")
        _seed_logs(db, "val-log")
        resp = client.get(f"{URL}/val-log/logs/", params={"record_range": "0:2"})
        pairs = _decode_stream_pairs(resp.content)
        for _, val_bytes in pairs:
            val = _decode_value(val_bytes)
            assert isinstance(val, str)
            assert "line" in val


class TestRunLogRecords:
    def test_log_records_returns_stream(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "lr-run")
        _seed_log_records(db, "lr-run")
        resp = client.get(f"{URL}/lr-run/log-records/")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"
        pairs = _decode_stream_pairs(resp.content)
        assert len(pairs) > 0

    def test_log_records_empty_run(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "nolr-run")
        resp = client.get(f"{URL}/nolr-run/log-records/")
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        assert len(pairs) == 1

    def test_log_records_includes_count(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "cnt-lr")
        _seed_log_records(db, "cnt-lr")
        resp = client.get(f"{URL}/cnt-lr/log-records/")
        pairs = _decode_stream_pairs(resp.content)
        first_key = pairs[0][0]
        assert b"log_records_count" in first_key

    def test_log_records_record_range(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "rng-lr")
        _seed_log_records(db, "rng-lr")
        resp_full = client.get(f"{URL}/rng-lr/log-records/")
        resp_range = client.get(f"{URL}/rng-lr/log-records/", params={"record_range": "0:3"})
        pairs_full = _decode_stream_pairs(resp_full.content)
        pairs_range = _decode_stream_pairs(resp_range.content)
        assert len(pairs_range) < len(pairs_full)

    def test_log_records_default_tail(self, client: TestClient, db: Database) -> None:
        """With 10 records and no range, all should be returned (< DEFAULT_TAIL of 200) plus count."""
        _seed_run(db, "tail-lr")
        _seed_log_records(db, "tail-lr")
        resp = client.get(f"{URL}/tail-lr/log-records/")
        pairs = _decode_stream_pairs(resp.content)
        message_pairs = [p for p in pairs if b"message" in p[0]]
        assert len(message_pairs) == 10


# ── Streaming: run search ───────────────────────────────────────────────────


class TestRunSearchStreaming:
    def test_empty_search(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/search/run/")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"

    def test_search_returns_run(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "s-run", name="Searchable")
        resp = client.get(f"{URL}/search/run/")
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        assert len(pairs) > 0

    def test_search_with_limit(self, client: TestClient, db: Database) -> None:
        for i in range(5):
            _seed_run(db, f"lim-{i}")
        resp = client.get(f"{URL}/search/run/", params={"limit": 2, "report_progress": False})
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        decoded_keys = set()
        for key_bytes, _ in pairs:
            parts = key_bytes.split(b"\xfe")
            if parts and parts[0]:
                decoded_keys.add(parts[0].decode("utf-8", errors="replace"))
        assert len(decoded_keys) <= 2


# ── Streaming: metric search ───────────────────────────────────────────────


class TestMetricSearchStreaming:
    def test_empty_metric_search(self, client: TestClient) -> None:
        resp = client.get(f"{URL}/search/metric/")
        assert resp.status_code == 200

    def test_metric_search_with_data(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "m-run")
        runs.set_trace_info(db, "m-run", 0, "loss", dtype="float", last=0.5, last_step=2)
        runs.set_context(db, "m-run", 0, {})
        write_sequence_step(db, "m-run", 0, "loss", 0, 1.0)
        write_sequence_step(db, "m-run", 0, "loss", 1, 0.8)
        write_sequence_step(db, "m-run", 0, "loss", 2, 0.5)

        resp = client.get(f"{URL}/search/metric/", params={"report_progress": False})
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        assert len(pairs) > 0


# ── Streaming: active runs ─────────────────────────────────────────────────


class TestActiveRunsStreaming:
    def test_no_active(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "fin-run")
        runs.update_run_meta(db, "fin-run", active=False)
        resp = client.get(f"{URL}/active/", params={"report_progress": False})
        assert resp.status_code == 200
        assert resp.content == b""

    def test_active_run_returned(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "act-run")
        resp = client.get(f"{URL}/active/", params={"report_progress": False})
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        assert len(pairs) > 0


# ── Metric batch ────────────────────────────────────────────────────────────


class TestMetricBatch:
    def test_get_batch(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "mb-run")
        runs.set_trace_info(db, "mb-run", 0, "acc", dtype="float", last=0.9, last_step=1)
        runs.set_context(db, "mb-run", 0, {})
        write_sequence_step(db, "mb-run", 0, "acc", 0, 0.7)
        write_sequence_step(db, "mb-run", 0, "acc", 1, 0.9)

        body = [{"name": "acc", "context": {}}]
        resp = client.post(f"{URL}/mb-run/metric/get-batch/", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "acc"
        assert data[0]["values"] == [0.7, 0.9]
        assert data[0]["iters"] == [0, 1]

    def test_get_batch_nonexistent_run(self, client: TestClient) -> None:
        resp = client.post(f"{URL}/no-run/metric/get-batch/", json=[{"name": "x", "context": {}}])
        assert resp.status_code == 404

    def test_get_batch_missing_metric(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "empty-mb")
        resp = client.post(f"{URL}/empty-mb/metric/get-batch/", json=[{"name": "nope", "context": {}}])
        assert resp.status_code == 200
        assert resp.json() == []


# ── Archived filter via secondary index ──────────────────────────────────


class TestArchivedFilterIndex:
    """The default query is ``run.is_archived == False``.

    Verify that the search endpoint only returns non-archived runs, and that
    archived runs are excluded from the results (leveraging the secondary index).
    """

    def test_search_excludes_archived(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "visible")
        _seed_run(db, "hidden")
        runs.update_run_meta(db, "hidden", is_archived=True)

        resp = client.get(
            f"{URL}/search/run/",
            params={"q": "", "report_progress": False},
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        run_hashes = {p[0].decode().split("/")[0] for p in pairs if b"/" in p[0]}
        assert "visible" in run_hashes or len(pairs) > 0
        decoded_bytes = resp.content
        assert b"hidden" not in decoded_bytes

    def test_search_archived_only(self, client: TestClient, db: Database) -> None:
        _seed_run(db, "norm")
        _seed_run(db, "arch")
        runs.update_run_meta(db, "arch", is_archived=True)

        resp = client.get(
            f"{URL}/search/run/",
            params={"q": "run.is_archived == True", "report_progress": False},
        )
        assert resp.status_code == 200
        decoded_bytes = resp.content
        assert b"norm" not in decoded_bytes
