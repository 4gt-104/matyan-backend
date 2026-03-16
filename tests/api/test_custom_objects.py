"""Integration tests for custom object endpoints (texts, distributions, figures).

Uses TextApiConfig endpoints since texts have resolve_blobs=True and don't need S3.
"""

from __future__ import annotations

import struct
import threading
from typing import TYPE_CHECKING
from unittest.mock import patch

from matyan_backend.api.runs._collections import iter_matching_sequences
from matyan_backend.storage import runs, sequences

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database

from .conftest import BASE

_SENTINEL = b"\xfe"


def _seed_text_run(db: Database, run_hash: str = "co1") -> str:
    """Create a run with text sequence data."""
    runs.create_run(db, run_hash)
    runs.set_context(db, run_hash, 0, {})
    runs.set_trace_info(db, run_hash, 0, "texts", dtype="text", last=0.0, last_step=2)
    for i in range(3):
        sequences.write_sequence_step(db, run_hash, 0, "texts", i, {"data": f"text_{i}"})
    return run_hash


def _seed_distribution_run(db: Database, run_hash: str = "co_dist") -> str:
    runs.create_run(db, run_hash)
    runs.set_context(db, run_hash, 0, {})
    runs.set_trace_info(db, run_hash, 0, "distributions", dtype="distribution", last=0.0, last_step=0)
    sequences.write_sequence_step(
        db,
        run_hash,
        0,
        "distributions",
        0,
        {
            "data": [1.0, 2.0, 3.0],
            "bin_count": 3,
            "range": [0, 10],
        },
    )
    return run_hash


class TestTextSearch:
    def test_search(self, db: Database, client: TestClient) -> None:
        _seed_text_run(db, "ts1")
        resp = client.get(f"{BASE}/runs/search/texts/")
        assert resp.status_code == 200
        assert len(resp.content) > 0

    def test_search_with_record_range(self, db: Database, client: TestClient) -> None:
        _seed_text_run(db, "ts2")
        resp = client.get(f"{BASE}/runs/search/texts/", params={"record_range": "0,2"})
        assert resp.status_code == 200


class TestTextGetBatch:
    def test_get_batch(self, db: Database, client: TestClient) -> None:
        _seed_text_run(db, "tb1")
        resp = client.post(
            f"{BASE}/runs/tb1/texts/get-batch/",
            json=[{"name": "texts", "context": {}}],
        )
        assert resp.status_code == 200
        assert len(resp.content) > 0

    def test_get_batch_nonexistent_run(self, client: TestClient) -> None:
        resp = client.post(
            f"{BASE}/runs/nonexistent/texts/get-batch/",
            json=[{"name": "texts", "context": {}}],
        )
        assert resp.status_code == 404


class TestTextGetStep:
    def test_get_last_step(self, db: Database, client: TestClient) -> None:
        _seed_text_run(db, "tgs1")
        resp = client.post(
            f"{BASE}/runs/tgs1/texts/get-step/",
            json=[{"name": "texts", "context": {}}],
            params={"record_step": -1},
        )
        assert resp.status_code == 200

    def test_get_specific_step(self, db: Database, client: TestClient) -> None:
        _seed_text_run(db, "tgs2")
        resp = client.post(
            f"{BASE}/runs/tgs2/texts/get-step/",
            json=[{"name": "texts", "context": {}}],
            params={"record_step": 1},
        )
        assert resp.status_code == 200

    def test_get_step_nonexistent_run(self, client: TestClient) -> None:
        resp = client.post(
            f"{BASE}/runs/missing/texts/get-step/",
            json=[{"name": "texts", "context": {}}],
        )
        assert resp.status_code == 404


class TestDistributionSearch:
    def test_search(self, db: Database, client: TestClient) -> None:
        _seed_distribution_run(db)
        resp = client.get(f"{BASE}/runs/search/distributions/")
        assert resp.status_code == 200


class TestFigureSearch:
    def test_search_empty(self, client: TestClient) -> None:
        resp = client.get(f"{BASE}/runs/search/figures/")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Custom object search — multi-run + background thread tests
# ---------------------------------------------------------------------------


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


def _key_starts_with(encoded_key: bytes, prefix: str) -> bool:
    return encoded_key.startswith(prefix.encode("utf-8") + _SENTINEL)


class TestCustomObjectSearchMultiRun:
    """Custom object search returns data from multiple runs."""

    def test_two_runs_both_present(self, db: Database, client: TestClient) -> None:
        _seed_text_run(db, "cos1")
        _seed_text_run(db, "cos2")
        resp = client.get(
            f"{BASE}/runs/search/texts/",
            params={"report_progress": "false"},
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        run1_keys = [k for k, _ in pairs if _key_starts_with(k, "cos1")]
        run2_keys = [k for k, _ in pairs if _key_starts_with(k, "cos2")]
        assert len(run1_keys) > 0, "run cos1 missing from custom object search"
        assert len(run2_keys) > 0, "run cos2 missing from custom object search"


class TestCustomObjectSearchEventLoopFree:
    """Custom object search iterator runs in a worker thread."""

    def test_iterator_runs_in_background_thread(self, db: Database, client: TestClient) -> None:
        _seed_text_run(db, "cosev1")

        producer_threads: list[str] = []
        real_iter = iter_matching_sequences

        def _tracking_iter(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            producer_threads.append(threading.current_thread().name)
            yield from real_iter(*args, **kwargs)

        with patch(
            "matyan_backend.api.runs._custom_objects.iter_matching_sequences",
            side_effect=_tracking_iter,
        ):
            resp = client.get(
                f"{BASE}/runs/search/texts/",
                params={"report_progress": "false"},
            )
        assert resp.status_code == 200
        assert len(producer_threads) >= 1, "iter_matching_sequences was not called"
        for t_name in producer_threads:
            assert t_name != "MainThread", f"iter_matching_sequences ran on {t_name}, expected a worker thread"
