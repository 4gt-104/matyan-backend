"""Additional streaming endpoint tests — x_axis, offset, skip_system, alignment, ranges, parallel traces."""

from __future__ import annotations

import asyncio
import struct
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

from matyan_backend.api.runs._collections import (
    iter_matching_runs,
    iter_matching_sequences_with_bundle,
)
from matyan_backend.api.runs._planner import PlanResult as _PlanResult
from matyan_backend.api.runs._streaming import (
    _collect_traces_async,
    _metric_search_candidate_streamer,
    _MetricTraceRef,
    run_search_api,
)
from matyan_backend.api.runs._streaming import _flush_buffered_traces as _real_flush
from matyan_backend.storage import runs, sequences
from matyan_backend.storage.entities import create_experiment

from .conftest import BASE

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database


# ── Binary stream helpers ───────────────────────────────────────────────────

_SENTINEL = b"\xfe"


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
    """True when the encoded path starts with the given string component."""  # noqa: D401
    return encoded_key.startswith(prefix.encode("utf-8") + _SENTINEL)


def _key_contains(encoded_key: bytes, component: str) -> bool:
    """True when any path component equals *component*."""  # noqa: D401
    return (component.encode("utf-8") + _SENTINEL) in encoded_key


def _seed_metric_run(db: Database, run_hash: str = "sm1") -> str:
    runs.create_run(db, run_hash)
    runs.set_context(db, run_hash, 0, {})
    runs.set_run_attrs(db, run_hash, (), {"hparams": {"lr": 0.01}, "__system_params": {"sys": True}})
    runs.set_trace_info(db, run_hash, 0, "loss", dtype="float", last=0.05, last_step=9)
    runs.set_trace_info(db, run_hash, 0, "acc", dtype="float", last=0.95, last_step=9)
    for i in range(10):
        sequences.write_sequence_step(db, run_hash, 0, "loss", i, 1.0 - i * 0.1, epoch=0, timestamp=1000.0 + i)
        sequences.write_sequence_step(db, run_hash, 0, "acc", i, i * 0.1, epoch=0, timestamp=1000.0 + i)
    return run_hash


class TestRunSearchExtended:
    def test_search_with_offset(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "rse1")
        _seed_metric_run(db, "rse2")
        resp = client.get(f"{BASE}/runs/search/run/", params={"offset": "rse1"})
        assert resp.status_code == 200

    def test_search_skip_system(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "rse3")
        resp = client.get(f"{BASE}/runs/search/run/", params={"skip_system": "true"})
        assert resp.status_code == 200

    def test_search_exclude_params(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "rse4")
        resp = client.get(f"{BASE}/runs/search/run/", params={"exclude_params": "true"})
        assert resp.status_code == 200

    def test_search_exclude_traces(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "rse5")
        resp = client.get(f"{BASE}/runs/search/run/", params={"exclude_traces": "true"})
        assert resp.status_code == 200


class TestMetricSearchExtended:
    def test_with_x_axis(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "mse1")
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"x_axis": "acc"},
        )
        assert resp.status_code == 200

    def test_with_density(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "mse2")
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"p": 3},
        )
        assert resp.status_code == 200

    def test_batch_produces_non_empty_traces(self, db: Database, client: TestClient) -> None:
        """Metric search uses sample_sequences_batch; verify traces have data."""
        _seed_metric_run(db, "mse3")
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"p": 5, "report_progress": "false"},
        )
        assert resp.status_code == 200
        assert len(resp.content) > 0


class TestActiveRunsExtended:
    def test_active_with_metrics(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "are1")
        resp = client.get(f"{BASE}/runs/active/", params={"report_progress": "true"})
        assert resp.status_code == 200
        assert len(resp.content) > 0


class TestMetricAlignment:
    def test_basic_alignment(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "ma1")
        resp = client.post(
            f"{BASE}/runs/search/metric/align/",
            json={
                "align_by": "acc",
                "runs": [
                    {
                        "run_id": "ma1",
                        "traces": [
                            {"name": "loss", "context": {}, "slice": [0, 10, 1]},
                        ],
                    },
                ],
            },
        )
        assert resp.status_code == 200

    def test_alignment_nonexistent_run(self, client: TestClient) -> None:
        resp = client.post(
            f"{BASE}/runs/search/metric/align/",
            json={
                "align_by": "acc",
                "runs": [
                    {
                        "run_id": "missing",
                        "traces": [{"name": "loss", "context": {}, "slice": [0, 10, 1]}],
                    },
                ],
            },
        )
        assert resp.status_code == 200


class TestMetricBatchExtended:
    def test_with_record_range(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "mbe1")
        resp = client.post(
            f"{BASE}/runs/mbe1/metric/get-batch/",
            json=[{"name": "loss", "context": {}}],
            params={"record_range": "2,8", "record_density": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "loss"
        assert len(data[0]["iters"]) <= 3

    def test_missing_metric(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "mbe2")
        resp = client.post(
            f"{BASE}/runs/mbe2/metric/get-batch/",
            json=[{"name": "nonexistent", "context": {}}],
        )
        assert resp.status_code == 200
        assert resp.json() == []


class TestMetricSearchCandidatePath:
    """Verify the candidate (hash) path is used and the lazy producer iterator is bypassed."""

    def test_candidate_path_avoids_iterator(self, db: Database, client: TestClient) -> None:
        """With q="" the planner returns candidates, so iter_matching_sequences_with_bundle must not be called."""
        _seed_metric_run(db, "mcp1")
        with patch(
            "matyan_backend.api.runs._streaming.iter_matching_sequences_with_bundle",
        ) as mock_iter:
            resp = client.get(
                f"{BASE}/runs/search/metric/",
                params={"report_progress": "false"},
            )
            assert resp.status_code == 200
            assert len(resp.content) > 0
            assert mock_iter.call_count == 0

    def test_candidate_path_returns_valid_data(self, db: Database, client: TestClient) -> None:
        """Candidate path should still return metric traces with data."""
        _seed_metric_run(db, "mcp2")
        _seed_metric_run(db, "mcp3")
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"p": 5, "report_progress": "false"},
        )
        assert resp.status_code == 200
        assert len(resp.content) > 0

    def test_lazy_path_used_when_planner_returns_none(self, db: Database, client: TestClient) -> None:
        """When plan_query returns None candidates, the lazy path via iter_matching_sequences_with_bundle is used."""
        _seed_metric_run(db, "mcp4")
        none_result = _PlanResult(candidates=None, exact=True)
        with (
            patch("matyan_backend.api.runs._streaming.plan_query", return_value=none_result),
            patch(
                "matyan_backend.api.runs._streaming.iter_matching_sequences_with_bundle",
                wraps=lambda *a, **kw: iter([]),  # noqa: ARG005
            ) as mock_iter,
        ):
            resp = client.get(
                f"{BASE}/runs/search/metric/",
                params={"report_progress": "false"},
            )
            assert resp.status_code == 200
            assert mock_iter.call_count >= 1

    def test_metric_name_eq_uses_candidate_path(self, db: Database, client: TestClient) -> None:
        """metric.name == "loss" is now index-backed and should use the candidate path."""
        _seed_metric_run(db, "mcp5")  # has "loss" and "acc"
        with patch(
            "matyan_backend.api.runs._streaming.iter_matching_sequences_with_bundle",
        ) as mock_iter:
            resp = client.get(
                f"{BASE}/runs/search/metric/",
                params={"q": 'metric.name == "loss"', "report_progress": "false"},
            )
            assert resp.status_code == 200
            assert len(resp.content) > 0
            assert mock_iter.call_count == 0, (
                "iter_matching_sequences_with_bundle should NOT be called for indexed metric.name"
            )

    def test_metric_name_eq_filters_traces(self, db: Database, client: TestClient) -> None:
        """Only the queried metric trace should appear in the response."""
        _seed_metric_run(db, "mcp6")  # has "loss" and "acc"
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"q": 'metric.name == "loss"', "report_progress": "false"},
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        _string_tag = 4
        trace_names: list[str] = []
        for key, val in pairs:
            if _key_starts_with(key, "mcp6") and _key_contains(key, "name") and len(val) > 1 and val[0] == _string_tag:
                trace_names.append(val[1:].decode("utf-8", errors="ignore"))
        assert "loss" in trace_names, "expected 'loss' trace in response"
        assert "acc" not in trace_names, "'acc' trace should be filtered out"

    def test_metric_name_startswith_still_uses_lazy_path(self, db: Database, client: TestClient) -> None:
        """Unindexed metric predicates still force the lazy path."""
        _seed_metric_run(db, "mcp7")
        with patch(
            "matyan_backend.api.runs._streaming.iter_matching_sequences_with_bundle",
            wraps=iter_matching_sequences_with_bundle,
        ) as mock_iter:
            resp = client.get(
                f"{BASE}/runs/search/metric/",
                params={"q": 'metric.name.startswith("lo")', "report_progress": "false"},
            )
            assert resp.status_code == 200
            assert mock_iter.call_count >= 1

    def test_exact_single_run_uses_single_bundle_fast_lane(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "mcp8")
        exact_result = _PlanResult(candidates=["mcp8"], exact=True)
        with (
            patch("matyan_backend.api.runs._streaming.plan_query", return_value=exact_result),
            patch(
                "matyan_backend.api.runs._streaming.get_metric_search_bundle",
                wraps=runs.get_metric_search_bundle,
            ) as mock_single,
            patch("matyan_backend.api.runs._streaming.get_run_bundles") as mock_batched,
        ):
            resp = client.get(
                f"{BASE}/runs/search/metric/",
                params={"q": "", "report_progress": "false"},
            )
            assert resp.status_code == 200
            assert len(resp.content) > 0
            assert mock_single.call_count == 1
            assert mock_batched.call_count == 0

    def test_multi_run_prefetches_next_batch_before_first_run_finishes(self, db: Database) -> None:
        candidates = [f"mcp_prefetch_{idx}" for idx in range(11)]
        second_batch_started = threading.Event()

        def _bundle(run_hash: str) -> dict:
            return {
                "meta": {"name": run_hash},
                "attrs": {},
                "traces": [{"name": "loss", "dtype": "float", "context_id": 0}],
                "contexts": {0: {}},
                "tags": [],
                "experiment": None,
            }

        def fake_get_run_bundles(
            _db: object,
            batch: list[str],
            *,
            include_attrs: bool = True,
            include_traces: bool = True,
        ) -> list[dict]:
            del _db, include_attrs, include_traces
            if len(batch) == 1 and batch[0] == candidates[-1]:
                second_batch_started.set()
            return [_bundle(run_hash) for run_hash in batch]

        async def fake_collect(
            _db: object,
            run_hash: str,
            buffered_svs: list[_MetricTraceRef],
            num_points: int,
            x_axis: str | None,
            trace_chunk_size: int,
        ) -> list[dict]:
            del _db, buffered_svs, num_points, x_axis, trace_chunk_size
            if run_hash == candidates[0]:
                assert second_batch_started.is_set(), "next batch should be prefetched while first run is in flight"
            return [{"name": "loss", "context": {}, "values": {"last": 1.0}}]

        async def consume() -> list[bytes]:
            with (
                patch("matyan_backend.api.runs._streaming.get_run_bundles", side_effect=fake_get_run_bundles),
                patch("matyan_backend.api.runs._streaming._collect_traces_async", side_effect=fake_collect),
                patch("matyan_backend.api.runs._streaming.encode_tree", side_effect=lambda data: data),
                patch(
                    "matyan_backend.api.runs._streaming.stream_tree_data",
                    side_effect=lambda data: [next(iter(data.keys())).encode("utf-8")],
                ),
            ):
                return [
                    chunk
                    async for chunk in _metric_search_candidate_streamer(
                        db,
                        candidates,
                        num_points=5,
                        x_axis=None,
                        skip_system=True,
                        report_progress=False,
                    )
                ]

        chunks = asyncio.run(consume())
        assert chunks[0] == candidates[0].encode("utf-8")

    def test_current_batch_yields_before_next_batch_prefetch_finishes(self, db: Database) -> None:
        candidates = [f"mcp_progress_{idx}" for idx in range(11)]
        release_second_batch = threading.Event()
        second_batch_started = threading.Event()

        def _bundle(run_hash: str) -> dict:
            return {
                "meta": {"name": run_hash},
                "attrs": {},
                "traces": [{"name": "loss", "dtype": "float", "context_id": 0}],
                "contexts": {0: {}},
                "tags": [],
                "experiment": None,
            }

        def fake_get_run_bundles(
            _db: object,
            batch: list[str],
            *,
            include_attrs: bool = True,
            include_traces: bool = True,
        ) -> list[dict]:
            del _db, include_attrs, include_traces
            if len(batch) == 1 and batch[0] == candidates[-1]:
                second_batch_started.set()
                assert release_second_batch.wait(timeout=1.0), "second batch fetch was never released"
            return [_bundle(run_hash) for run_hash in batch]

        async def fake_collect(
            _db: object,
            run_hash: str,
            buffered_svs: list[_MetricTraceRef],
            num_points: int,
            x_axis: str | None,
            trace_chunk_size: int,
        ) -> list[dict]:
            del _db, buffered_svs, num_points, x_axis, trace_chunk_size
            if run_hash == candidates[0]:
                assert second_batch_started.is_set(), "expected next-batch prefetch to have started"
            return [{"name": "loss", "context": {}, "values": {"last": 1.0}}]

        async def consume_first_chunk() -> bytes:
            with (
                patch("matyan_backend.api.runs._streaming.get_run_bundles", side_effect=fake_get_run_bundles),
                patch("matyan_backend.api.runs._streaming._collect_traces_async", side_effect=fake_collect),
                patch("matyan_backend.api.runs._streaming.encode_tree", side_effect=lambda data: data),
                patch(
                    "matyan_backend.api.runs._streaming.stream_tree_data",
                    side_effect=lambda data: [next(iter(data.keys())).encode("utf-8")],
                ),
            ):
                streamer = _metric_search_candidate_streamer(
                    db,
                    candidates,
                    num_points=5,
                    x_axis=None,
                    skip_system=True,
                    report_progress=False,
                )
                try:
                    chunk = await asyncio.wait_for(streamer.__anext__(), timeout=0.05)
                finally:
                    release_second_batch.set()
                    await streamer.aclose()
                return chunk

        chunk = asyncio.run(consume_first_chunk())
        assert chunk == candidates[0].encode("utf-8")


class TestCollectTracesAdaptiveBehavior:
    def test_x_axis_is_sampled_once_across_chunks(self, db: Database) -> None:
        trace_refs = [_MetricTraceRef(f"metric_{idx}", {}, 0) for idx in range(4)]
        x_axis_calls: list[str | None] = []

        def fake_sample_sequences_batch(
            _db: object,
            run_hash: str,
            requests: list[tuple[int, str]],
            num_points: int,
            *,
            columns: tuple[str, ...] = ("val",),
            x_axis_name: str | None = None,
        ) -> tuple[dict[tuple[int, str], dict[str, list]], dict[int, dict[str, list]] | None]:
            del _db, run_hash, num_points, columns
            x_axis_calls.append(x_axis_name)
            main = {req: {"steps": [0, 1], "val": [1.0, 2.0]} for req in requests}
            x_axis = {0: {"steps": [0, 1], "val": [10.0, 20.0]}} if x_axis_name is not None else None
            return main, x_axis

        async def collect() -> list[dict]:
            with patch(
                "matyan_backend.api.runs._streaming.sample_sequences_batch",
                side_effect=fake_sample_sequences_batch,
            ):
                return await _collect_traces_async(
                    db,
                    "trace_chunk_run",
                    trace_refs,
                    num_points=50,
                    x_axis="step",
                    trace_chunk_size=2,
                )

        traces = asyncio.run(collect())
        assert len(traces) == 4
        assert x_axis_calls.count("step") == 1

    def test_chunk_work_is_bounded_instead_of_unbounded_gather(self, db: Database) -> None:
        trace_refs = [_MetricTraceRef(f"metric_{idx}", {}, 0) for idx in range(6)]
        current = 0
        peak = 0
        lock = threading.Lock()

        def fake_flush(
            _db: object,
            run_hash: str,
            buffered_svs: list[object],
            num_points: int,
            x_axis: str | None,
            prefetched_x_axis: dict[int, dict[str, list]] | None = None,
        ) -> list[dict]:
            del _db, run_hash, buffered_svs, num_points, x_axis, prefetched_x_axis
            nonlocal current, peak
            with lock:
                current += 1
                peak = max(peak, current)
            time.sleep(0.05)
            with lock:
                current -= 1
            return [{"name": "loss", "context": {}, "values": {"last": 1.0}}]

        async def collect() -> list[dict]:
            with patch(
                "matyan_backend.api.runs._streaming._flush_buffered_traces",
                side_effect=fake_flush,
            ):
                return await _collect_traces_async(
                    db,
                    "trace_chunk_run",
                    trace_refs,
                    num_points=50,
                    x_axis=None,
                    trace_chunk_size=2,
                )

        traces = asyncio.run(collect())
        assert len(traces) == 3
        assert peak == 2


class TestSupersetFiltering:
    """When the planner returns a superset (exact=False), callers must filter."""

    def _seed_named_run(self, db: Database, run_hash: str, *, name: str) -> None:
        runs.create_run(db, run_hash, name=name)
        runs.set_context(db, run_hash, 0, {})
        runs.set_run_attrs(db, run_hash, (), {"hparams": {"lr": 0.01}})
        runs.set_trace_info(db, run_hash, 0, "loss", dtype="float", last=0.5, last_step=4)
        for i in range(5):
            sequences.write_sequence_step(db, run_hash, 0, "loss", i, 1.0 - i * 0.2, epoch=0, timestamp=1000.0 + i)

    def test_run_search_superset_filters_by_name(self, db: Database, client: TestClient) -> None:
        """run.name is unindexed; superset path should filter via check()."""
        self._seed_named_run(db, "sup1", name="alpha")
        self._seed_named_run(db, "sup2", name="beta")

        resp = client.get(
            f"{BASE}/runs/search/run/",
            params={"q": 'run.name == "alpha"', "report_progress": "false"},
        )
        assert resp.status_code == 200
        body = resp.content
        assert b"sup1" in body
        assert b"sup2" not in body

    def test_metric_search_superset_filters_by_name(self, db: Database, client: TestClient) -> None:
        """Metric search candidate path should filter when exact=False."""
        self._seed_named_run(db, "msup1", name="gamma")
        self._seed_named_run(db, "msup2", name="delta")

        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={
                "q": 'run.name == "gamma"',
                "report_progress": "false",
            },
        )
        assert resp.status_code == 200
        body = resp.content
        assert b"msup1" in body
        assert b"msup2" not in body


class TestCandidatePathSequenceLevelPredicates:
    """Candidate path should handle metric-level predicates via real SequenceView."""

    def _seed_run_with_context(
        self,
        db: Database,
        run_hash: str,
        *,
        name: str,
        ctx: dict,
        metric_name: str,
        last: float,
    ) -> None:
        runs.create_run(db, run_hash, name=name)
        runs.set_context(db, run_hash, 0, ctx)
        runs.set_run_attrs(db, run_hash, (), {"hparams": {"lr": 0.01}})
        runs.set_trace_info(db, run_hash, 0, metric_name, dtype="float", last=last, last_step=4)
        for i in range(5):
            sequences.write_sequence_step(
                db,
                run_hash,
                0,
                metric_name,
                i,
                last - i * 0.1,
                epoch=0,
                timestamp=1000.0 + i,
            )

    def test_metric_name_with_run_predicate(self, db: Database, client: TestClient) -> None:
        """Query combining run.description != 'x' and metric.name == 'loss'."""
        self._seed_run_with_context(
            db,
            "csp1",
            name="r1",
            ctx={},
            metric_name="loss",
            last=0.5,
        )
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={
                "q": '(run.description != "nonexistent") and (metric.name == "loss")',
                "report_progress": "false",
            },
        )
        assert resp.status_code == 200
        assert b"csp1" in resp.content

    def test_metric_name_filters_non_matching(self, db: Database, client: TestClient) -> None:
        """Only runs with matching metric name should be returned."""
        self._seed_run_with_context(
            db,
            "csp2",
            name="r2",
            ctx={},
            metric_name="loss",
            last=0.5,
        )
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={
                "q": 'metric.name == "nonexistent_metric"',
                "report_progress": "false",
            },
        )
        assert resp.status_code == 200
        assert b"csp2" not in resp.content

    def test_metric_context_filtering(self, db: Database, client: TestClient) -> None:
        """metric.context.subset == 'train' should match runs with that context."""
        self._seed_run_with_context(
            db,
            "csp3",
            name="r3",
            ctx={"subset": "train"},
            metric_name="loss",
            last=0.5,
        )
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={
                "q": 'metric.context.subset == "train"',
                "report_progress": "false",
            },
        )
        assert resp.status_code == 200
        assert b"csp3" in resp.content

    def test_metric_context_filters_non_matching(self, db: Database, client: TestClient) -> None:
        """metric.context.subset == 'val' should not match a 'train' context."""
        self._seed_run_with_context(
            db,
            "csp4",
            name="r4",
            ctx={"subset": "train"},
            metric_name="loss",
            last=0.5,
        )
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={
                "q": 'metric.context.subset == "val"',
                "report_progress": "false",
            },
        )
        assert resp.status_code == 200
        assert b"csp4" not in resp.content

    def test_metric_last_filtering(self, db: Database, client: TestClient) -> None:
        """metric.last > 0.3 should match runs where last value is above threshold."""
        self._seed_run_with_context(
            db,
            "csp5",
            name="r5",
            ctx={},
            metric_name="loss",
            last=0.5,
        )
        self._seed_run_with_context(
            db,
            "csp6",
            name="r6",
            ctx={},
            metric_name="loss",
            last=0.1,
        )
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={
                "q": "metric.last > 0.3",
                "report_progress": "false",
            },
        )
        assert resp.status_code == 200
        body = resp.content
        assert b"csp5" in body
        assert b"csp6" not in body


class TestRunBundleFlagsPropagation:
    """Verify ``get_run_bundles`` is called with the correct ``include_*``
    flags from the run search endpoint.
    """

    def test_exclude_traces_passes_flag(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "et1")
        with patch(
            "matyan_backend.api.runs._streaming.get_run_bundles",
            wraps=runs.get_run_bundles,
        ) as mock_bundles:
            resp = client.get(
                f"{BASE}/runs/search/run/",
                params={"exclude_traces": "true"},
            )
            assert resp.status_code == 200
            assert mock_bundles.call_count >= 1
            for call in mock_bundles.call_args_list:
                assert call.kwargs.get("include_traces") is False

    def test_include_traces_passes_flag(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "et2")
        with patch(
            "matyan_backend.api.runs._streaming.get_run_bundles",
            wraps=runs.get_run_bundles,
        ) as mock_bundles:
            resp = client.get(
                f"{BASE}/runs/search/run/",
                params={"exclude_traces": "false"},
            )
            assert resp.status_code == 200
            assert mock_bundles.call_count >= 1
            for call in mock_bundles.call_args_list:
                assert call.kwargs.get("include_traces") is True

    def test_exclude_params_passes_flag(self, db: Database, client: TestClient) -> None:
        _seed_metric_run(db, "et3")
        with patch(
            "matyan_backend.api.runs._streaming.get_run_bundles",
            wraps=runs.get_run_bundles,
        ) as mock_bundles:
            resp = client.get(
                f"{BASE}/runs/search/run/",
                params={"exclude_params": "true"},
            )
            assert resp.status_code == 200
            assert mock_bundles.call_count >= 1
            for call in mock_bundles.call_args_list:
                assert call.kwargs.get("include_attrs") is False


# ---------------------------------------------------------------------------
# Lazy metric trace streaming (trace_chunk_size controls FDB batch size)
# ---------------------------------------------------------------------------


def _seed_many_traces_run(db: Database, run_hash: str, n_metrics: int = 12) -> list[str]:
    """Seed a run with *n_metrics* distinct metric traces (each with a few steps)."""
    runs.create_run(db, run_hash)
    runs.set_context(db, run_hash, 0, {})
    runs.set_run_attrs(db, run_hash, (), {"hparams": {"lr": 0.01}})
    names = [f"metric_{i}" for i in range(n_metrics)]
    for name in names:
        runs.set_trace_info(db, run_hash, 0, name, dtype="float", last=0.5, last_step=4)
        for step in range(5):
            sequences.write_sequence_step(
                db,
                run_hash,
                0,
                name,
                step,
                float(step),
                epoch=0,
                timestamp=1000.0 + step,
            )
    return names


class TestLazyMetricTraceStreaming:
    """Verify metric search streams traces inline (no __append_metric_traces__ key)."""

    def test_candidate_path_has_inline_traces(self, db: Database, client: TestClient) -> None:
        """Candidate path: traces are inline under the run hash, not in a separate append key."""
        _seed_many_traces_run(db, "chk1", n_metrics=5)
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"report_progress": "false", "trace_chunk_size": 2},
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)

        run_keys = [k for k, _v in pairs if _key_starts_with(k, "chk1")]
        assert len(run_keys) > 0, "expected encoded data for the run"

        append_keys = [k for k in run_keys if _key_contains(k, "__append_metric_traces__")]
        assert len(append_keys) == 0, "should NOT contain __append_metric_traces__ key"

        trace_keys = [k for k in run_keys if _key_contains(k, "values")]
        assert len(trace_keys) > 0, "expected inline trace data (values)"

    def test_lazy_path_has_inline_traces(self, db: Database, client: TestClient) -> None:
        """Lazy path (sequence-level query): traces are inline, not in append key."""
        _seed_many_traces_run(db, "chk2", n_metrics=5)
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={
                "q": 'metric.name.startswith("metric")',
                "report_progress": "false",
                "trace_chunk_size": 2,
            },
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)

        run_keys = [k for k, _v in pairs if _key_starts_with(k, "chk2")]
        append_keys = [k for k in run_keys if _key_contains(k, "__append_metric_traces__")]
        assert len(append_keys) == 0, "should NOT contain __append_metric_traces__ key"

        trace_keys = [k for k in run_keys if _key_contains(k, "values")]
        assert len(trace_keys) > 0, "expected inline trace data"

    def test_trace_chunk_size_accepted(self, db: Database, client: TestClient) -> None:
        """trace_chunk_size=1 is accepted and produces valid trace data."""
        _seed_many_traces_run(db, "chk3", n_metrics=4)
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"report_progress": "false", "trace_chunk_size": 1},
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        trace_keys = [k for k, _v in pairs if _key_starts_with(k, "chk3") and _key_contains(k, "values")]
        assert len(trace_keys) >= 4

    def test_large_chunk_size(self, db: Database, client: TestClient) -> None:
        """trace_chunk_size >= total traces still works."""
        _seed_many_traces_run(db, "chk4", n_metrics=3)
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"report_progress": "false", "trace_chunk_size": 500},
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        trace_keys = [k for k, _v in pairs if _key_starts_with(k, "chk4") and _key_contains(k, "values")]
        assert len(trace_keys) >= 3

    def test_chunk_size_clamped_to_minimum(self, db: Database, client: TestClient) -> None:
        """trace_chunk_size=0 or negative is clamped to 1 (still works, no 4xx)."""
        _seed_many_traces_run(db, "chk5", n_metrics=2)
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"report_progress": "false", "trace_chunk_size": 0},
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        assert len(pairs) > 0

    def test_default_chunk_size(self, db: Database, client: TestClient) -> None:
        """Without trace_chunk_size param, default (10) is used — still works."""
        _seed_many_traces_run(db, "chk6", n_metrics=2)
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"report_progress": "false"},
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        trace_keys = [k for k, _v in pairs if _key_starts_with(k, "chk6") and _key_contains(k, "values")]
        assert len(trace_keys) >= 2


# ---------------------------------------------------------------------------
# Parallel metric fetch (batch bundles, parallel trace chunks, to_thread)
# ---------------------------------------------------------------------------


class TestBatchBundleReads:
    """Candidate path batches bundle reads via get_run_bundles (batch_size=10)."""

    def test_many_runs_all_returned(self, db: Database, client: TestClient) -> None:
        """Seed 15 runs (> batch size of 10); all must appear in the response."""
        hashes = [f"bb{i:02d}" for i in range(15)]
        for h in hashes:
            _seed_metric_run(db, h)

        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"report_progress": "false"},
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        for h in hashes:
            run_keys = [k for k, _v in pairs if _key_starts_with(k, h)]
            assert len(run_keys) > 0, f"run {h} missing from response"

    def test_batch_bundles_called(self, db: Database, client: TestClient) -> None:
        """get_run_bundles is used (not get_run_bundle) on the candidate path."""
        for i in range(3):
            _seed_metric_run(db, f"bbc{i}")
        with patch(
            "matyan_backend.api.runs._streaming.get_run_bundles",
            wraps=runs.get_run_bundles,
        ) as mock_bundles:
            resp = client.get(
                f"{BASE}/runs/search/metric/",
                params={"report_progress": "false"},
            )
            assert resp.status_code == 200
            assert mock_bundles.call_count >= 1

    def test_progress_reporting_with_batches(self, db: Database, client: TestClient) -> None:
        """Progress keys are emitted correctly even with batched bundle reads."""
        for i in range(12):
            _seed_metric_run(db, f"bbp{i:02d}")
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"report_progress": "true"},
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        progress_keys = [k for k, _v in pairs if b"progress" in k]
        assert len(progress_keys) > 0, "expected progress reports"


class TestProgressHeartbeat:
    def test_run_search_emits_initial_progress_before_slow_bundle_batch(self, db: Database) -> None:
        release_bundles = threading.Event()

        def fake_iter_matching_runs(*args: object, **kwargs: object):  # noqa: ANN202
            del args, kwargs
            yield SimpleNamespace(hash="rshb0"), (1, 1)

        def fake_get_run_bundles(
            _db: object,
            hashes: list[str],
            *,
            include_attrs: bool = True,
            include_traces: bool = True,
        ) -> list[dict]:
            del _db, include_attrs, include_traces
            assert release_bundles.wait(timeout=1.0), "bundle fetch was never released"
            return [
                {
                    "meta": {"name": run_hash},
                    "attrs": {},
                    "traces": [],
                    "contexts": {},
                    "tags": [],
                    "experiment": None,
                }
                for run_hash in hashes
            ]

        def fake_collect_streamable_data(data: object) -> bytes:
            if isinstance(data, dict) and any(str(key).startswith("progress") for key in data):
                return b"PROGRESS"
            return b"RUN"

        def _as_bytes(chunk: str | bytes | memoryview) -> bytes:
            if isinstance(chunk, bytes):
                return chunk
            if isinstance(chunk, memoryview):
                return bytes(chunk)
            return chunk.encode("utf-8")

        async def consume_first_chunk() -> bytes:
            response = await run_search_api(
                db,
                q="",
                report_progress=True,
            )
            iterator = response.body_iterator
            typed_iterator: AsyncIterator[str | bytes | memoryview[int]] = iterator  # ty: ignore[invalid-assignment]
            try:
                return _as_bytes(await asyncio.wait_for(anext(typed_iterator), timeout=0.05))
            finally:
                release_bundles.set()
                await iterator.aclose()

        with (
            patch("matyan_backend.api.runs._streaming.iter_matching_runs", side_effect=fake_iter_matching_runs),
            patch("matyan_backend.api.runs._streaming.get_run_bundles", side_effect=fake_get_run_bundles),
            patch("matyan_backend.api.runs._streaming.encode_tree", side_effect=lambda data: data),
            patch(
                "matyan_backend.api.runs._streaming.collect_streamable_data",
                side_effect=fake_collect_streamable_data,
            ),
        ):
            first_chunk = asyncio.run(consume_first_chunk())
        assert first_chunk == b"PROGRESS"

    def test_run_search_emits_heartbeat_progress_while_bundle_batch_is_slow(
        self,
        db: Database,
    ) -> None:
        release_bundles = threading.Event()

        def fake_iter_matching_runs(*args: object, **kwargs: object):  # noqa: ANN202
            del args, kwargs
            yield SimpleNamespace(hash="rshb1"), (1, 1)

        def fake_get_run_bundles(
            _db: object,
            hashes: list[str],
            *,
            include_attrs: bool = True,
            include_traces: bool = True,
        ) -> list[dict]:
            del _db, include_attrs, include_traces
            assert release_bundles.wait(timeout=1.0), "bundle fetch was never released"
            return [
                {
                    "meta": {"name": run_hash},
                    "attrs": {},
                    "traces": [],
                    "contexts": {},
                    "tags": [],
                    "experiment": None,
                }
                for run_hash in hashes
            ]

        def fake_collect_streamable_data(data: object) -> bytes:
            if isinstance(data, dict) and any(str(key).startswith("progress") for key in data):
                return b"PROGRESS"
            return b"RUN"

        def _as_bytes(chunk: str | bytes | memoryview) -> bytes:
            if isinstance(chunk, bytes):
                return chunk
            if isinstance(chunk, memoryview):
                return bytes(chunk)
            return chunk.encode("utf-8")

        async def consume_two_chunks() -> list[bytes]:
            response = await run_search_api(
                db,
                q="",
                report_progress=True,
            )
            iterator = response.body_iterator
            typed_iterator: AsyncIterator[str | bytes | memoryview[int]] = iterator  # ty: ignore[invalid-assignment]
            try:
                return [
                    _as_bytes(await asyncio.wait_for(anext(typed_iterator), timeout=0.05)),
                    _as_bytes(await asyncio.wait_for(anext(typed_iterator), timeout=0.08)),
                ]
            finally:
                release_bundles.set()
                await iterator.aclose()

        with (
            patch("matyan_backend.api.runs._streaming.iter_matching_runs", side_effect=fake_iter_matching_runs),
            patch("matyan_backend.api.runs._streaming.get_run_bundles", side_effect=fake_get_run_bundles),
            patch("matyan_backend.api.runs._streaming.encode_tree", side_effect=lambda data: data),
            patch(
                "matyan_backend.api.runs._streaming.collect_streamable_data",
                side_effect=fake_collect_streamable_data,
            ),
            patch("matyan_backend.api.runs._streaming.PROGRESS_REPORT_INTERVAL", 0.01),
        ):
            observed_chunks = asyncio.run(consume_two_chunks())
        assert observed_chunks == [b"PROGRESS", b"PROGRESS"]

    def test_metric_candidate_emits_initial_progress_before_initial_batch_fetch_finishes(
        self,
        db: Database,
    ) -> None:
        release_fetch = threading.Event()

        async def fake_fetch_candidate_batch(
            _db: object,
            batch: list[str],
        ) -> tuple[list[str], list[dict | None], float]:
            del _db
            assert await asyncio.to_thread(
                release_fetch.wait,
                1.0,
            ), "initial batch fetch was never released"
            return batch, [None for _ in batch], 0.01

        async def consume_first_chunk() -> bytes:
            with (
                patch(
                    "matyan_backend.api.runs._streaming._fetch_candidate_batch",
                    side_effect=fake_fetch_candidate_batch,
                ),
                patch("matyan_backend.api.runs._streaming.encode_tree", side_effect=lambda data: data),
                patch(
                    "matyan_backend.api.runs._streaming.collect_streamable_data",
                    side_effect=lambda data: (
                        b"PROGRESS" if any(str(key).startswith("progress") for key in data) else b"RUN"
                    ),
                ),
            ):
                streamer = _metric_search_candidate_streamer(
                    db,
                    ["mchb0", "mchb1"],
                    num_points=5,
                    x_axis=None,
                    skip_system=True,
                    report_progress=True,
                )
                try:
                    return await asyncio.wait_for(streamer.__anext__(), timeout=0.05)
                finally:
                    release_fetch.set()
                    await streamer.aclose()

        chunk = asyncio.run(consume_first_chunk())
        assert chunk == b"PROGRESS"

    def test_metric_candidate_emits_heartbeat_while_ordered_output_is_blocked(
        self,
        db: Database,
    ) -> None:
        release_first = threading.Event()

        async def fake_process_candidate_run(
            _db: object,
            run_hash: str,
            bundle: dict | None,
            num_points: int,
            x_axis: str | None,
            skip_system: bool,
            q: object | None,
            has_seq_pred: bool,
            tz_offset: int,
            trace_chunk_size: int,
            trace_names: frozenset[str] | None,
            timing: bool,
            counter: int,
        ) -> tuple[int, str, dict | None, float, float, bool]:
            del _db, bundle, num_points, x_axis, skip_system, q, has_seq_pred, tz_offset, trace_chunk_size
            del trace_names, timing
            if counter == 1:
                assert await asyncio.to_thread(
                    release_first.wait,
                    1.0,
                ), "first run was never released"
            return counter, run_hash, {"traces": []}, 0.01, 0.02, True

        async def fake_fetch_candidate_batch(
            _db: object,
            batch: list[str],
        ) -> tuple[list[str], list[dict | None], float]:
            del _db
            return (
                batch,
                [
                    {
                        "meta": {},
                        "attrs": {},
                        "traces": [],
                        "contexts": {},
                        "tags": [],
                        "experiment": None,
                    }
                    for _ in batch
                ],
                0.01,
            )

        async def consume_two_chunks() -> list[bytes]:
            with (
                patch(
                    "matyan_backend.api.runs._streaming._process_candidate_run",
                    side_effect=fake_process_candidate_run,
                ),
                patch(
                    "matyan_backend.api.runs._streaming._fetch_candidate_batch",
                    side_effect=fake_fetch_candidate_batch,
                ),
                patch("matyan_backend.api.runs._streaming.encode_tree", side_effect=lambda data: data),
                patch(
                    "matyan_backend.api.runs._streaming.collect_streamable_data",
                    side_effect=lambda data: (
                        b"PROGRESS" if any(str(key).startswith("progress") for key in data) else b"RUN"
                    ),
                ),
                patch(
                    "matyan_backend.api.runs._streaming.stream_tree_data",
                    side_effect=lambda _data: [b"RUN"],
                ),
                patch("matyan_backend.api.runs._streaming.PROGRESS_REPORT_INTERVAL", 0.01),
            ):
                streamer = _metric_search_candidate_streamer(
                    db,
                    ["mchb2", "mchb3"],
                    num_points=5,
                    x_axis=None,
                    skip_system=True,
                    report_progress=True,
                )
                try:
                    chunks = [
                        await asyncio.wait_for(streamer.__anext__(), timeout=0.05),
                        await asyncio.wait_for(streamer.__anext__(), timeout=0.05),
                    ]
                finally:
                    release_first.set()
                    await streamer.aclose()
                return chunks

        chunks = asyncio.run(consume_two_chunks())
        assert chunks == [b"PROGRESS", b"PROGRESS"]


class TestParallelTraceChunks:
    """Trace chunks are fetched in parallel via asyncio.gather + to_thread."""

    def test_trace_order_preserved(self, db: Database, client: TestClient) -> None:
        """12 metrics with chunk_size=3 (4 parallel chunks) must appear in order."""
        names = _seed_many_traces_run(db, "ptc1", n_metrics=12)
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"report_progress": "false", "trace_chunk_size": 3},
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)

        _string_tag = 4
        observed_names: list[str] = []
        for key, val in pairs:
            if _key_starts_with(key, "ptc1") and _key_contains(key, "name") and len(val) > 1 and val[0] == _string_tag:
                decoded_val = val[1:].decode("utf-8", errors="ignore")
                if decoded_val.startswith("metric_"):
                    observed_names.append(decoded_val)

        assert set(observed_names) == set(names), f"missing metrics: expected {set(names)}, got {set(observed_names)}"
        assert observed_names == sorted(names), "trace order not preserved"

    def test_all_trace_data_present(self, db: Database, client: TestClient) -> None:
        """Each metric trace must contain values and iters keys."""
        n = 8
        _seed_many_traces_run(db, "ptc2", n_metrics=n)
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={"report_progress": "false", "trace_chunk_size": 2},
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        value_keys = [k for k, _v in pairs if _key_starts_with(k, "ptc2") and _key_contains(k, "values")]
        iter_keys = [k for k, _v in pairs if _key_starts_with(k, "ptc2") and _key_contains(k, "iters")]
        assert len(value_keys) >= n
        assert len(iter_keys) >= n


class TestFdbOffloadedToThreads:
    """Verify FDB work runs in worker threads, not on the main/event-loop thread."""

    def test_flush_runs_in_worker_thread(self, db: Database, client: TestClient) -> None:
        """_flush_buffered_traces should execute in a non-main thread."""
        _seed_many_traces_run(db, "thr1", n_metrics=3)

        call_threads: list[str] = []

        def _tracking_flush(*args: object, **kwargs: object) -> list[dict]:
            call_threads.append(threading.current_thread().name)
            return _real_flush(*args, **kwargs)  # type: ignore[arg-type]

        with patch(
            "matyan_backend.api.runs._streaming._flush_buffered_traces",
            side_effect=_tracking_flush,
        ):
            resp = client.get(
                f"{BASE}/runs/search/metric/",
                params={"report_progress": "false", "trace_chunk_size": 1},
            )
        assert resp.status_code == 200
        assert len(call_threads) >= 3, f"expected >=3 flush calls, got {len(call_threads)}"
        for t_name in call_threads:
            assert t_name != "MainThread", f"_flush_buffered_traces ran on {t_name}, expected a worker thread"


# ---------------------------------------------------------------------------
# Run search regression test (asyncio.to_thread for get_run_bundles)
# ---------------------------------------------------------------------------


class TestRunSearchToThread:
    """Verify run search returns correct run frames after to_thread offload."""

    def test_multiple_runs_all_returned(self, db: Database, client: TestClient) -> None:
        """Seed 3 runs, GET run search, decode and assert 3 distinct run hashes."""
        hashes = ["rst1", "rst2", "rst3"]
        for h in hashes:
            _seed_metric_run(db, h)

        resp = client.get(
            f"{BASE}/runs/search/run/",
            params={"report_progress": "false"},
        )
        assert resp.status_code == 200
        body = resp.content
        for h in hashes:
            assert h.encode() in body, f"run {h} missing from run search response"


# ---------------------------------------------------------------------------
# Lazy metric loop — iterator in thread + queue tests
# ---------------------------------------------------------------------------


class TestLazyLoopTraceData:
    """Lazy path returns valid trace data for a run with 2+ metrics."""

    def test_lazy_path_returns_traces(self, db: Database, client: TestClient) -> None:
        """Seed one run with 3 metrics, force lazy path (metric.name query),
        decode stream and assert trace data with 'values' and 'iters' keys.
        """
        _seed_many_traces_run(db, "llt1", n_metrics=3)
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={
                "q": 'metric.name.startswith("metric")',
                "report_progress": "false",
            },
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        run_keys = [k for k, _v in pairs if _key_starts_with(k, "llt1")]
        assert len(run_keys) > 0, "expected data for run llt1"
        value_keys = [k for k in run_keys if _key_contains(k, "values")]
        iter_keys = [k for k in run_keys if _key_contains(k, "iters")]
        assert len(value_keys) >= 3, f"expected >=3 value keys, got {len(value_keys)}"
        assert len(iter_keys) >= 3, f"expected >=3 iter keys, got {len(iter_keys)}"


class TestLazyLoopEventLoopFree:
    """Event loop is not blocked during lazy metric search — iterator runs in a worker thread."""

    def test_iterator_runs_in_background_thread(self, db: Database, client: TestClient) -> None:
        """Patch iter_matching_sequences_with_bundle to record the thread it runs on;
        assert it is NOT the main thread.
        """
        _seed_many_traces_run(db, "llev1", n_metrics=2)

        producer_threads: list[str] = []
        real_iter = iter_matching_sequences_with_bundle

        def _tracking_iter(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            producer_threads.append(threading.current_thread().name)
            yield from real_iter(*args, **kwargs)

        with patch(
            "matyan_backend.api.runs._streaming.iter_matching_sequences_with_bundle",
            side_effect=_tracking_iter,
        ):
            resp = client.get(
                f"{BASE}/runs/search/metric/",
                params={
                    "q": 'metric.name.startswith("metric")',
                    "report_progress": "false",
                },
            )
        assert resp.status_code == 200
        assert len(producer_threads) >= 1, "iter_matching_sequences_with_bundle was not called"
        for t_name in producer_threads:
            assert t_name != "MainThread", (
                f"iter_matching_sequences_with_bundle ran on {t_name}, expected a worker thread"
            )

    def test_lazy_path_consumer_does_not_call_get_run_bundle(self, db: Database, client: TestClient) -> None:
        """Lazy metric path uses bundle from producer; consumer must not call get_run_bundle."""
        _seed_metric_run(db, "llev2")
        none_result = _PlanResult(candidates=None, exact=True)
        with (
            patch("matyan_backend.api.runs._streaming.plan_query", return_value=none_result),
            patch(
                "matyan_backend.api.runs._collections.get_run_bundle",
                wraps=runs.get_run_bundle,
            ) as mock_bundle,
        ):
            resp = client.get(
                f"{BASE}/runs/search/metric/",
                params={"report_progress": "false"},
            )
            assert resp.status_code == 200
        # Producer (iter_matching_sequences_with_bundle) calls get_run_bundle once per run; consumer does not.
        assert mock_bundle.call_count == 1, "get_run_bundle should be called once (producer only, 1 run)"


class TestLazyLoopMultiRun:
    """Lazy path with multiple runs aggregates correctly."""

    def test_two_runs_both_present(self, db: Database, client: TestClient) -> None:
        """Seed two runs with metrics, use a query matching both, assert both run hashes appear."""
        _seed_many_traces_run(db, "llm1", n_metrics=2)
        _seed_many_traces_run(db, "llm2", n_metrics=2)
        resp = client.get(
            f"{BASE}/runs/search/metric/",
            params={
                "q": 'metric.name.startswith("metric")',
                "report_progress": "false",
            },
        )
        assert resp.status_code == 200
        pairs = _decode_stream_pairs(resp.content)
        run1_keys = [k for k, _v in pairs if _key_starts_with(k, "llm1")]
        run2_keys = [k for k, _v in pairs if _key_starts_with(k, "llm2")]
        assert len(run1_keys) > 0, "run llm1 missing from lazy metric search"
        assert len(run2_keys) > 0, "run llm2 missing from lazy metric search"


# ---------------------------------------------------------------------------
# Run search — iterator runs in background thread
# ---------------------------------------------------------------------------


class TestRunSearchEventLoopFree:
    """Run search iterator runs in a worker thread, not on the main thread."""

    def test_iterator_runs_in_background_thread(self, db: Database, client: TestClient) -> None:
        """Patch iter_matching_runs to record thread name; assert not MainThread."""
        _seed_metric_run(db, "rsev1")

        producer_threads: list[str] = []
        real_iter = iter_matching_runs

        def _tracking_iter(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            producer_threads.append(threading.current_thread().name)
            yield from real_iter(*args, **kwargs)

        with patch(
            "matyan_backend.api.runs._streaming.iter_matching_runs",
            side_effect=_tracking_iter,
        ):
            resp = client.get(
                f"{BASE}/runs/search/run/",
                params={"report_progress": "false"},
            )
        assert resp.status_code == 200
        assert len(producer_threads) >= 1, "iter_matching_runs was not called"
        for t_name in producer_threads:
            assert t_name != "MainThread", f"iter_matching_runs ran on {t_name}, expected a worker thread"


class TestCancellationCleanup:
    """Verify that abandoned streaming generators cancel pending tasks."""

    def test_candidate_streamer_cancels_tasks_on_early_close(self, db: Database) -> None:
        """Closing the candidate streamer mid-flight must cancel pending process_one tasks."""
        release_event = threading.Event()
        created_tasks: list[asyncio.Task] = []
        _real_create_task = asyncio.create_task

        def tracking_create_task(coro: object, **kwargs: object) -> asyncio.Task:
            task = _real_create_task(coro, **kwargs)  # type: ignore[arg-type]
            created_tasks.append(task)
            return task

        async def slow_process_candidate_run(
            _db: object,
            run_hash: str,
            bundle: dict | None,
            num_points: int,
            x_axis: str | None,
            skip_system: bool,
            q: object | None,
            has_seq_pred: bool,
            tz_offset: int,
            trace_chunk_size: int,
            trace_names: frozenset[str] | None,
            timing: bool,
            counter: int,
        ) -> tuple[int, str, dict | None, float, float, bool]:
            del _db, bundle, num_points, x_axis, skip_system, q, has_seq_pred
            del tz_offset, trace_chunk_size, trace_names, timing
            await asyncio.to_thread(release_event.wait, 2.0)
            return counter, run_hash, None, 0.0, 0.0, True

        async def fast_fetch_candidate_batch(
            _db: object,
            batch: list[str],
        ) -> tuple[list[str], list[dict | None], float]:
            del _db
            return (
                batch,
                [{"meta": {}, "attrs": {}, "traces": [], "contexts": {}, "tags": [], "experiment": None}] * len(batch),
                0.01,
            )

        async def run_test() -> None:
            with (
                patch(
                    "matyan_backend.api.runs._streaming._process_candidate_run",
                    side_effect=slow_process_candidate_run,
                ),
                patch(
                    "matyan_backend.api.runs._streaming._fetch_candidate_batch",
                    side_effect=fast_fetch_candidate_batch,
                ),
                patch("matyan_backend.api.runs._streaming.encode_tree", side_effect=lambda data: data),
                patch(
                    "matyan_backend.api.runs._streaming.collect_streamable_data",
                    side_effect=lambda _data: b"PROGRESS",
                ),
                patch("asyncio.create_task", side_effect=tracking_create_task),
            ):
                streamer = _metric_search_candidate_streamer(
                    db,
                    ["c1", "c2", "c3", "c4"],
                    num_points=5,
                    x_axis=None,
                    skip_system=True,
                    report_progress=True,
                )
                await anext(streamer)
                await streamer.aclose()

            release_event.set()
            await asyncio.sleep(0.05)
            for t in created_tasks:
                assert t.done(), f"task {t.get_name()} was not cleaned up"

        asyncio.run(run_test())

    def test_run_search_cancels_bundles_task_on_early_close(self, db: Database) -> None:
        """Closing run_search mid-bundle-fetch must cancel the bundles_task."""
        release_bundles = threading.Event()
        bundles_task_cancelled = threading.Event()

        def fake_iter_matching_runs(*args: object, **kwargs: object):  # noqa: ANN202
            del args, kwargs
            for i in range(15):
                yield SimpleNamespace(hash=f"rs{i}"), (i + 1, 15)

        def fake_get_run_bundles(
            _db: object,
            hashes: list[str],
            *,
            include_attrs: bool = True,
            include_traces: bool = True,
        ) -> list[dict]:
            del _db, include_attrs, include_traces
            if not release_bundles.wait(timeout=2.0):
                bundles_task_cancelled.set()
            return [
                {"meta": {}, "attrs": {}, "traces": [], "contexts": {}, "tags": [], "experiment": None} for _ in hashes
            ]

        async def run_test() -> None:
            with (
                patch(
                    "matyan_backend.api.runs._streaming.iter_matching_runs",
                    side_effect=fake_iter_matching_runs,
                ),
                patch(
                    "matyan_backend.api.runs._streaming.get_run_bundles",
                    side_effect=fake_get_run_bundles,
                ),
                patch("matyan_backend.api.runs._streaming.encode_tree", side_effect=lambda data: data),
                patch(
                    "matyan_backend.api.runs._streaming.collect_streamable_data",
                    side_effect=lambda _data: b"CHUNK",
                ),
            ):
                response = await run_search_api(db, q="", report_progress=True)
                iterator = response.body_iterator
                typed_iterator: AsyncIterator[str | bytes | memoryview[int]] = iterator  # ty: ignore[invalid-assignment]
                await asyncio.wait_for(anext(typed_iterator), timeout=0.5)
                await iterator.aclose()

            release_bundles.set()
            await asyncio.sleep(0.1)

        asyncio.run(run_test())

    def test_thread_pool_not_saturated_after_cancelled_requests(self, db: Database) -> None:
        """Multiple cancelled candidate streamers must not permanently fill the thread pool."""

        async def fast_fetch_candidate_batch(
            _db: object,
            batch: list[str],
        ) -> tuple[list[str], list[dict | None], float]:
            del _db
            await asyncio.sleep(0.3)
            return (
                batch,
                [{"meta": {}, "attrs": {}, "traces": [], "contexts": {}, "tags": [], "experiment": None}] * len(batch),
                0.01,
            )

        async def slow_process_candidate_run(
            _db: object,
            run_hash: str,
            bundle: dict | None,
            num_points: int,
            x_axis: str | None,
            skip_system: bool,
            q: object | None,
            has_seq_pred: bool,
            tz_offset: int,
            trace_chunk_size: int,
            trace_names: frozenset[str] | None,
            timing: bool,
            counter: int,
        ) -> tuple[int, str, dict | None, float, float, bool]:
            del _db, bundle, num_points, x_axis, skip_system, q, has_seq_pred
            del tz_offset, trace_chunk_size, trace_names, timing
            await asyncio.sleep(0.5)
            return counter, run_hash, None, 0.0, 0.0, True

        async def run_test() -> None:
            with (
                patch(
                    "matyan_backend.api.runs._streaming._process_candidate_run",
                    side_effect=slow_process_candidate_run,
                ),
                patch(
                    "matyan_backend.api.runs._streaming._fetch_candidate_batch",
                    side_effect=fast_fetch_candidate_batch,
                ),
                patch("matyan_backend.api.runs._streaming.encode_tree", side_effect=lambda data: data),
                patch(
                    "matyan_backend.api.runs._streaming.collect_streamable_data",
                    side_effect=lambda _data: b"PROGRESS",
                ),
                patch(
                    "matyan_backend.api.runs._streaming.stream_tree_data",
                    side_effect=lambda _data: [b"RUN"],
                ),
            ):
                for _ in range(5):
                    streamer = _metric_search_candidate_streamer(
                        db,
                        [f"sat{j}" for j in range(10)],
                        num_points=5,
                        x_axis=None,
                        skip_system=True,
                        report_progress=True,
                    )
                    await anext(streamer)
                    await streamer.aclose()

                await asyncio.sleep(0.2)

                streamer = _metric_search_candidate_streamer(
                    db,
                    ["final0"],
                    num_points=5,
                    x_axis=None,
                    skip_system=True,
                    report_progress=True,
                    exact=True,
                )
                chunks = [chunk async for chunk in streamer]

            assert len(chunks) > 0, "final streamer produced no output — thread pool likely saturated"

        asyncio.run(run_test())


# ---------------------------------------------------------------------------
# Multi-key sort tests
# ---------------------------------------------------------------------------


def _extract_run_hashes_ordered(body: bytes) -> list[str]:
    """Extract run hashes from a run-search stream in the order they appear.

    Each run's data begins with a key whose first path component is the run
    hash.  Progress frames start with ``progress`` and are skipped.
    """
    pairs = _decode_stream_pairs(body)
    seen: list[str] = []
    for key, _val in pairs:
        first_component = key.split(_SENTINEL)[0].decode("utf-8", errors="replace")
        if first_component.startswith("progress"):
            continue
        if first_component not in seen:
            seen.append(first_component)
    return seen


class TestRunSearchSort:
    """Verify server-side multi-key sorting of run search results."""

    @staticmethod
    def _seed_named_run(db: Database, run_hash: str, *, name: str, created_at: float | None = None) -> None:
        runs.create_run(db, run_hash, name=name)
        runs.set_context(db, run_hash, 0, {})
        runs.set_run_attrs(db, run_hash, (), {"hparams": {"lr": 0.01}})
        runs.set_trace_info(db, run_hash, 0, "loss", dtype="float", last=0.5, last_step=4)
        for i in range(5):
            sequences.write_sequence_step(db, run_hash, 0, "loss", i, 1.0 - i * 0.2, epoch=0, timestamp=1000.0 + i)
        if created_at is not None:
            runs.update_run_meta(db, run_hash, created_at=created_at)

    def test_sort_by_name_asc(self, db: Database, client: TestClient) -> None:
        """Runs sorted by name ascending should appear in alphabetical order."""
        self._seed_named_run(db, "sn1", name="charlie")
        self._seed_named_run(db, "sn2", name="alpha")
        self._seed_named_run(db, "sn3", name="bravo")

        sort_param = '[{"field":"run","order":"asc"}]'
        resp = client.get(
            f"{BASE}/runs/search/run/",
            params={"sort": sort_param, "report_progress": "false"},
        )
        assert resp.status_code == 200
        hashes = _extract_run_hashes_ordered(resp.content)
        assert hashes == ["sn2", "sn3", "sn1"], f"expected alpha,bravo,charlie order, got {hashes}"

    def test_sort_by_name_desc(self, db: Database, client: TestClient) -> None:
        """Runs sorted by name descending should appear in reverse alphabetical order."""
        self._seed_named_run(db, "snd1", name="charlie")
        self._seed_named_run(db, "snd2", name="alpha")
        self._seed_named_run(db, "snd3", name="bravo")

        sort_param = '[{"field":"run","order":"desc"}]'
        resp = client.get(
            f"{BASE}/runs/search/run/",
            params={"sort": sort_param, "report_progress": "false"},
        )
        assert resp.status_code == 200
        hashes = _extract_run_hashes_ordered(resp.content)
        assert hashes == ["snd1", "snd3", "snd2"], f"expected charlie,bravo,alpha order, got {hashes}"

    def test_sort_by_date_desc(self, db: Database, client: TestClient) -> None:
        """Runs sorted by date descending should have newest first."""
        self._seed_named_run(db, "sd1", name="first", created_at=1000.0)
        self._seed_named_run(db, "sd2", name="second", created_at=2000.0)
        self._seed_named_run(db, "sd3", name="third", created_at=3000.0)

        sort_param = '[{"field":"date","order":"desc"}]'
        resp = client.get(
            f"{BASE}/runs/search/run/",
            params={"sort": sort_param, "report_progress": "false"},
        )
        assert resp.status_code == 200
        hashes = _extract_run_hashes_ordered(resp.content)
        assert hashes == ["sd3", "sd2", "sd1"], f"expected newest-first, got {hashes}"

    def test_sort_by_date_asc(self, db: Database, client: TestClient) -> None:
        """Runs sorted by date ascending should have oldest first."""
        self._seed_named_run(db, "sda1", name="first", created_at=1000.0)
        self._seed_named_run(db, "sda2", name="second", created_at=2000.0)
        self._seed_named_run(db, "sda3", name="third", created_at=3000.0)

        sort_param = '[{"field":"date","order":"asc"}]'
        resp = client.get(
            f"{BASE}/runs/search/run/",
            params={"sort": sort_param, "report_progress": "false"},
        )
        assert resp.status_code == 200
        hashes = _extract_run_hashes_ordered(resp.content)
        assert hashes == ["sda1", "sda2", "sda3"], f"expected oldest-first, got {hashes}"

    def test_multi_key_sort(self, db: Database, client: TestClient) -> None:
        """Multi-key sort: experiment asc then name desc within each experiment."""
        exp_a = create_experiment(db, "exp-aaa")
        exp_b = create_experiment(db, "exp-bbb")

        runs.create_run(db, "mk1", name="zulu", experiment_id=exp_a["id"])
        runs.create_run(db, "mk2", name="alpha", experiment_id=exp_a["id"])
        runs.create_run(db, "mk3", name="mike", experiment_id=exp_b["id"])
        runs.create_run(db, "mk4", name="bravo", experiment_id=exp_b["id"])

        for h in ["mk1", "mk2", "mk3", "mk4"]:
            runs.set_context(db, h, 0, {})
            runs.set_trace_info(db, h, 0, "loss", dtype="float", last=0.5, last_step=0)

        sort_param = '[{"field":"experiment","order":"asc"},{"field":"run","order":"desc"}]'
        resp = client.get(
            f"{BASE}/runs/search/run/",
            params={"sort": sort_param, "report_progress": "false"},
        )
        assert resp.status_code == 200
        hashes = _extract_run_hashes_ordered(resp.content)
        assert hashes == ["mk1", "mk2", "mk3", "mk4"], (
            f"expected exp-aaa(zulu,alpha) then exp-bbb(mike,bravo), got {hashes}"
        )

    def test_sort_with_offset_and_limit(self, db: Database, client: TestClient) -> None:
        """Offset and limit should apply after sorting."""
        self._seed_named_run(db, "sol1", name="alpha", created_at=1000.0)
        self._seed_named_run(db, "sol2", name="bravo", created_at=2000.0)
        self._seed_named_run(db, "sol3", name="charlie", created_at=3000.0)
        self._seed_named_run(db, "sol4", name="delta", created_at=4000.0)

        sort_param = '[{"field":"run","order":"asc"}]'
        resp = client.get(
            f"{BASE}/runs/search/run/",
            params={"sort": sort_param, "offset": "sol2", "limit": "2", "report_progress": "false"},
        )
        assert resp.status_code == 200
        hashes = _extract_run_hashes_ordered(resp.content)
        assert hashes == ["sol3", "sol4"], f"expected charlie,delta after offset=bravo with limit=2, got {hashes}"

    def test_default_sort_is_date_desc(self, db: Database, client: TestClient) -> None:
        """With no sort param, results should come back in date descending order."""
        self._seed_named_run(db, "dsd1", name="aaa", created_at=1000.0)
        self._seed_named_run(db, "dsd2", name="bbb", created_at=3000.0)
        self._seed_named_run(db, "dsd3", name="ccc", created_at=2000.0)

        resp = client.get(
            f"{BASE}/runs/search/run/",
            params={"report_progress": "false"},
        )
        assert resp.status_code == 200
        hashes = _extract_run_hashes_ordered(resp.content)
        assert hashes == ["dsd2", "dsd3", "dsd1"], f"expected newest-first default order, got {hashes}"

    def test_invalid_sort_field_returns_400(self, client: TestClient) -> None:
        """An unknown sort field should be rejected with HTTP 400."""
        sort_param = '[{"field":"nonexistent","order":"asc"}]'
        resp = client.get(
            f"{BASE}/runs/search/run/",
            params={"sort": sort_param},
        )
        assert resp.status_code == 400

    def test_invalid_sort_json_returns_400(self, client: TestClient) -> None:
        """Malformed JSON in the sort param should be rejected with HTTP 400."""
        resp = client.get(
            f"{BASE}/runs/search/run/",
            params={"sort": "not-valid-json"},
        )
        assert resp.status_code == 400
