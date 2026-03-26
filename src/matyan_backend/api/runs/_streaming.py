"""Streaming search endpoints for runs and metrics.

All endpoints return ``StreamingResponse`` using the binary codec from
``api.streaming`` which the Aim UI understands.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import struct as _struct
import time
from typing import TYPE_CHECKING, Literal

from fastapi import Header, HTTPException
from loguru import logger
from starlette.responses import StreamingResponse

from matyan_backend.api.streaming import (
    PROGRESS_REPORT_INTERVAL,
    collect_streamable_data,
    encode_tree,
    make_progress_key,
    stream_tree_data,
)
from matyan_backend.config import SETTINGS
from matyan_backend.deps import FdbDb  # noqa: TC001
from matyan_backend.metrics import PLANNER_PATH_TOTAL
from matyan_backend.storage.entities import get_run_experiment_names
from matyan_backend.storage.indexes import lookup_by_active
from matyan_backend.storage.runs import (
    get_all_contexts,
    get_metric_search_bundle,
    get_run,
    get_run_bundles,
    get_run_meta,
    get_run_traces_info,
)
from matyan_backend.storage.sequences import read_and_sample_sequence, read_sequence, sample_sequences_batch
from matyan_backend.thread_pool import FDB_EXECUTOR, to_fdb_thread

from ._collections import (
    _NON_METRIC_DTYPES,
    RunHashRef,
    _log_timing,
    iter_matching_runs,
    iter_matching_sequences_with_bundle,
)
from ._planner import plan_query, query_has_sequence_level_predicate, query_has_unindexed_sequence_predicate
from ._pydantic_models import MetricAlignApiIn, RunTracesBatchApiIn  # noqa: TC001
from ._query import RestrictedPythonQuery, prepare_query
from ._range_utils import parse_range
from ._run import rest_router_runs
from ._views import RunView, SequenceView, build_props_dict, build_props_from_bundle

if TYPE_CHECKING:
    import ast
    from collections.abc import AsyncGenerator

_SLEEP = 0.00001

TRACE_CHUNK_SIZE_DEFAULT = 10


class _ProgressReporter:
    """Track and encode progress frames with optional heartbeat re-emission."""

    def __init__(self, *, enabled: bool, total: int | None = None) -> None:
        self.enabled = enabled
        self.total = total
        self.progress_idx = 0
        self.latest_checked = 0
        self.last_progress_time = 0.0
        self._last_emitted: tuple[int, int] | None = None

    def update(self, checked: int, total: int | None = None) -> None:
        """Store the latest observed progress without emitting a frame."""
        self.latest_checked = checked
        if total is not None:
            self.total = total

    def emit(
        self,
        checked: int | None = None,
        total: int | None = None,
        *,
        force: bool = False,
    ) -> bytes | None:
        """Return an encoded progress frame, or ``None`` when disabled/skipped."""
        if checked is not None:
            self.latest_checked = checked
        if total is not None:
            self.total = total
        if not self.enabled or self.total is None:
            return None
        current = (self.latest_checked, self.total)
        if not force and self._last_emitted == current:
            return None
        payload = collect_streamable_data(
            encode_tree({make_progress_key(self.progress_idx): current}),
        )
        self.progress_idx += 1
        self.last_progress_time = time.time()
        self._last_emitted = current
        return payload

    def heartbeat(self) -> bytes | None:
        """Re-emit the latest progress after a quiet interval."""
        if not self.enabled or self.total is None:
            return None
        if time.time() - self.last_progress_time < PROGRESS_REPORT_INTERVAL:
            return None
        return self.emit(force=True)

    def finish(self) -> bytes | None:
        """Emit the terminal ``(0, 0)`` progress frame."""
        return self.emit(0, 0, force=True)


def _numpy_to_encodable(values: list) -> dict:
    """Pack a list of numbers into the ``EncodedNumpyArray`` format the UI expects.

    Returns ``{"type": "numpy", "shape": N, "dtype": "float64", "blob": <bytes>}``
    where *blob* is little-endian float64 binary data.
    """
    floats = [float(v) for v in values]
    return {
        "type": "numpy",
        "shape": len(floats),
        "dtype": "float64",
        "blob": _struct.pack(f"<{len(floats)}d", *floats) if floats else b"",
    }


# ---------------------------------------------------------------------------
# Sort specification
# ---------------------------------------------------------------------------

_VALID_SORT_FIELDS = frozenset({"run", "experiment", "hash", "date", "duration", "description", "creator"})


@dataclasses.dataclass(frozen=True, slots=True)
class SortSpec:
    """A single sort key: field name + direction."""

    field: str
    order: Literal["asc", "desc"] = "desc"


_DEFAULT_SORT: list[SortSpec] = [SortSpec(field="date", order="desc")]


def _parse_sort_param(raw: str) -> list[SortSpec]:
    """Parse a JSON-encoded sort parameter into a list of :class:`SortSpec`.

    :param raw: JSON string, e.g. ``'[{"field":"run","order":"asc"}]'``.
    :returns: Parsed list; falls back to :data:`_DEFAULT_SORT` on empty input.
    :raises HTTPException: On malformed JSON or invalid field/order values.
    """
    if not raw:
        return _DEFAULT_SORT
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid sort JSON: {exc}") from exc
    if not isinstance(items, list) or not items:
        return _DEFAULT_SORT
    specs: list[SortSpec] = []
    for item in items:
        field = item.get("field", "")
        order = item.get("order", "asc")
        if field not in _VALID_SORT_FIELDS:
            raise HTTPException(status_code=400, detail=f"Unknown sort field: {field!r}")
        if order not in ("asc", "desc"):
            raise HTTPException(status_code=400, detail=f"Invalid sort order: {order!r}")
        specs.append(SortSpec(field=field, order=order))
    return specs or _DEFAULT_SORT


def _extract_sort_value(  # noqa: PLR0911
    meta: dict,
    run_hash: str,
    field: str,
    experiment_name: str,
) -> str | float:
    """Return a comparable sort value for *field*, mirroring the UI's ``getRunSortValue``.

    :param meta: Run metadata dict from ``get_run_meta``.
    :param run_hash: The run's hash string.
    :param field: One of the :data:`_VALID_SORT_FIELDS`.
    :param experiment_name: Pre-resolved experiment name for this run.
    :returns: A string or float suitable for comparison.
    """
    if field == "run":
        return meta.get("name") or ""
    if field == "experiment":
        return experiment_name or ""
    if field == "hash":
        return run_hash
    if field == "date":
        return meta.get("created_at") or 0.0
    if field == "duration":
        created = meta.get("created_at") or 0.0
        ended = meta.get("finalized_at") or time.time()
        return ended - created
    if field == "description":
        return meta.get("description") or ""
    if field == "creator":
        return ""
    return ""


# ---------------------------------------------------------------------------
# Run search
# ---------------------------------------------------------------------------


_RUN_BUNDLE_BATCH_SIZE = 10


def _bundle_to_run_dict(
    run_hash: str,
    bundle: dict,
    *,
    skip_system: bool,
    exclude_params: bool,
    exclude_traces: bool,
) -> dict:
    """Build a streamable run-dict from a pre-fetched bundle."""
    meta = bundle["meta"]
    meta["hash"] = run_hash

    params = bundle["attrs"] or {}
    if isinstance(params, dict):
        params.pop("__blobs__", None)
        if skip_system:
            params.pop("__system_params", None)

    metric_traces: list[dict] = []
    if not exclude_traces:
        contexts = bundle["contexts"]
        for t in bundle["traces"]:
            dtype = t.get("dtype", "")
            if dtype in _NON_METRIC_DTYPES:
                continue
            ctx_id = t.get("context_id", 0)
            ctx = contexts.get(ctx_id, {})
            last_val = t.get("last", 0.0)
            metric_traces.append(
                {
                    "name": t.get("name", ""),
                    "context": ctx,
                    "values": {
                        "last": last_val,
                        "last_step": t.get("last_step", 0),
                        "first": last_val,
                        "min": last_val,
                        "max": last_val,
                    },
                },
            )

    return {
        run_hash: {
            "params": params if not exclude_params else {},
            "traces": {"metric": metric_traces},
            "props": build_props_from_bundle(meta, bundle["tags"], bundle["experiment"]),
        },
    }


def _run_search_producer(  # noqa: C901
    db: FdbDb,
    prepared_ast: ast.Expression,
    tz_offset: int,
    sort_specs: list[SortSpec],
    queue: asyncio.Queue[tuple | None],
    loop: asyncio.AbstractEventLoop,
    stop_flag: list[bool],
) -> None:
    """Collect matching runs, sort them, and push sorted hashes into *queue*.

    Phase 1 — Collect all matching run hashes from ``iter_matching_runs``.
    Phase 2 — Batch-load metadata and experiment names needed for sorting.
    Phase 3 — Multi-pass stable sort (reverse priority order).
    Phase 4 — Push sorted hashes into the queue with sequential progress tuples.
    """
    try:
        matches: list[str] = []
        for rv, _progress in iter_matching_runs(db, prepared_ast=prepared_ast, tz_offset=tz_offset):
            if stop_flag[0]:
                break
            matches.append(rv.hash)

        if stop_flag[0] or not matches:
            return

        needs_experiment = any(s.field == "experiment" for s in sort_specs)
        needs_meta = any(s.field != "hash" for s in sort_specs)

        metas: dict[str, dict] = {}
        if needs_meta:
            for h in matches:
                if stop_flag[0]:
                    return
                metas[h] = get_run_meta(db, h)

        exp_names: dict[str, str] = {}
        if needs_experiment:
            exp_names = get_run_experiment_names(db, matches, run_metas=metas or None)

        for spec in reversed(sort_specs):
            matches.sort(
                key=lambda h, _f=spec.field: _extract_sort_value(
                    metas.get(h, {}),
                    h,
                    _f,
                    exp_names.get(h, ""),
                ),
                reverse=(spec.order == "desc"),
            )

        total = len(matches)
        for counter, h in enumerate(matches, 1):
            if stop_flag[0]:
                break
            asyncio.run_coroutine_threadsafe(queue.put((RunHashRef(h), (counter, total))), loop).result()
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()


@rest_router_runs.get("/search/run/")
async def run_search_api(  # noqa: C901, PLR0915
    db: FdbDb,
    q: str = "",
    limit: int = 0,
    offset: str = "",
    sort: str = "",
    skip_system: bool = True,
    report_progress: bool = True,
    exclude_params: bool = False,
    exclude_traces: bool = False,
    x_timezone_offset: int = Header(default=0),
) -> StreamingResponse:
    """Search runs with optional multi-key sorting.

    :param sort: JSON-encoded sort specification, e.g.
        ``'[{"field":"run","order":"asc"},{"field":"date","order":"desc"}]'``.
        Supported fields: ``run``, ``experiment``, ``hash``, ``date``,
        ``duration``, ``description``, ``creator``.
        Defaults to ``[{"field":"date","order":"desc"}]``.
    """
    try:
        prepared_ast = prepare_query(q, x_timezone_offset)
    except SyntaxError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid query syntax: {exc}") from exc

    sort_specs = _parse_sort_param(sort)
    inc_attrs = not exclude_params
    inc_traces = not exclude_traces

    async def _streamer() -> AsyncGenerator[bytes]:  # noqa: C901, PLR0912, PLR0915
        queue: asyncio.Queue[tuple | None] = asyncio.Queue(maxsize=SETTINGS.run_search_queue_maxsize)
        stop_flag: list[bool] = [False]
        loop = asyncio.get_running_loop()

        producer_future = loop.run_in_executor(
            FDB_EXECUTOR,
            _run_search_producer,
            db,
            prepared_ast,
            x_timezone_offset,
            sort_specs,
            queue,
            loop,
            stop_flag,
        )

        count = 0
        past_offset = not offset
        done = False

        buffer: list[tuple[str, tuple[int, int]]] = []
        progress_reporter = _ProgressReporter(enabled=report_progress)
        bundles_task: asyncio.Task | None = None

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break

                rv, progress = item
                checked, total = progress
                if progress_reporter.total is None:
                    initial_progress = progress_reporter.emit(0, total)
                    if initial_progress is not None:
                        yield initial_progress
                progress_reporter.update(checked, total)

                if not past_offset:
                    if rv.hash == offset:
                        past_offset = True
                    continue

                buffer.append((rv.hash, progress))

                if len(buffer) >= _RUN_BUNDLE_BATCH_SIZE:
                    hashes = [h for h, _ in buffer]
                    bundles_task = asyncio.create_task(
                        to_fdb_thread(
                            get_run_bundles,
                            db,
                            hashes,
                            include_attrs=inc_attrs,
                            include_traces=inc_traces,
                        ),
                    )
                    while not bundles_task.done():
                        done_set, _ = await asyncio.wait({bundles_task}, timeout=PROGRESS_REPORT_INTERVAL)
                        if done_set:
                            break
                        heartbeat = progress_reporter.heartbeat()
                        if heartbeat is not None:
                            yield heartbeat
                    bundles = bundles_task.result()
                    bundles_task = None
                    for (rh, prog), bundle in zip(buffer, bundles, strict=True):
                        await asyncio.sleep(_SLEEP)
                        if bundle is None:
                            continue
                        run_dict = _bundle_to_run_dict(
                            rh,
                            bundle,
                            skip_system=skip_system,
                            exclude_params=exclude_params,
                            exclude_traces=exclude_traces,
                        )
                        yield collect_streamable_data(encode_tree(run_dict))
                        if report_progress:
                            progress_frame = progress_reporter.emit(*prog)
                            if progress_frame is not None:
                                yield progress_frame
                        count += 1
                        if limit and count >= limit:
                            done = True
                            break
                    buffer.clear()
                    if done:
                        break

            if buffer and not done:
                hashes = [h for h, _ in buffer]
                bundles_task = asyncio.create_task(
                    to_fdb_thread(
                        get_run_bundles,
                        db,
                        hashes,
                        include_attrs=inc_attrs,
                        include_traces=inc_traces,
                    ),
                )
                while not bundles_task.done():
                    done_set, _ = await asyncio.wait({bundles_task}, timeout=PROGRESS_REPORT_INTERVAL)
                    if done_set:
                        break
                    heartbeat = progress_reporter.heartbeat()
                    if heartbeat is not None:
                        yield heartbeat
                bundles = bundles_task.result()
                bundles_task = None
                for (rh, prog), bundle in zip(buffer, bundles, strict=True):
                    await asyncio.sleep(_SLEEP)
                    if bundle is None:
                        continue
                    run_dict = _bundle_to_run_dict(
                        rh,
                        bundle,
                        skip_system=skip_system,
                        exclude_params=exclude_params,
                        exclude_traces=exclude_traces,
                    )
                    yield collect_streamable_data(encode_tree(run_dict))
                    if report_progress:
                        progress_frame = progress_reporter.emit(*prog)
                        if progress_frame is not None:
                            yield progress_frame
                    count += 1
                    if limit and count >= limit:
                        break
        finally:
            stop_flag[0] = True
            if bundles_task is not None and not bundles_task.done():
                bundles_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await bundles_task
            await producer_future

        if report_progress:
            final_progress = progress_reporter.finish()
            if final_progress is not None:
                yield final_progress

    return StreamingResponse(_streamer(), media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Metric search
# ---------------------------------------------------------------------------


class _MetricTraceRef:
    """Lightweight stand-in for ``SequenceView`` used by the candidate path.

    Provides the ``.name``, ``._context``, and ``._trace_info`` interface
    that ``_flush_buffered_traces`` and ``_build_trace_view_from_sampled``
    expect, without needing a full ``RunView`` or FDB reads.
    """

    __slots__ = ("_context", "_trace_info", "name")

    def __init__(self, name: str, context: dict, context_id: int) -> None:
        self.name = name
        self._context = context
        self._trace_info: dict = {"context_id": context_id}


def _build_run_data_from_bundle(
    bundle: dict,
    run_hash: str,
    *,
    skip_system: bool,
) -> dict:
    meta = bundle["meta"].copy()
    meta["hash"] = run_hash
    run_params = bundle["attrs"] or {}
    if isinstance(run_params, dict):
        run_params.pop("__blobs__", None)
        if skip_system:
            run_params.pop("__system_params", None)
    return {
        "params": run_params,
        "traces": [],
        "props": build_props_from_bundle(meta, bundle["tags"], bundle["experiment"]),
    }


def _build_trace_view_from_sampled(
    sv: _MetricTraceRef,
    sampled: dict[str, list],
    x_seq: dict[str, list] | None,
) -> dict:
    """Build a single trace-view dict from pre-sampled data (no FDB calls)."""
    steps = sampled.get("steps", [])
    trace_view: dict = {
        "name": sv.name,  # type: ignore[attr-defined]
        "context": sv._context,  # type: ignore[attr-defined]  # noqa: SLF001
        "slice": [0, steps[-1] if steps else 0, 1],
        "values": _numpy_to_encodable(sampled.get("val", [])),
        "iters": _numpy_to_encodable(steps),
        "epochs": _numpy_to_encodable(sampled.get("epoch", [])),
        "timestamps": _numpy_to_encodable(sampled.get("time", [])),
    }
    if x_seq is not None:
        trace_view["x_axis_values"] = _numpy_to_encodable(x_seq.get("val", []))
        trace_view["x_axis_iters"] = _numpy_to_encodable(x_seq.get("steps", []))
    return trace_view


def _flush_buffered_traces(
    db: FdbDb,
    run_hash: str,
    buffered_svs: list[_MetricTraceRef],
    num_points: int,
    x_axis: str | None,
    prefetched_x_axis: dict[int, dict[str, list]] | None = None,
) -> list[dict]:
    """Sample all buffered traces for one run in a single FDB transaction."""
    requests = [
        (sv._trace_info.get("context_id", 0), sv.name)  # noqa: SLF001
        for sv in buffered_svs
    ]
    main_results, x_axis_results = sample_sequences_batch(
        db,
        run_hash,
        requests,
        num_points,
        columns=("val",),
        x_axis_name=x_axis if prefetched_x_axis is None else None,
    )
    x_axis_results = prefetched_x_axis if prefetched_x_axis is not None else x_axis_results
    traces: list[dict] = []
    for sv in buffered_svs:
        ctx_id: int = sv._trace_info.get("context_id", 0)  # noqa: SLF001
        sampled = main_results.get((ctx_id, sv.name), {"steps": [], "val": []})
        x_seq = x_axis_results.get(ctx_id) if x_axis_results is not None else None
        traces.append(_build_trace_view_from_sampled(sv, sampled, x_seq))
    return traces


def _adaptive_trace_chunk_size(trace_count: int, num_points: int, trace_chunk_size: int) -> int:
    """Pick a larger chunk size for low-density requests to reduce overhead."""
    if trace_count <= trace_chunk_size:
        return trace_count
    if num_points <= trace_chunk_size:
        return min(trace_count, trace_chunk_size * 2)
    if num_points <= trace_chunk_size * 4:
        return min(trace_count, trace_chunk_size + (trace_chunk_size // 2))
    return min(trace_count, trace_chunk_size)


def _prefetch_x_axis_results(
    db: FdbDb,
    run_hash: str,
    buffered_svs: list,
    num_points: int,
    x_axis: str,
) -> dict[int, dict[str, list]]:
    """Sample x-axis sequences once per context for a run."""
    ctx_ids = list(dict.fromkeys(sv._trace_info.get("context_id", 0) for sv in buffered_svs))  # noqa: SLF001
    if not ctx_ids:
        return {}
    main_results, _ = sample_sequences_batch(
        db,
        run_hash,
        [(ctx_id, x_axis) for ctx_id in ctx_ids],
        num_points,
        columns=("val",),
        x_axis_name=x_axis,
    )
    return {ctx_id: main_results.get((ctx_id, x_axis), {"steps": [], "val": []}) for ctx_id in ctx_ids}


async def _collect_traces_async(
    db: FdbDb,
    run_hash: str,
    buffered_svs: list[_MetricTraceRef],
    num_points: int,
    x_axis: str | None,
    trace_chunk_size: int,
) -> list[dict]:
    """Fetch trace data from FDB using parallel thread-pool tasks.

    When *buffered_svs* has at most *trace_chunk_size* traces, samples them
    in a single FDB transaction.  Otherwise splits into chunks and runs
    ``_flush_buffered_traces`` per chunk via ``asyncio.to_thread``, merging
    results in order.
    """
    if not buffered_svs:
        return []
    effective_chunk_size = _adaptive_trace_chunk_size(len(buffered_svs), num_points, trace_chunk_size)
    if len(buffered_svs) <= effective_chunk_size:
        return await to_fdb_thread(
            _flush_buffered_traces,
            db,
            run_hash,
            buffered_svs,
            num_points,
            x_axis,
        )
    chunks = [buffered_svs[i : i + effective_chunk_size] for i in range(0, len(buffered_svs), effective_chunk_size)]
    prefetched_x_axis = None
    if x_axis is not None:
        prefetched_x_axis = await to_fdb_thread(
            _prefetch_x_axis_results,
            db,
            run_hash,
            buffered_svs,
            num_points,
            x_axis,
        )
    chunk_limit = max(1, min(SETTINGS.metric_trace_chunk_concurrency, len(chunks)))
    semaphore = asyncio.Semaphore(chunk_limit)

    async def process_chunk(chunk: list) -> list[dict]:
        async with semaphore:
            return await to_fdb_thread(
                _flush_buffered_traces,
                db,
                run_hash,
                chunk,
                num_points,
                x_axis,
                prefetched_x_axis,
            )

    chunk_results = await asyncio.gather(*(process_chunk(chunk) for chunk in chunks))
    return [trace for result in chunk_results for trace in result]


async def _fetch_candidate_batch(db: FdbDb, batch: list[str]) -> tuple[list[str], list[dict | None], float]:
    """Fetch one candidate bundle batch and report elapsed time."""
    t_start = time.perf_counter()
    bundles = await to_fdb_thread(get_run_bundles, db, batch)
    return batch, bundles, time.perf_counter() - t_start


async def _process_candidate_run(
    db: FdbDb,
    run_hash: str,
    bundle: dict | None,
    num_points: int,
    x_axis: str | None,
    skip_system: bool,
    q: RestrictedPythonQuery | None,
    has_seq_pred: bool,
    tz_offset: int,
    trace_chunk_size: int,
    trace_names: frozenset[str] | None,
    timing: bool,
    counter: int,
) -> tuple[int, str, dict | None, float, float, bool]:
    """Process one candidate run and return streamable data plus timings."""
    if bundle is None:
        return counter, run_hash, None, 0.0, 0.0, False

    t_build_check = time.perf_counter()
    trace_refs: list[_MetricTraceRef]
    if q is not None and not has_seq_pred:
        meta = bundle["meta"].copy()
        meta["hash"] = run_hash
        rv = RunView(db, run_hash, meta, tz_offset=tz_offset)
        if not q.check(run=rv):
            dt_build_check = time.perf_counter() - t_build_check
            if timing:
                _log_timing(
                    "build_run_view_and_check",
                    dt_build_check,
                    path="candidate",
                    endpoint="metric_search",
                    run_hash=run_hash,
                    run_index=counter,
                )
            return counter, run_hash, None, dt_build_check, 0.0, True

    if q is not None and has_seq_pred:
        meta = bundle["meta"].copy()
        meta["hash"] = run_hash
        rv = RunView(db, run_hash, meta, tz_offset=tz_offset)
        contexts = bundle["contexts"]
        trace_refs = _filter_traces_with_check(
            bundle,
            rv,
            q,
            trace_names=trace_names,
            contexts=contexts,
        )
    else:
        trace_refs = _metric_traces_from_bundle(bundle, trace_names=trace_names)

    dt_build_check = time.perf_counter() - t_build_check
    if timing:
        _log_timing(
            "build_run_view_and_check",
            dt_build_check,
            path="candidate",
            endpoint="metric_search",
            run_hash=run_hash,
            run_index=counter,
        )
    if not trace_refs:
        return counter, run_hash, None, dt_build_check, 0.0, True

    run_data = _build_run_data_from_bundle(bundle, run_hash, skip_system=skip_system)
    t_collect_traces = time.perf_counter()
    traces = await _collect_traces_async(
        db,
        run_hash,
        trace_refs,
        num_points,
        x_axis,
        trace_chunk_size,
    )
    dt_collect_traces = time.perf_counter() - t_collect_traces
    if timing:
        _log_timing(
            "collect_traces",
            dt_collect_traces,
            path="candidate",
            endpoint="metric_search",
            run_hash=run_hash,
            run_index=counter,
        )
    run_data["traces"] = iter(traces)
    return counter, run_hash, run_data, dt_build_check, dt_collect_traces, True


def _metric_traces_from_bundle(
    bundle: dict,
    *,
    trace_names: frozenset[str] | None = None,
) -> list[_MetricTraceRef]:
    """Extract metric-type trace descriptors from a pre-fetched bundle.

    When *trace_names* is not ``None``, only traces whose name is in the
    set are included (used when the planner resolved a ``metric.name``
    predicate via the trace index).
    """
    contexts = bundle["contexts"]
    refs: list[_MetricTraceRef] = []
    for t in bundle["traces"]:
        if t.get("dtype", "") in _NON_METRIC_DTYPES:
            continue
        name = t.get("name", "")
        if trace_names is not None and name not in trace_names:
            continue
        ctx_id = t.get("context_id", 0)
        refs.append(_MetricTraceRef(name, contexts.get(ctx_id, {}), ctx_id))
    return refs


def _filter_traces_with_check(
    bundle: dict,
    rv: RunView,
    q: RestrictedPythonQuery,
    *,
    trace_names: frozenset[str] | None = None,
    contexts: dict[int, dict] | None = None,
) -> list[_MetricTraceRef]:
    """Build trace refs and filter them using ``q.check(run=rv, metric=sv)``.

    For each metric-type trace in the bundle, builds a real ``SequenceView``
    and evaluates the full query. Returns only the trace refs that pass.
    """
    if contexts is None:
        contexts = bundle["contexts"]
    refs: list[_MetricTraceRef] = []
    for t in bundle["traces"]:
        if t.get("dtype", "") in _NON_METRIC_DTYPES:
            continue
        name = t.get("name", "")
        if trace_names is not None and name not in trace_names:
            continue
        ctx_id = t.get("context_id", 0)
        ctx = contexts.get(ctx_id, {})
        sv = SequenceView(name, ctx, rv, trace_info=t)
        if q.check(run=rv, metric=sv):
            refs.append(_MetricTraceRef(name, ctx, ctx_id))
    return refs


@rest_router_runs.get("/search/metric/")
async def metric_search_api(
    db: FdbDb,
    q: str = "",
    p: int = 50,
    x_axis: str | None = None,
    skip_system: bool = True,
    report_progress: bool = True,
    trace_chunk_size: int = TRACE_CHUNK_SIZE_DEFAULT,
    x_timezone_offset: int = Header(default=0),
) -> StreamingResponse:
    try:
        prepared_ast = prepare_query(q, x_timezone_offset)
    except SyntaxError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid query syntax: {exc}") from exc

    chunk_size = max(1, min(trace_chunk_size, 500))

    t0 = time.perf_counter()
    result = await to_fdb_thread(plan_query, db, prepared_ast)
    candidates = result.candidates
    had_candidates_before_reset = candidates is not None
    if candidates is not None and query_has_unindexed_sequence_predicate(prepared_ast):
        candidates = None

    if candidates is not None:
        if SETTINGS.query_timing_enabled:
            _log_timing("plan_query", time.perf_counter() - t0, path="candidate", endpoint="metric_search")
        if SETTINGS.metrics_enabled:
            PLANNER_PATH_TOTAL.labels(path="fast", endpoint="metric_search", reason="").inc()
        logger.trace(
            "metric search: candidate path (n_candidates={}, exact={}, trace_names={}, q={!r})",
            len(candidates),
            result.exact,
            result.trace_names,
            q,
        )
        return StreamingResponse(
            _metric_search_candidate_streamer(
                db,
                candidates,
                p,
                x_axis,
                skip_system,
                report_progress,
                exact=result.exact,
                prepared_ast=prepared_ast,
                tz_offset=x_timezone_offset,
                trace_chunk_size=chunk_size,
                trace_names=result.trace_names,
                timing=SETTINGS.query_timing_enabled,
            ),
            media_type="application/octet-stream",
        )

    lazy_reason = "unindexed_sequence" if had_candidates_before_reset else "no_candidates"
    if SETTINGS.metrics_enabled:
        PLANNER_PATH_TOTAL.labels(path="lazy", endpoint="metric_search", reason=lazy_reason).inc()
    logger.debug("metric search: lazy path (reason={}, q_len={})", lazy_reason, len(q))
    logger.trace("metric search: lazy path (q={!r})", q)
    return StreamingResponse(
        _metric_search_lazy_streamer(
            db,
            q,
            p,
            x_axis,
            skip_system,
            report_progress,
            x_timezone_offset,
            trace_chunk_size=chunk_size,
            prepared_ast=prepared_ast,
        ),
        media_type="application/octet-stream",
    )


async def _metric_search_candidate_streamer(  # noqa: C901, PLR0912, PLR0915
    db: FdbDb,
    candidates: list[str],
    num_points: int,
    x_axis: str | None,
    skip_system: bool,
    report_progress: bool,
    *,
    exact: bool = True,
    prepared_ast: ast.Expression | None = None,
    tz_offset: int = 0,
    trace_chunk_size: int = TRACE_CHUNK_SIZE_DEFAULT,
    trace_names: frozenset[str] | None = None,
    timing: bool = False,
) -> AsyncGenerator[bytes]:
    """Fast path: iterate run hashes from the planner.

    When *exact* is ``False`` (superset from planner), each candidate is
    verified with ``RestrictedPythonQuery.check()`` before streaming.

    When *trace_names* is set (from an indexed ``metric.name`` predicate),
    only the matching traces are streamed.

    FDB work is offloaded to worker threads so the event loop stays free:

    * Bundle reads are batched via ``get_run_bundles`` (one FDB transaction
      per batch of ``_RUN_BUNDLE_BATCH_SIZE`` runs).
    * Trace sampling runs in parallel thread-pool tasks (one per trace
      chunk) via ``_collect_traces_async``.
    """
    q = RestrictedPythonQuery(prepared_ast=prepared_ast) if (not exact and prepared_ast is not None) else None
    has_seq_pred = (
        query_has_sequence_level_predicate(prepared_ast) if (q is not None and prepared_ast is not None) else False
    )

    total = len(candidates)
    t_start = time.perf_counter()
    sum_fetch_bundles = 0.0
    sum_build_check = 0.0
    sum_collect_traces = 0.0
    runs_checked = 0
    runs_matched = 0
    progress_reporter = _ProgressReporter(enabled=report_progress, total=total)
    progress_checked = 0

    if report_progress and total > 0:
        initial_progress = progress_reporter.emit(0, total)
        if initial_progress is not None:
            yield initial_progress

    _pending: list[asyncio.Task] = []
    try:
        if exact and total == 1:
            run_hash = candidates[0]
            t_fetch_bundle = time.perf_counter()
            bundle_task = asyncio.create_task(to_fdb_thread(get_metric_search_bundle, db, run_hash))
            _pending.append(bundle_task)
            while not bundle_task.done():
                done_set, _ = await asyncio.wait({bundle_task}, timeout=PROGRESS_REPORT_INTERVAL)
                if done_set:
                    break
                heartbeat = progress_reporter.heartbeat()
                if heartbeat is not None:
                    yield heartbeat
            bundle = bundle_task.result()
            if timing:
                dt_fetch_bundle = time.perf_counter() - t_fetch_bundle
                sum_fetch_bundles += dt_fetch_bundle
                _log_timing(
                    "fetch_bundles",
                    dt_fetch_bundle,
                    path="candidate",
                    endpoint="metric_search",
                    extra="batch_size=1",
                )

            if bundle is not None:
                runs_checked += 1
                t_build_check = time.perf_counter()
                trace_refs = _metric_traces_from_bundle(bundle, trace_names=trace_names)
                dt_build_check = time.perf_counter() - t_build_check
                if timing:
                    sum_build_check += dt_build_check
                    _log_timing(
                        "build_run_view_and_check",
                        dt_build_check,
                        path="candidate",
                        endpoint="metric_search",
                        run_hash=run_hash,
                        run_index=1,
                    )
                if trace_refs:
                    run_data = _build_run_data_from_bundle(bundle, run_hash, skip_system=skip_system)
                    t_collect_traces = time.perf_counter()
                    traces = await _collect_traces_async(
                        db,
                        run_hash,
                        trace_refs,
                        num_points,
                        x_axis,
                        trace_chunk_size,
                    )
                    dt_collect_traces = time.perf_counter() - t_collect_traces
                    if timing:
                        sum_collect_traces += dt_collect_traces
                        _log_timing(
                            "collect_traces",
                            dt_collect_traces,
                            path="candidate",
                            endpoint="metric_search",
                            run_hash=run_hash,
                            run_index=1,
                        )
                    run_data["traces"] = iter(traces)
                    runs_matched += 1
                    for chunk in stream_tree_data(encode_tree({run_hash: run_data})):
                        yield chunk
                progress_checked = 1
                if report_progress:
                    progress_frame = progress_reporter.emit(progress_checked, total)
                    if progress_frame is not None:
                        yield progress_frame

            if timing:
                _log_timing(
                    "summary",
                    time.perf_counter() - t_start,
                    path="candidate",
                    endpoint="metric_search",
                    extra=(
                        f"runs_checked={runs_checked} runs_matched={runs_matched} "
                        f"total_fetch_bundles_sec={sum_fetch_bundles:.6f} "
                        f"total_build_check_sec={sum_build_check:.6f} "
                        f"total_collect_traces_sec={sum_collect_traces:.6f}"
                    ),
                )
            if report_progress:
                final_progress = progress_reporter.finish()
                if final_progress is not None:
                    yield final_progress
            return

        batches = [
            candidates[batch_start : batch_start + _RUN_BUNDLE_BATCH_SIZE]
            for batch_start in range(0, total, _RUN_BUNDLE_BATCH_SIZE)
        ]
        if not batches:
            if timing:
                _log_timing(
                    "summary",
                    time.perf_counter() - t_start,
                    path="candidate",
                    endpoint="metric_search",
                    extra=(
                        f"runs_checked={runs_checked} runs_matched={runs_matched} "
                        f"total_fetch_bundles_sec={sum_fetch_bundles:.6f} "
                        f"total_build_check_sec={sum_build_check:.6f} "
                        f"total_collect_traces_sec={sum_collect_traces:.6f}"
                    ),
                )
            if report_progress:
                final_progress = progress_reporter.finish()
                if final_progress is not None:
                    yield final_progress
            return

        prefetched_task: asyncio.Task | None = asyncio.create_task(
            _fetch_candidate_batch(db, batches[0]),
        )
        _pending.append(prefetched_task)
        while not prefetched_task.done():
            done_set, _ = await asyncio.wait({prefetched_task}, timeout=PROGRESS_REPORT_INTERVAL)
            if done_set:
                break
            heartbeat = progress_reporter.heartbeat()
            if heartbeat is not None:
                yield heartbeat
        prefetched_batch = prefetched_task.result()
        for batch_index, _batch in enumerate(batches):
            current_batch, bundles, dt_fetch_bundles = prefetched_batch
            if batch_index + 1 < len(batches):
                prefetched_task = asyncio.create_task(_fetch_candidate_batch(db, batches[batch_index + 1]))
                _pending.append(prefetched_task)
                await asyncio.wait({prefetched_task}, timeout=0.001)
            else:
                prefetched_task = None
            if timing:
                sum_fetch_bundles += dt_fetch_bundles
                _log_timing(
                    "fetch_bundles",
                    dt_fetch_bundles,
                    path="candidate",
                    endpoint="metric_search",
                    extra=f"batch_size={len(current_batch)}",
                )
            semaphore = asyncio.Semaphore(
                max(1, min(SETTINGS.metric_candidate_run_concurrency, len(current_batch))),
            )

            async def process_one(
                counter: int,
                run_hash: str,
                bundle: dict | None,
                *,
                run_semaphore: asyncio.Semaphore = semaphore,
            ) -> tuple[int, str, dict | None, float, float, bool]:
                nonlocal progress_checked
                async with run_semaphore:
                    await asyncio.sleep(_SLEEP)
                    result = await _process_candidate_run(
                        db,
                        run_hash,
                        bundle,
                        num_points,
                        x_axis,
                        skip_system,
                        q,
                        has_seq_pred,
                        tz_offset,
                        trace_chunk_size,
                        trace_names,
                        timing,
                        counter,
                    )
                    progress_checked += 1
                    return result

            batch_start = batch_index * _RUN_BUNDLE_BATCH_SIZE
            tasks = [
                asyncio.create_task(
                    process_one(
                        batch_start + idx_in_batch + 1,
                        run_hash,
                        bundle,
                    ),
                )
                for idx_in_batch, (run_hash, bundle) in enumerate(zip(current_batch, bundles, strict=True))
            ]
            _pending.extend(tasks)

            for task in tasks:
                while not task.done():
                    done_set, _ = await asyncio.wait({task}, timeout=PROGRESS_REPORT_INTERVAL)
                    if done_set:
                        break
                    heartbeat = progress_reporter.heartbeat()
                    if heartbeat is not None:
                        yield heartbeat
                _counter, run_hash, run_data, dt_build_check, dt_collect_traces, was_checked = task.result()
                if not was_checked:
                    progress_frame = progress_reporter.emit(progress_checked, total)
                    if progress_frame is not None:
                        yield progress_frame
                    continue
                runs_checked += 1
                sum_build_check += dt_build_check
                sum_collect_traces += dt_collect_traces
                progress_frame = progress_reporter.emit(progress_checked, total)
                if progress_frame is not None:
                    yield progress_frame
                if run_data is None:
                    continue
                runs_matched += 1
                for chunk in stream_tree_data(encode_tree({run_hash: run_data})):
                    yield chunk

            if prefetched_task is not None:
                while not prefetched_task.done():
                    done_set, _ = await asyncio.wait({prefetched_task}, timeout=PROGRESS_REPORT_INTERVAL)
                    if done_set:
                        break
                    heartbeat = progress_reporter.heartbeat()
                    if heartbeat is not None:
                        yield heartbeat
                prefetched_batch = prefetched_task.result()

        if timing:
            _log_timing(
                "summary",
                time.perf_counter() - t_start,
                path="candidate",
                endpoint="metric_search",
                extra=(
                    f"runs_checked={runs_checked} runs_matched={runs_matched} "
                    f"total_fetch_bundles_sec={sum_fetch_bundles:.6f} "
                    f"total_build_check_sec={sum_build_check:.6f} "
                    f"total_collect_traces_sec={sum_collect_traces:.6f}"
                ),
            )

        if report_progress:
            final_progress = progress_reporter.finish()
            if final_progress is not None:
                yield final_progress
    finally:
        for t in _pending:
            t.cancel()
        for t in _pending:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t


_LAZY_QUEUE_SENTINEL = None


def _lazy_iterator_producer(
    db: FdbDb,
    q: str,
    tz_offset: int,
    queue: asyncio.Queue[tuple | None],
    loop: asyncio.AbstractEventLoop,
    stop_flag: list[bool],
    prepared_ast: ast.Expression | None = None,
) -> None:
    """Run ``iter_matching_sequences_with_bundle`` synchronously, pushing items into *queue*.

    Yields (rv, sv, progress, bundle) so the consumer can use the bundle without
    calling get_run_bundle.  Puts ``_LAZY_QUEUE_SENTINEL`` when the iterator is
    exhausted or the consumer sets *stop_flag[0]* to ``True``.
    """
    try:
        for rv, sv, progress, bundle in iter_matching_sequences_with_bundle(
            db,
            q,
            tz_offset=tz_offset,
            seq_type="metric",
            prepared_ast=prepared_ast,
        ):
            if stop_flag[0]:
                break
            asyncio.run_coroutine_threadsafe(queue.put((rv, sv, progress, bundle)), loop).result()
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(_LAZY_QUEUE_SENTINEL), loop).result()


async def _metric_search_lazy_streamer(  # noqa: C901
    db: FdbDb,
    q: str,
    num_points: int,
    x_axis: str | None,
    skip_system: bool,
    report_progress: bool,
    tz_offset: int,
    *,
    trace_chunk_size: int = TRACE_CHUNK_SIZE_DEFAULT,
    prepared_ast: ast.Expression | None = None,
) -> AsyncGenerator[bytes]:
    """Slow path: full MatyanQL evaluation via iter_matching_sequences_with_bundle.

    The synchronous iterator runs in a background thread and pushes
    (rv, sv, progress, bundle) into an ``asyncio.Queue``.  The consumer
    uses the bundle from the queue (no get_run_bundle call).  Trace
    sampling is offloaded via ``asyncio.to_thread`` / ``_collect_traces_async``.
    """
    queue: asyncio.Queue[tuple | None] = asyncio.Queue(maxsize=SETTINGS.lazy_metric_queue_maxsize)
    stop_flag: list[bool] = [False]
    loop = asyncio.get_running_loop()

    producer_future = loop.run_in_executor(
        FDB_EXECUTOR,
        _lazy_iterator_producer,
        db,
        q,
        tz_offset,
        queue,
        loop,
        stop_flag,
        prepared_ast,
    )

    progress_idx = 0
    last_progress_time = time.time()

    current_hash: str | None = None
    current_data: dict | None = None
    buffered_svs: list = []

    try:
        while True:
            item = await queue.get()
            if item is _LAZY_QUEUE_SENTINEL:
                break

            rv, sv, (checked, total), bundle = item

            if report_progress and time.time() - last_progress_time > PROGRESS_REPORT_INTERVAL:
                yield collect_streamable_data(encode_tree({make_progress_key(progress_idx): (checked, total)}))
                progress_idx += 1
                last_progress_time = time.time()

            run_hash = rv.hash

            if run_hash != current_hash:
                if current_hash is not None and current_data is not None:
                    traces = await _collect_traces_async(
                        db,
                        current_hash,
                        buffered_svs,
                        num_points,
                        x_axis,
                        trace_chunk_size,
                    )
                    current_data["traces"] = iter(traces)
                    for chunk in stream_tree_data(encode_tree({current_hash: current_data})):
                        yield chunk
                    if report_progress:
                        yield collect_streamable_data(
                            encode_tree({make_progress_key(progress_idx): (checked, total)}),
                        )
                        progress_idx += 1
                        last_progress_time = time.time()

                current_hash = run_hash
                buffered_svs = []
                current_data = _build_run_data_from_bundle(bundle, run_hash, skip_system=skip_system)

            buffered_svs.append(sv)

        if current_hash is not None and current_data is not None:
            traces = await _collect_traces_async(
                db,
                current_hash,
                buffered_svs,
                num_points,
                x_axis,
                trace_chunk_size,
            )
            current_data["traces"] = iter(traces)
            for chunk in stream_tree_data(encode_tree({current_hash: current_data})):
                yield chunk

        if report_progress:
            yield collect_streamable_data(encode_tree({make_progress_key(progress_idx): (0, 0)}))
    finally:
        stop_flag[0] = True
        await producer_future


# ---------------------------------------------------------------------------
# Active runs
# ---------------------------------------------------------------------------


def _build_active_run_payload(db: FdbDb, rh: str) -> dict | None:
    """Build the streamed dict for one active run (all FDB, sync)."""
    meta = get_run_meta(db, rh)
    if not meta or meta.get("pending_deletion"):
        return None
    meta["hash"] = rh
    traces_info = get_run_traces_info(db, rh)
    contexts = get_all_contexts(db, rh)

    metric_traces: list[dict] = []
    for t in traces_info:
        ctx_id = t.get("context_id", 0)
        ctx = contexts.get(ctx_id, {})
        metric_traces.append(
            {
                "name": t.get("name", ""),
                "context": ctx,
                "values": {
                    "last": t.get("last", 0.0),
                    "last_step": t.get("last_step", 0),
                },
            },
        )

    return {
        rh: {
            "traces": {"metric": metric_traces},
            "props": build_props_dict(meta, db),
        },
    }


@rest_router_runs.get("/active/")
async def active_runs_api(
    db: FdbDb,
    report_progress: bool = True,
) -> StreamingResponse:
    async def _streamer() -> AsyncGenerator[bytes]:
        progress_idx = 0
        active_hashes = await to_fdb_thread(lookup_by_active, db, True)
        total = len(active_hashes)

        for idx, rh in enumerate(active_hashes, start=1):
            await asyncio.sleep(_SLEEP)
            run_dict = await to_fdb_thread(_build_active_run_payload, db, rh)
            if run_dict is None:
                continue

            yield collect_streamable_data(encode_tree(run_dict))

            if report_progress:
                yield collect_streamable_data(encode_tree({make_progress_key(progress_idx): (idx, total)}))
                progress_idx += 1

    return StreamingResponse(_streamer(), media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Metric alignment
# ---------------------------------------------------------------------------


def _build_align_traces_list(
    db: FdbDb,
    run_hash: str,
    trace_requests: list,
    align_by: str,
) -> list[dict] | None:
    """Build aligned traces for one run (all FDB, sync).

    Returns ``None`` when the run does not exist.
    """
    run = get_run(db, run_hash)
    if not run:
        return None

    traces_info = get_run_traces_info(db, run_hash)
    contexts = get_all_contexts(db, run_hash)

    traces_list: list[dict] = []
    for trace_req in trace_requests:
        ctx_id: int | None = None
        for t in traces_info:
            c = contexts.get(t.get("context_id", 0), {})
            if t.get("name") == trace_req.name and c == trace_req.context:
                ctx_id = t.get("context_id", 0)
                break
        if ctx_id is None:
            continue

        x_ctx_id: int | None = None
        for t in traces_info:
            c = contexts.get(t.get("context_id", 0), {})
            if t.get("name") == align_by and c == trace_req.context:
                x_ctx_id = t.get("context_id", 0)
                break
        if x_ctx_id is None:
            continue

        max_step = trace_req.slice[1] if trace_req.slice[1] > 0 else None
        x_seq = read_sequence(db, run_hash, x_ctx_id, align_by, end_step=max_step)
        traces_list.append(
            {
                "name": trace_req.name,
                "context": trace_req.context,
                "x_axis_values": _numpy_to_encodable(x_seq.get("val", [])),
                "x_axis_iters": _numpy_to_encodable(x_seq.get("steps", [])),
            },
        )
    return traces_list


@rest_router_runs.post("/search/metric/align/")
async def metric_custom_align_api(request_data: MetricAlignApiIn, db: FdbDb) -> StreamingResponse:
    async def _streamer() -> AsyncGenerator[bytes]:
        for run_data in request_data.runs:
            await asyncio.sleep(_SLEEP)
            traces_list = await to_fdb_thread(
                _build_align_traces_list,
                db,
                run_data.run_id,
                run_data.traces,
                request_data.align_by,
            )
            if traces_list is None:
                continue
            yield collect_streamable_data(encode_tree({run_data.run_id: traces_list}))

    return StreamingResponse(_streamer(), media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Metric batch
# ---------------------------------------------------------------------------


def _fetch_metric_batch_result(
    db: FdbDb,
    run_id: str,
    requests: list[tuple[str, dict]],
    start_step: int | None,
    end_step: int | None,
    density: int,
) -> list[dict] | None:
    """Load and sample requested traces for one run (all FDB, sync).

    Returns ``None`` when the run does not exist.
    *requests* is a list of ``(name, context_dict)`` tuples.
    """
    run = get_run(db, run_id)
    if not run:
        return None

    traces_info = get_run_traces_info(db, run_id)
    contexts = get_all_contexts(db, run_id)

    results: list[dict] = []
    for req_name, req_context in requests:
        ctx_id: int | None = None
        for t in traces_info:
            c = contexts.get(t.get("context_id", 0), {})
            if t.get("name") == req_name and c == req_context:
                ctx_id = t.get("context_id", 0)
                break
        if ctx_id is None:
            continue

        seq = read_and_sample_sequence(
            db,
            run_id,
            ctx_id,
            req_name,
            start_step=start_step,
            end_step=end_step,
            density=density,
        )
        results.append(
            {
                "name": req_name,
                "context": req_context,
                "iters": seq.get("steps", []),
                "values": seq.get("val", []),
            },
        )
    return results


@rest_router_runs.post("/{run_id}/metric/get-batch/")
async def run_metric_batch_api(
    run_id: str,
    body: RunTracesBatchApiIn,
    db: FdbDb,
    record_range: str = "",
    record_density: int = 50,
) -> list[dict]:
    rr = parse_range(record_range)
    effective_end = rr.stop - 1 if rr.stop is not None else None
    requests = [(req.name, req.context) for req in body]

    results = await to_fdb_thread(
        _fetch_metric_batch_result,
        db,
        run_id,
        requests,
        rr.start,
        effective_end,
        record_density,
    )
    if results is None:
        raise HTTPException(status_code=404)
    return results
