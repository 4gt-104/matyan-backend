"""Run/sequence collection iterators backed by FDB.

Provides query-filtered iteration over runs, yielding per-run sequence
collections together with progress tuples for streaming progress reporting.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

from matyan_backend.config import SETTINGS
from matyan_backend.metrics import PLANNER_PATH_TOTAL
from matyan_backend.storage.indexes import iter_run_hashes_from_index
from matyan_backend.storage.runs import get_all_contexts, get_run_bundle, get_run_meta, get_run_traces_info

from ._planner import plan_query
from ._query import RestrictedPythonQuery, prepare_query
from ._views import RunView, SequenceView

if TYPE_CHECKING:
    import ast
    from collections.abc import Iterator

    from matyan_backend.fdb_types import Database

# ---------------------------------------------------------------------------
# Query timing helper
# ---------------------------------------------------------------------------

_TIMING_PREFIX = "(QUERY_TIMING)"


def _log_timing(
    step: str,
    duration_sec: float,
    *,
    path: str = "",
    endpoint: str = "",
    run_hash: str = "",
    run_index: int = 0,
    extra: str = "",
) -> None:
    """Emit a single structured timing line when ``query_timing_enabled``."""
    parts = [
        _TIMING_PREFIX,
        f"step={step}",
        f"path={path}",
        f"endpoint={endpoint}",
        f"run_index={run_index}",
    ]
    if run_hash:
        parts.append(f"run_hash={run_hash}")
    parts.append(f"duration_sec={duration_sec:.6f}")
    if extra:
        parts.append(extra)
    logger.info(" ".join(parts))


# ---------------------------------------------------------------------------
# Run hash reference
# ---------------------------------------------------------------------------


class RunHashRef(NamedTuple):
    """Lightweight reference yielded by the candidate-list path of
    ``iter_matching_runs``.  Carries only ``.hash`` so the streamer
    can call ``get_run_bundle`` without a redundant ``get_run_meta``.
    """

    hash: str


# ---------------------------------------------------------------------------
# iter_matching_runs
# ---------------------------------------------------------------------------


def iter_matching_runs(  # noqa: C901, PLR0912, PLR0915
    db: Database,
    query: str = "",
    *,
    tz_offset: int = 0,
    prepared_ast: ast.Expression | None = None,
) -> Iterator[tuple[RunView | RunHashRef, tuple[int, int]]]:
    """Iterate all runs, yielding only those matching the MatyanQL *query*.

    Yields ``(run_ref, (counter, total))`` tuples so callers can report
    progress.

    Three paths:

    1. **Exact candidate list** (``result.candidates`` with ``exact=True``):
       Yield ``RunHashRef`` directly — no ``get_run_meta`` or ``check()``.
    2. **Superset candidate list** (``result.candidates`` with ``exact=False``):
       Iterate candidates, load meta, build ``RunView``, call ``q.check()``
       and yield only runs that pass.
    3. **Lazy path** (``result.candidates is None``):
       Walk the ``created_at`` index, build ``RunView`` per run, call
       ``q.check()`` and yield matching runs.
    """
    timing = SETTINGS.query_timing_enabled
    _ep = "run_search"

    if prepared_ast is None:
        prepared_ast = prepare_query(query, tz_offset)

    t0 = time.perf_counter()
    result = plan_query(db, prepared_ast)
    if timing:
        path_label = (
            "exact"
            if (result.candidates is not None and result.exact)
            else ("superset" if result.candidates is not None else "lazy")
        )
        _log_timing("plan_query", time.perf_counter() - t0, path=path_label, endpoint=_ep)

    if SETTINGS.metrics_enabled:
        if result.candidates is not None and result.exact:
            PLANNER_PATH_TOTAL.labels(path="fast", endpoint="run_search", reason="exact").inc()
        elif result.candidates is not None:
            PLANNER_PATH_TOTAL.labels(path="fast", endpoint="run_search", reason="superset").inc()
        else:
            PLANNER_PATH_TOTAL.labels(path="lazy", endpoint="run_search", reason="no_candidates").inc()

    if result.candidates is not None and result.exact:
        total = len(result.candidates)
        for counter, h in enumerate(result.candidates, 1):
            yield RunHashRef(h), (counter, total)

    elif result.candidates is not None:
        q = RestrictedPythonQuery(prepared_ast=prepared_ast)
        total = len(result.candidates)

        sum_fetch = 0.0
        sum_build = 0.0
        sum_check = 0.0
        runs_checked = 0
        runs_matched = 0

        for counter, h in enumerate(result.candidates, 1):
            t1 = time.perf_counter()
            meta = get_run_meta(db, h)
            dt_fetch = time.perf_counter() - t1

            if not meta or meta.get("pending_deletion"):
                continue
            meta["hash"] = h

            t2 = time.perf_counter()
            rv = RunView(db, h, meta, tz_offset=tz_offset)
            dt_build = time.perf_counter() - t2

            t3 = time.perf_counter()
            matched = q.check(run=rv)
            dt_check = time.perf_counter() - t3

            runs_checked += 1
            if timing:
                sum_fetch += dt_fetch
                sum_build += dt_build
                sum_check += dt_check
                _log_timing("fetch_meta", dt_fetch, path="superset", endpoint=_ep, run_hash=h, run_index=counter)
                _log_timing("build_run_view", dt_build, path="superset", endpoint=_ep, run_hash=h, run_index=counter)
                _log_timing("check", dt_check, path="superset", endpoint=_ep, run_hash=h, run_index=counter)

            if matched:
                runs_matched += 1
                yield rv, (counter, total)

        if timing:
            _log_timing(
                "summary",
                sum_fetch + sum_build + sum_check,
                path="superset",
                endpoint=_ep,
                extra=(
                    f"runs_checked={runs_checked} runs_matched={runs_matched} "
                    f"total_fetch_sec={sum_fetch:.6f} total_build_sec={sum_build:.6f} "
                    f"total_check_sec={sum_check:.6f}"
                ),
            )

    else:
        logger.debug("iter_matching_runs: lazy path (no_candidates)")
        hashes = iter_run_hashes_from_index(db)
        q = RestrictedPythonQuery(prepared_ast=prepared_ast)

        sum_fetch = 0.0
        sum_build = 0.0
        sum_check = 0.0
        runs_checked = 0
        runs_matched = 0
        counter = 0

        for h in hashes:
            counter += 1

            t1 = time.perf_counter()
            meta = get_run_meta(db, h)
            dt_fetch = time.perf_counter() - t1

            if not meta or meta.get("pending_deletion"):
                continue
            meta["hash"] = h

            t2 = time.perf_counter()
            rv = RunView(db, h, meta, tz_offset=tz_offset)
            dt_build = time.perf_counter() - t2

            t3 = time.perf_counter()
            matched = q.check(run=rv)
            dt_check = time.perf_counter() - t3

            runs_checked += 1
            if timing:
                sum_fetch += dt_fetch
                sum_build += dt_build
                sum_check += dt_check
                _log_timing("fetch_meta", dt_fetch, path="lazy", endpoint=_ep, run_hash=h, run_index=counter)
                _log_timing("build_run_view", dt_build, path="lazy", endpoint=_ep, run_hash=h, run_index=counter)
                _log_timing("check", dt_check, path="lazy", endpoint=_ep, run_hash=h, run_index=counter)

            if matched:
                runs_matched += 1
                yield rv, (counter, 0)

        if timing:
            _log_timing(
                "summary",
                sum_fetch + sum_build + sum_check,
                path="lazy",
                endpoint=_ep,
                extra=(
                    f"runs_checked={runs_checked} runs_matched={runs_matched} "
                    f"total_fetch_sec={sum_fetch:.6f} total_build_sec={sum_build:.6f} "
                    f"total_check_sec={sum_check:.6f}"
                ),
            )


_SEQ_TYPE_TO_DTYPES: dict[str, frozenset[str]] = {
    "images": frozenset({"image"}),
    "texts": frozenset({"text"}),
    "distributions": frozenset({"distribution"}),
    "audios": frozenset({"audio"}),
    "figures": frozenset({"figure"}),
}

_NON_METRIC_DTYPES = frozenset().union(
    *_SEQ_TYPE_TO_DTYPES.values(),
    {"logs", "log_records"},
)


def iter_matching_sequences(  # noqa: C901, PLR0912, PLR0915
    db: Database,
    query: str = "",
    *,
    tz_offset: int = 0,
    seq_type: str = "metric",
    prepared_ast: ast.Expression | None = None,
) -> Iterator[tuple[RunView, SequenceView, tuple[int, int]]]:
    """Iterate all runs and their traces, yielding matching run+sequence pairs.

    Yields ``(run_view, seq_view, (counter, total))`` for every matching
    sequence across all runs.  Both the superset and lazy paths build
    ``RunView`` and call ``q.check()``, so filtering is always applied.
    """
    timing = SETTINGS.query_timing_enabled
    _ep = "iter_matching_sequences"

    allowed_dtypes = _SEQ_TYPE_TO_DTYPES.get(seq_type)

    if prepared_ast is None:
        prepared_ast = prepare_query(query, tz_offset)

    t0 = time.perf_counter()
    result = plan_query(db, prepared_ast)

    if SETTINGS.metrics_enabled:
        if result.candidates is not None:
            PLANNER_PATH_TOTAL.labels(path="fast", endpoint="iter_matching_sequences", reason="").inc()
        else:
            PLANNER_PATH_TOTAL.labels(path="lazy", endpoint="iter_matching_sequences", reason="no_candidates").inc()
            logger.debug("iter_matching_sequences: lazy path (no_candidates)")

    if result.candidates is not None:
        hashes = result.candidates
        total = len(result.candidates)
        path_label = "superset"
    else:
        hashes = iter_run_hashes_from_index(db)
        total = 0
        path_label = "lazy"

    if timing:
        _log_timing("plan_query", time.perf_counter() - t0, path=path_label, endpoint=_ep)

    q = RestrictedPythonQuery(prepared_ast=prepared_ast)
    counter = 0

    sum_fetch_meta = 0.0
    sum_fetch_traces = 0.0
    sum_fetch_contexts = 0.0
    sum_build = 0.0
    sum_trace_loop = 0.0
    runs_checked = 0
    runs_matched = 0
    total_traces_checked = 0
    total_trace_matches = 0

    for h in hashes:
        counter += 1

        t1 = time.perf_counter()
        meta = get_run_meta(db, h)
        dt_fetch_meta = time.perf_counter() - t1

        if not meta or meta.get("pending_deletion"):
            continue
        meta["hash"] = h

        t2 = time.perf_counter()
        rv = RunView(db, h, meta, tz_offset=tz_offset)
        dt_build = time.perf_counter() - t2

        t3 = time.perf_counter()
        traces = get_run_traces_info(db, h)
        dt_fetch_traces = time.perf_counter() - t3

        t4 = time.perf_counter()
        contexts = get_all_contexts(db, h)
        dt_fetch_contexts = time.perf_counter() - t4

        runs_checked += 1
        run_traces_checked = 0
        run_trace_matches = 0

        t5 = time.perf_counter()
        for t in traces:
            dtype = t.get("dtype", "")
            if allowed_dtypes is not None:
                if dtype not in allowed_dtypes:
                    continue
            elif dtype in _NON_METRIC_DTYPES:
                continue

            ctx_id = t.get("context_id", 0)
            ctx = contexts.get(ctx_id, {})
            sv = SequenceView(t.get("name", ""), ctx, rv, trace_info=t)

            run_traces_checked += 1
            matched = q.check(run=rv, **{seq_type: sv})
            if matched:
                run_trace_matches += 1
                yield rv, sv, (counter, total)
        dt_trace_loop = time.perf_counter() - t5

        total_traces_checked += run_traces_checked
        total_trace_matches += run_trace_matches
        if run_trace_matches:
            runs_matched += 1

        if timing:
            sum_fetch_meta += dt_fetch_meta
            sum_fetch_traces += dt_fetch_traces
            sum_fetch_contexts += dt_fetch_contexts
            sum_build += dt_build
            sum_trace_loop += dt_trace_loop
            _kw = {"path": path_label, "endpoint": _ep, "run_hash": h, "run_index": counter}
            _log_timing("fetch_meta", dt_fetch_meta, **_kw)
            _log_timing("build_run_view", dt_build, **_kw)
            _log_timing("fetch_traces", dt_fetch_traces, **_kw)
            _log_timing("fetch_contexts", dt_fetch_contexts, **_kw)
            _log_timing(
                "trace_loop",
                dt_trace_loop,
                path=path_label,
                endpoint=_ep,
                run_hash=h,
                run_index=counter,
                extra=f"traces_checked={run_traces_checked} matches={run_trace_matches}",
            )

    if timing:
        _log_timing(
            "summary",
            sum_fetch_meta + sum_fetch_traces + sum_fetch_contexts + sum_build + sum_trace_loop,
            path=path_label,
            endpoint=_ep,
            extra=(
                f"runs_checked={runs_checked} runs_matched={runs_matched} "
                f"traces_checked={total_traces_checked} trace_matches={total_trace_matches} "
                f"total_fetch_meta_sec={sum_fetch_meta:.6f} total_fetch_traces_sec={sum_fetch_traces:.6f} "
                f"total_fetch_contexts_sec={sum_fetch_contexts:.6f} total_build_sec={sum_build:.6f} "
                f"total_trace_loop_sec={sum_trace_loop:.6f}"
            ),
        )


def iter_matching_sequences_with_bundle(  # noqa: C901, PLR0912, PLR0915
    db: Database,
    query: str = "",
    *,
    tz_offset: int = 0,
    seq_type: str = "metric",
    prepared_ast: ast.Expression | None = None,
) -> Iterator[tuple[RunView, SequenceView, tuple[int, int], dict]]:
    """Like iter_matching_sequences but yields (rv, sv, progress, bundle) for each match.

    Uses one get_run_bundle per run instead of get_run_meta + get_run_traces_info
    + get_all_contexts. Intended for the lazy metric streamer so the consumer
    can use the bundle without re-fetching.
    """
    timing = SETTINGS.query_timing_enabled
    _ep = "iter_matching_sequences_with_bundle"

    allowed_dtypes = _SEQ_TYPE_TO_DTYPES.get(seq_type)

    if prepared_ast is None:
        prepared_ast = prepare_query(query, tz_offset)

    t0 = time.perf_counter()
    result = plan_query(db, prepared_ast)

    if SETTINGS.metrics_enabled:
        if result.candidates is not None:
            PLANNER_PATH_TOTAL.labels(path="fast", endpoint="iter_matching_sequences", reason="").inc()
        else:
            PLANNER_PATH_TOTAL.labels(path="lazy", endpoint="iter_matching_sequences", reason="no_candidates").inc()
            logger.debug("iter_matching_sequences: lazy path (no_candidates)")

    if result.candidates is not None:
        hashes = result.candidates
        total = len(result.candidates)
        path_label = "superset"
    else:
        hashes = iter_run_hashes_from_index(db)
        total = 0
        path_label = "lazy"

    if timing:
        _log_timing("plan_query", time.perf_counter() - t0, path=path_label, endpoint=_ep)

    q = RestrictedPythonQuery(prepared_ast=prepared_ast)
    counter = 0

    sum_fetch_bundle = 0.0
    sum_build = 0.0
    sum_trace_loop = 0.0
    runs_checked = 0
    runs_matched = 0
    total_traces_checked = 0
    total_trace_matches = 0

    for h in hashes:
        counter += 1

        t1 = time.perf_counter()
        bundle = get_run_bundle(db, h)
        dt_fetch_bundle = time.perf_counter() - t1

        if bundle is None:
            continue

        meta = bundle["meta"]
        meta["hash"] = h
        traces = bundle["traces"]
        contexts = bundle["contexts"]

        t2 = time.perf_counter()
        rv = RunView(db, h, meta, tz_offset=tz_offset)
        dt_build = time.perf_counter() - t2

        runs_checked += 1
        run_traces_checked = 0
        run_trace_matches = 0

        t5 = time.perf_counter()
        for t in traces:
            dtype = t.get("dtype", "")
            if allowed_dtypes is not None:
                if dtype not in allowed_dtypes:
                    continue
            elif dtype in _NON_METRIC_DTYPES:
                continue

            ctx_id = t.get("context_id", 0)
            ctx = contexts.get(ctx_id, {})
            sv = SequenceView(t.get("name", ""), ctx, rv, trace_info=t)

            run_traces_checked += 1
            matched = q.check(run=rv, **{seq_type: sv})
            if matched:
                run_trace_matches += 1
                yield rv, sv, (counter, total), bundle
        dt_trace_loop = time.perf_counter() - t5

        total_traces_checked += run_traces_checked
        total_trace_matches += run_trace_matches
        if run_trace_matches:
            runs_matched += 1

        if timing:
            sum_fetch_bundle += dt_fetch_bundle
            sum_build += dt_build
            sum_trace_loop += dt_trace_loop
            _kw = {"path": path_label, "endpoint": _ep, "run_hash": h, "run_index": counter}
            _log_timing("fetch_bundle", dt_fetch_bundle, **_kw)  # ty:ignore[invalid-argument-type]
            _log_timing("build_run_view", dt_build, **_kw)  # ty:ignore[invalid-argument-type]
            _log_timing(
                "trace_loop",
                dt_trace_loop,
                path=path_label,
                endpoint=_ep,
                run_hash=h,
                run_index=counter,
                extra=f"traces_checked={run_traces_checked} matches={run_trace_matches}",
            )

    if timing:
        _log_timing(
            "summary",
            sum_fetch_bundle + sum_build + sum_trace_loop,
            path=path_label,
            endpoint=_ep,
            extra=(
                f"runs_checked={runs_checked} runs_matched={runs_matched} "
                f"traces_checked={total_traces_checked} trace_matches={total_trace_matches} "
                f"total_fetch_bundle_sec={sum_fetch_bundle:.6f} total_build_sec={sum_build:.6f} "
                f"total_trace_loop_sec={sum_trace_loop:.6f}"
            ),
        )
