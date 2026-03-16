"""Custom object sequence endpoints (images, texts, distributions, audios, figures).

Each object type registers search/get-batch/get-step endpoints, and types
with ``resolve_blobs=False`` also register a blob-batch endpoint.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
from typing import TYPE_CHECKING, Annotated, Any, override

from cryptography.fernet import InvalidToken
from fastapi import APIRouter, Header, HTTPException
from loguru import logger
from starlette.responses import StreamingResponse

from matyan_backend.api.streaming import (
    PROGRESS_REPORT_INTERVAL,
    collect_streamable_data,
    encode_tree,
    make_progress_key,
)
from matyan_backend.config import SETTINGS
from matyan_backend.deps import FdbDb  # noqa: TC001
from matyan_backend.storage.runs import get_all_contexts, get_run, get_run_attrs, get_run_traces_info
from matyan_backend.storage.s3_client import get_blob
from matyan_backend.storage.sequences import get_sequence_step_bounds, read_and_sample_sequence, read_sequence
from matyan_backend.thread_pool import FDB_EXECUTOR, to_fdb_thread

from ._blob_uri import decode_uri, generate_uri
from ._collections import iter_matching_sequences
from ._pydantic_models import RunTracesBatchApiIn  # noqa: TC001
from ._query import prepare_query
from ._range_utils import parse_range
from ._views import build_props_dict

if TYPE_CHECKING:
    import ast
    from collections.abc import AsyncGenerator, Callable

    from matyan_backend.fdb_types import Database

_SLEEP = 0.00001


def _downsample(items: list, density: int) -> list:
    """Return up to *density* evenly-spaced elements from *items*."""
    if density <= 0 or len(items) <= density:
        return items
    if density == 1:
        return [items[0]]
    step = (len(items) - 1) / (density - 1)
    indices = list(dict.fromkeys(round(i * step) for i in range(density)))
    return [items[idx] for idx in indices]


def _custom_search_producer(
    db: Database,
    q: str,
    tz_offset: int,
    seq: str,
    queue: asyncio.Queue[tuple | None],
    loop: asyncio.AbstractEventLoop,
    stop_flag: list[bool],
    prepared_ast: ast.Expression | None = None,
) -> None:
    """Run ``iter_matching_sequences`` in a background thread, pushing items into *queue*."""
    try:
        for rv, sv, progress in iter_matching_sequences(
            db,
            q,
            tz_offset=tz_offset,
            seq_type=seq,
            prepared_ast=prepared_ast,
        ):
            if stop_flag[0]:
                break
            asyncio.run_coroutine_threadsafe(queue.put((rv, sv, progress)), loop).result()
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()


def _build_custom_run_payload(  # noqa: C901
    db: Database,
    run_hash: str,
    meta: dict,
    sv_list: list[tuple[dict, int, str, dict]],
    rr_start: int | None,
    effective_end: int | None,
    record_density: int,
    ir_start: int | None,
    ir_stop: int | None,
    index_density: int,
    wrap: bool,
    skip_system: bool,
    dump: Callable[..., Any],
) -> dict:
    """Build the complete payload dict for a single run in custom object search (sync)."""
    run_params = get_run_attrs(db, run_hash) or {}
    if isinstance(run_params, dict):
        run_params.pop("__blobs__", None)
        if skip_system:
            run_params.pop("__system_params", None)

    meta_copy = meta.copy()
    meta_copy["hash"] = run_hash
    props = build_props_dict(meta_copy, db)

    traces: list[dict] = []
    full_rec_start: int | None = None
    full_rec_stop: int | None = None
    idx_stop = 0

    for _trace_info, ctx_id, name, context in sv_list:
        b_first, b_last = get_sequence_step_bounds(db, run_hash, ctx_id, name)
        if b_first is not None and b_last is not None:
            if full_rec_start is None or b_first < full_rec_start:
                full_rec_start = b_first
            if full_rec_stop is None or b_last > full_rec_stop:
                full_rec_stop = b_last

        sampled = read_and_sample_sequence(
            db,
            run_hash,
            ctx_id,
            name,
            start_step=rr_start,
            end_step=effective_end,
            density=record_density,
            columns=("val", "epoch", "time"),
        )
        steps = sampled.get("steps", [])
        values = sampled.get("val", [])

        if wrap and values:
            idx_stop = max(idx_stop, *(len(v) if isinstance(v, list) else 1 for v in values))

        processed = []
        for s, v in zip(steps, values, strict=True):
            if wrap:
                items = v if isinstance(v, list) else [v]
                lo = ir_start or 0
                hi = ir_stop if ir_stop is not None else len(items)
                items = items[lo:hi]
                if index_density and len(items) > index_density:
                    items = _downsample(items, index_density)
                processed.append([dump(it, run_hash, ctx_id, name, s, j) for j, it in enumerate(items)])
            else:
                processed.append(dump(v, run_hash, ctx_id, name, s, 0))

        traces.append(
            {
                "name": name,
                "context": context,
                "values": processed,
                "iters": steps,
                "epochs": sampled.get("epoch", []),
                "timestamps": sampled.get("time", []),
            },
        )

    rec_total = (
        full_rec_start if full_rec_start is not None else 0,
        full_rec_stop + 1 if full_rec_stop is not None else 0,
    )
    rec_used = (
        rr_start if rr_start is not None else rec_total[0],
        effective_end + 1 if effective_end is not None else rec_total[1],
    )
    ranges: dict[str, Any] = {
        "record_range_total": rec_total,
        "record_range_used": rec_used,
    }
    if wrap:
        idx_total = (0, max(idx_stop, 1))
        ranges["index_range_total"] = idx_total
        ranges["index_range_used"] = (
            ir_start if ir_start is not None else 0,
            ir_stop if ir_stop is not None else idx_total[1],
        )

    return {
        "params": run_params,
        "traces": traces,
        "props": props,
        "ranges": ranges,
    }


def _fetch_custom_batch(
    db: Database,
    run_id: str,
    requests: list,
    rr_start: int | None,
    rr_stop: int | None,
    record_density: int,
    ir_start: int | None,
    ir_stop: int | None,
    index_density: int,
    wrap: bool,
    dump: Callable[..., Any],
) -> list[dict] | None:
    """Synchronous helper — build trace_view list for custom get-batch."""  # noqa: D401
    run = get_run(db, run_id)
    if not run:
        return None

    traces_info = get_run_traces_info(db, run_id)
    contexts = get_all_contexts(db, run_id)
    effective_end = rr_stop - 1 if rr_stop is not None else None
    result: list[dict] = []

    for req in requests:
        ctx_id: int | None = None
        for t in traces_info:
            c = contexts.get(t.get("context_id", 0), {})
            if t.get("name") == req.name and c == req.context:
                ctx_id = t.get("context_id", 0)
                break
        if ctx_id is None:
            continue

        total_bounds = get_sequence_step_bounds(db, run_id, ctx_id, req.name)
        seq_data = read_and_sample_sequence(
            db,
            run_id,
            ctx_id,
            req.name,
            start_step=rr_start,
            end_step=effective_end,
            density=record_density,
        )
        steps = seq_data.get("steps", [])
        values = seq_data.get("val", [])

        processed = []
        idx_max = 0
        for i, (s, v) in enumerate(zip(steps, values, strict=True)):
            if wrap:
                items = v if isinstance(v, list) else [v]
                idx_max = max(idx_max, len(items))
                lo = ir_start or 0
                hi = ir_stop if ir_stop is not None else len(items)
                items = items[lo:hi]
                if index_density and len(items) > index_density:
                    items = _downsample(items, index_density)
                processed.append([dump(it, run_id, ctx_id, req.name, s, j) for j, it in enumerate(items)])
            else:
                processed.append(dump(v, run_id, ctx_id, req.name, s, i))

        tb0, tb1 = total_bounds
        range_total = (tb0, tb1 + 1) if tb0 is not None and tb1 is not None else (0, 0)
        range_used = (steps[0], steps[-1] + 1) if steps else (0, 0)

        idx_total_v: tuple[int, int] | None = None
        idx_used_v: tuple[int, int] | None = None
        if wrap:
            idx_max = max(idx_max, 1)
            idx_total_v = (0, idx_max)
            idx_used_v = (
                ir_start if ir_start is not None else 0,
                ir_stop if ir_stop is not None else idx_max,
            )

        result.append(
            {
                "name": req.name,
                "context": req.context,
                "values": processed,
                "iters": steps,
                "record_range_used": range_used,
                "record_range_total": range_total,
                "index_range_used": idx_used_v,
                "index_range_total": idx_total_v,
            },
        )

    return result


def _fetch_custom_step(  # noqa: C901, PLR0912
    db: Database,
    run_id: str,
    requests: list,
    record_step: int,
    ir_start: int | None,
    ir_stop: int | None,
    index_density: int,
    wrap: bool,
    dump: Callable[..., Any],
) -> list[dict] | None:
    """Synchronous helper — build trace_view list for custom get-step."""  # noqa: D401
    run = get_run(db, run_id)
    if not run:
        return None

    traces_info = get_run_traces_info(db, run_id)
    contexts = get_all_contexts(db, run_id)
    result: list[dict] = []

    for req in requests:
        ctx_id: int | None = None
        for t in traces_info:
            c = contexts.get(t.get("context_id", 0), {})
            if t.get("name") == req.name and c == req.context:
                ctx_id = t.get("context_id", 0)
                break
        if ctx_id is None:
            continue

        total_bounds = get_sequence_step_bounds(db, run_id, ctx_id, req.name)
        seq_data = read_sequence(db, run_id, ctx_id, req.name)
        steps = seq_data.get("steps", [])
        values = seq_data.get("val", [])

        if record_step == -1 and steps:
            val = values[-1]
            step = steps[-1]
        elif record_step >= 0:
            try:
                idx = steps.index(record_step)
            except ValueError:
                continue
            val = values[idx]
            step = steps[idx]
        else:
            continue

        step_values = [val]
        if isinstance(val, list):
            items = val
            if ir_start is not None or ir_stop is not None:
                lo = ir_start or 0
                hi = ir_stop if ir_stop is not None else len(items)
                items = items[lo:hi]
            if index_density and len(items) > index_density:
                items = _downsample(items, index_density)
            step_values = items

        dumped = [dump(v, run_id, ctx_id, req.name, step, idx) for idx, v in enumerate(step_values)]
        tb0, tb1 = total_bounds
        range_total = (tb0, tb1 + 1) if tb0 is not None and tb1 is not None else (0, 0)

        idx_total_v: tuple[int, int] | None = None
        idx_used_v: tuple[int, int] | None = None
        if wrap:
            n = len(step_values)
            idx_total_v = (0, n)
            idx_used_v = (
                ir_start if ir_start is not None else 0,
                ir_stop if ir_stop is not None else n,
            )

        result.append(
            {
                "name": req.name,
                "context": req.context,
                "values": dumped,
                "iters": [step],
                "record_range_used": (step, step),
                "record_range_total": range_total,
                "index_range_used": idx_used_v,
                "index_range_total": idx_total_v,
            },
        )

    return result


class CustomObjectApiConfig:
    """Base configuration for a custom object sequence type."""

    seq_name: str = ""
    resolve_blobs: bool = True
    use_list: bool = False

    @classmethod
    def _dump_value(cls, step_data: Any, run_hash: str, ctx_id: int, seq_name: str, step: int, index: int = 0) -> Any:  # noqa: ANN401, ARG003
        """Convert a raw stored value into the API response format."""
        return step_data

    @classmethod
    def register_endpoints(cls, router: APIRouter) -> None:  # noqa: C901, PLR0915
        seq = cls.seq_name
        resolve = cls.resolve_blobs
        dump = cls._dump_value
        wrap = cls.use_list

        @router.get(f"/search/{seq}/")
        async def search_custom_objects(  # noqa: C901
            db: FdbDb,
            q: str = "",
            skip_system: bool = True,
            report_progress: bool = True,
            record_range: str = "",
            record_density: int = 50,
            index_range: str = "",
            index_density: int = 5,
            x_timezone_offset: Annotated[int, Header()] = 0,
        ) -> StreamingResponse:
            try:
                p_ast = prepare_query(q, x_timezone_offset)
            except SyntaxError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid query syntax: {exc}") from exc

            rr = parse_range(record_range)
            ir = parse_range(index_range)

            async def _streamer() -> AsyncGenerator[bytes]:
                queue: asyncio.Queue[tuple | None] = asyncio.Queue(maxsize=SETTINGS.custom_search_queue_maxsize)
                stop_flag: list[bool] = [False]
                loop = asyncio.get_running_loop()

                producer_future = loop.run_in_executor(
                    FDB_EXECUTOR,
                    _custom_search_producer,
                    db,
                    q,
                    x_timezone_offset,
                    seq,
                    queue,
                    loop,
                    stop_flag,
                    p_ast,
                )

                progress_idx = 0
                last_progress = time.time()
                effective_end = rr.stop - 1 if rr.stop is not None else None

                current_hash: str | None = None
                current_meta: dict | None = None
                current_svs: list[tuple[dict, int, str, dict]] = []

                try:
                    while True:
                        item = await queue.get()
                        if item is None:
                            break

                        rv, sv, (checked, total) = item

                        if report_progress and time.time() - last_progress > PROGRESS_REPORT_INTERVAL:
                            yield collect_streamable_data(
                                encode_tree({make_progress_key(progress_idx): (checked, total)}),
                            )
                            progress_idx += 1
                            last_progress = time.time()

                        rh = rv.hash

                        if rh != current_hash:
                            if current_hash is not None and current_meta is not None:
                                payload = await to_fdb_thread(
                                    _build_custom_run_payload,
                                    db,
                                    current_hash,
                                    current_meta,
                                    current_svs,
                                    rr.start,
                                    effective_end,
                                    record_density,
                                    ir.start,
                                    ir.stop,
                                    index_density,
                                    wrap,
                                    skip_system,
                                    dump,
                                )
                                yield collect_streamable_data(encode_tree({current_hash: payload}))
                                if report_progress:
                                    yield collect_streamable_data(
                                        encode_tree({make_progress_key(progress_idx): (checked, total)}),
                                    )
                                    progress_idx += 1
                                    last_progress = time.time()

                            current_hash = rh
                            current_meta = rv._meta.copy()  # noqa: SLF001
                            current_svs = []

                        ctx_id = sv._trace_info.get("context_id", 0)  # noqa: SLF001
                        current_svs.append((sv._trace_info, ctx_id, sv.name, sv._context))  # noqa: SLF001

                    if current_hash is not None and current_meta is not None:
                        payload = await to_fdb_thread(
                            _build_custom_run_payload,
                            db,
                            current_hash,
                            current_meta,
                            current_svs,
                            rr.start,
                            effective_end,
                            record_density,
                            ir.start,
                            ir.stop,
                            index_density,
                            wrap,
                            skip_system,
                            dump,
                        )
                        yield collect_streamable_data(encode_tree({current_hash: payload}))

                    if report_progress:
                        yield collect_streamable_data(encode_tree({make_progress_key(progress_idx): (0, 0)}))
                finally:
                    stop_flag[0] = True
                    await producer_future

            return StreamingResponse(_streamer(), media_type="application/octet-stream")

        search_custom_objects.__name__ = f"search_{seq}"

        @router.post(f"/{{run_id}}/{seq}/get-batch/")
        async def get_custom_batch(
            run_id: str,
            body: RunTracesBatchApiIn,
            db: FdbDb,
            record_range: str = "",
            record_density: int = 50,
            index_range: str = "",
            index_density: int = 5,
        ) -> StreamingResponse:
            rr = parse_range(record_range)
            ir = parse_range(index_range)

            trace_views = await to_fdb_thread(
                _fetch_custom_batch,
                db,
                run_id,
                list(body),
                rr.start,
                rr.stop,
                record_density,
                ir.start,
                ir.stop,
                index_density,
                wrap,
                dump,
            )
            if trace_views is None:
                raise HTTPException(status_code=404)

            async def _streamer() -> AsyncGenerator[bytes]:
                for tv in trace_views:
                    yield collect_streamable_data(encode_tree(tv))

            return StreamingResponse(_streamer(), media_type="application/octet-stream")

        get_custom_batch.__name__ = f"get_{seq}_batch"

        @router.post(f"/{{run_id}}/{seq}/get-step/")
        async def get_custom_step(
            run_id: str,
            body: RunTracesBatchApiIn,
            db: FdbDb,
            record_step: int = -1,
            index_range: str = "",
            index_density: int = 5,
        ) -> StreamingResponse:
            ir = parse_range(index_range)

            trace_views = await to_fdb_thread(
                _fetch_custom_step,
                db,
                run_id,
                list(body),
                record_step,
                ir.start,
                ir.stop,
                index_density,
                wrap,
                dump,
            )
            if trace_views is None:
                raise HTTPException(status_code=404)

            async def _streamer() -> AsyncGenerator[bytes]:
                for tv in trace_views:
                    yield collect_streamable_data(encode_tree(tv))

            return StreamingResponse(_streamer(), media_type="application/octet-stream")

        get_custom_step.__name__ = f"get_{seq}_step"

        if not resolve:

            @router.post(f"/{seq}/get-batch/")
            async def get_blob_batch(
                body: list[str],
                db: FdbDb,
            ) -> StreamingResponse:
                async def _streamer() -> AsyncGenerator[bytes]:
                    for uri in body:
                        try:
                            run_hash, ctx_id, sn, step, idx = decode_uri(uri)
                        except (InvalidToken, ValueError, IndexError):
                            logger.debug("Invalid blob URI, skipping", uri=uri[:32])
                            continue
                        blob_data = await to_fdb_thread(
                            _fetch_blob_from_s3,
                            db,
                            run_hash,
                            ctx_id,
                            sn,
                            step,
                            idx,
                        )
                        yield collect_streamable_data(encode_tree({uri: blob_data}))

                return StreamingResponse(_streamer(), media_type="application/octet-stream")

            get_blob_batch.__name__ = f"get_{seq}_blob_batch"


def _fetch_blob_from_s3(
    db: Database,
    run_hash: str,
    ctx_id: int,
    seq_name: str,
    step: int,
    index: int = 0,
) -> bytes:
    """Read the metadata dict from FDB, extract the ``s3_key``, and fetch from S3.

    *index* selects the item within a list-valued step (used by ``use_list``
    types such as images and audios).
    """
    seq_data = read_sequence(db, run_hash, ctx_id, seq_name, start_step=step, end_step=step)
    values = seq_data.get("val", [])
    if not values:
        return b""
    meta = values[0]
    if isinstance(meta, list):
        if index < len(meta):
            meta = meta[index]
        else:
            return b""
    if not isinstance(meta, dict):
        return b""
    s3_key = meta.get("s3_key")
    if not s3_key:
        return b""
    return get_blob(s3_key)


# ---------------------------------------------------------------------------
# Concrete configs per type
# ---------------------------------------------------------------------------


class ImageApiConfig(CustomObjectApiConfig):
    seq_name = "images"
    resolve_blobs = False
    use_list = True

    @override
    @classmethod
    def _dump_value(cls, step_data: Any, run_hash: str, ctx_id: int, seq_name: str, step: int, index: int = 0) -> dict:
        if isinstance(step_data, dict):
            return {
                "caption": step_data.get("caption", ""),
                "width": step_data.get("width", 0),
                "height": step_data.get("height", 0),
                "format": step_data.get("format", "png"),
                "blob_uri": generate_uri(run_hash, ctx_id, seq_name, step, index),
                "index": index,
            }
        return {
            "caption": "",
            "width": 0,
            "height": 0,
            "format": "png",
            "blob_uri": generate_uri(run_hash, ctx_id, seq_name, step, index),
            "index": index,
        }


class TextApiConfig(CustomObjectApiConfig):
    seq_name = "texts"
    resolve_blobs = True
    use_list = True

    @override
    @classmethod
    def _dump_value(cls, step_data: Any, run_hash: str, ctx_id: int, seq_name: str, step: int, index: int = 0) -> dict:
        if isinstance(step_data, dict):
            return {"data": step_data.get("data", ""), "index": index}
        return {"data": str(step_data) if step_data else "", "index": index}


class DistributionApiConfig(CustomObjectApiConfig):
    seq_name = "distributions"
    resolve_blobs = True

    @override
    @classmethod
    def _dump_value(cls, step_data: Any, run_hash: str, ctx_id: int, seq_name: str, step: int, index: int = 0) -> dict:
        """Return ``{data: {type, shape, dtype, blob}, bin_count, range}``.

        The UI decodes ``data.blob`` as a raw float64 buffer via
        ``float64FromUint8`` (see ``streamEncoding.ts``), so the blob
        must be little-endian float64 bytes matching ``numpy.tobytes()``.
        """
        if isinstance(step_data, dict):
            raw_blob = step_data.get("data", b"")
            if isinstance(raw_blob, (bytes, bytearray, memoryview)):
                blob = bytes(raw_blob)
            elif isinstance(raw_blob, list):
                blob = struct.pack(f"<{len(raw_blob)}d", *raw_blob)
            else:
                blob = b""
            bin_count = step_data.get("bin_count", 0)
            return {
                "data": {
                    "type": "numpy",
                    "shape": bin_count,
                    "dtype": "float64",
                    "blob": blob,
                },
                "bin_count": bin_count,
                "range": step_data.get("range", [0, 0]),
            }
        return {
            "data": {"type": "numpy", "shape": 0, "dtype": "float64", "blob": b""},
            "bin_count": 0,
            "range": [0, 0],
        }


class AudioApiConfig(CustomObjectApiConfig):
    seq_name = "audios"
    resolve_blobs = False
    use_list = True

    @override
    @classmethod
    def _dump_value(cls, step_data: Any, run_hash: str, ctx_id: int, seq_name: str, step: int, index: int = 0) -> dict:
        if isinstance(step_data, dict):
            return {
                "caption": step_data.get("caption", ""),
                "format": step_data.get("format", "wav"),
                "blob_uri": generate_uri(run_hash, ctx_id, seq_name, step, index),
                "index": index,
            }
        return {
            "caption": "",
            "format": "wav",
            "blob_uri": generate_uri(run_hash, ctx_id, seq_name, step, index),
            "index": index,
        }


class FigureApiConfig(CustomObjectApiConfig):
    seq_name = "figures"
    resolve_blobs = True

    @override
    @classmethod
    def _dump_value(cls, step_data: Any, run_hash: str, ctx_id: int, seq_name: str, step: int, index: int = 0) -> dict:
        if isinstance(step_data, dict):
            raw = step_data.get("data", {})
            return {"data": json.dumps(raw) if isinstance(raw, (dict, list)) else raw, "index": index}
        return {"data": "{}", "index": index}


def register_all_custom_object_endpoints(router: APIRouter) -> None:
    """Register all custom object sequence endpoints on the given router."""
    ImageApiConfig.register_endpoints(router)
    TextApiConfig.register_endpoints(router)
    DistributionApiConfig.register_endpoints(router)
    AudioApiConfig.register_endpoints(router)
    FigureApiConfig.register_endpoints(router)
