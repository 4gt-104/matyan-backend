"""Run API endpoints — non-streaming CRUD operations.

Streaming search endpoints (run search, metric search, etc.) live in
``_streaming.py`` and are registered on the same ``rest_router_runs`` router.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query
from starlette.responses import Response, StreamingResponse
from stream_zip import _NO_COMPRESSION_64_TYPE, NO_COMPRESSION_64, stream_zip

from matyan_backend.api.streaming import collect_streamable_data, encode_tree
from matyan_backend.config import SETTINGS
from matyan_backend.deps import FdbDb, IngestionProducerDep, KafkaProducerDep  # noqa: TC001
from matyan_backend.kafka.producer import emit_control_event, emit_delete_run
from matyan_backend.storage import entities, indexes
from matyan_backend.storage.runs import (
    get_all_contexts,
    get_run,
    get_run_artifacts,
    get_run_attrs,
    get_run_meta,
    get_run_traces_info,
    mark_pending_deletion,
    set_run_experiment,
    update_run_meta,
)
from matyan_backend.storage.s3_client import get_blob, stream_blob
from matyan_backend.storage.sequences import get_sequence_length, get_sequence_step_bounds, read_sequence

from ._range_utils import parse_range

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Iterator

    from matyan_backend.fdb_types import Database

from ._pydantic_models import (
    NoteIn,
    RunInfoOut,
    RunsBatchIn,
    StructuredRunAddTagIn,
    StructuredRunAddTagOut,
    StructuredRunRemoveTagOut,
    StructuredRunsArchivedOut,
    StructuredRunUpdateIn,
    StructuredRunUpdateOut,
)

rest_router_runs = APIRouter(prefix="/runs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_end_time(meta: dict) -> float | None:
    duration = meta.get("duration")
    if duration is not None:
        return meta.get("created_at", 0) + duration
    return meta.get("finalized_at")


def _build_props(meta: dict, db: Database) -> dict:
    """Assemble a ``PropsView``-compatible dict from run meta."""
    tags_raw = entities.get_tags_for_run(db, meta.get("hash", ""))
    tags = [
        {"id": t["id"], "name": t.get("name", ""), "color": t.get("color"), "description": t.get("description")}
        for t in tags_raw
    ]
    exp_dict: dict = {"id": None, "name": None, "description": None}
    exp_id = meta.get("experiment_id")
    if exp_id:
        exp = entities.get_experiment(db, exp_id)
        if exp:
            exp_dict = {"id": exp["id"], "name": exp.get("name", ""), "description": exp.get("description", "")}

    return {
        "name": meta.get("name"),
        "description": meta.get("description"),
        "experiment": exp_dict,
        "tags": tags,
        "creation_time": meta.get("created_at", 0),
        "end_time": _compute_end_time(meta),
        "archived": meta.get("is_archived", False),
        "active": meta.get("active", False),
    }


_DTYPE_TO_SEQ_TYPE: dict[str, str] = {
    "image": "images",
    "text": "texts",
    "distribution": "distributions",
    "audio": "audios",
    "figure": "figures",
}

_HIDDEN_DTYPES = frozenset({"logs", "log_records"})


def _build_traces_overview(
    traces_info: list[dict],
    contexts: dict,
    sequence_filter: tuple[str, ...] = (),
) -> dict[str, list]:
    """Group trace info by sequence type, producing ``{"metric": [TraceOverview]}``.

    When *sequence_filter* is non-empty, only the requested sequence types are
    included.  An empty tuple means "return all types".
    """
    buckets: dict[str, list] = {}
    for bucket_name in ("metric", "images", "texts", "distributions", "audios", "figures"):
        if not sequence_filter or bucket_name in sequence_filter:
            buckets[bucket_name] = []

    for t in traces_info:
        dtype = t.get("dtype", "")
        if dtype in _HIDDEN_DTYPES:
            continue
        bucket = _DTYPE_TO_SEQ_TYPE.get(dtype, "metric")
        if bucket not in buckets:
            continue
        ctx_id = t.get("context_id", 0)
        ctx = contexts.get(ctx_id, {})
        entry: dict = {
            "name": t.get("name", ""),
            "context": ctx,
        }
        if bucket == "metric":
            entry["last_value"] = t.get("last", 0.0)
        buckets[bucket].append(entry)

    return buckets


# ---------------------------------------------------------------------------
# Run Info
# ---------------------------------------------------------------------------


def _fetch_run_info(
    db: Database,
    run_id: str,
    skip_system: bool,
    sequence: tuple[str, ...],
) -> dict | None:
    """Synchronous helper — all FDB reads for run info."""  # noqa: D401
    run = get_run(db, run_id)
    if not run:
        return None

    meta = get_run_meta(db, run_id)
    meta["hash"] = run_id
    params = get_run_attrs(db, run_id) or {}
    if isinstance(params, dict):
        params.pop("__blobs__", None)
        if skip_system:
            params.pop("__system_params", None)
    traces_info = get_run_traces_info(db, run_id)
    contexts = get_all_contexts(db, run_id)

    raw_artifacts = get_run_artifacts(db, run_id)
    artifacts = [
        {
            "name": a["name"],
            "path": a["path"],
            "uri": f"s3://{SETTINGS.s3_bucket}/{a['s3_key']}",
        }
        for a in raw_artifacts
    ]

    return {
        "params": params,
        "traces": _build_traces_overview(traces_info, contexts, sequence_filter=sequence),
        "props": _build_props(meta, db),
        "artifacts": artifacts,
    }


@rest_router_runs.get("/{run_id}/info/", response_model=RunInfoOut)
async def get_run_info_api(
    run_id: str,
    db: FdbDb,
    skip_system: bool = False,
    sequence: Annotated[tuple[str, ...], Query()] = (),
) -> dict:
    """Return run info: params, traces overview, props, and artifacts list.

    :param run_id: Run hash.
    :param skip_system: If True, omit ``__system_params`` from params.
    :param sequence: Limit traces to these sequence types; empty means all.
    :returns: Dict with params, traces, props, artifacts.
    :raises HTTPException: 404 if run not found.
    """
    result = await asyncio.to_thread(_fetch_run_info, db, run_id, skip_system, sequence)
    if result is None:
        raise HTTPException(status_code=404)
    return result


def _fetch_artifact_download(
    db: Database,
    run_id: str,
    path: str,
) -> tuple[bytes, str, str] | None:
    """Synchronous helper — locate artifact and fetch from S3."""  # noqa: D401
    run = get_run(db, run_id)
    if not run:
        return None

    raw_artifacts = get_run_artifacts(db, run_id)
    artifact = next((a for a in raw_artifacts if a["path"] == path), None)
    if not artifact:
        return None

    content = get_blob(artifact["s3_key"])
    return content, artifact.get("content_type", "application/octet-stream"), artifact["name"]


@rest_router_runs.get("/{run_id}/artifacts/download/")
async def download_artifact_api(run_id: str, db: FdbDb, path: Annotated[str, Query()]) -> Response:
    """Download a single artifact file by path (S3 blob streamed to client).

    :param run_id: Run hash.
    :param path: Artifact path/name within the run.
    :returns: Raw response with file content and Content-Disposition.
    :raises HTTPException: 404 if run or artifact not found.
    """
    result = await asyncio.to_thread(_fetch_artifact_download, db, run_id, path)
    if result is None:
        raise HTTPException(status_code=404)
    content, media_type, filename = result
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _fetch_artifacts_list(db: Database, run_id: str) -> list[dict] | None:
    """Synchronous helper — get run + artifacts list from FDB."""  # noqa: D401
    run = get_run(db, run_id)
    if not run:
        return None
    artifacts = get_run_artifacts(db, run_id)
    if not artifacts:
        return None
    return artifacts


@rest_router_runs.get("/{run_id}/artifacts/download-all/")
async def download_all_artifacts_api(run_id: str, db: FdbDb) -> StreamingResponse:
    """Stream all run artifacts as a ZIP archive.

    :param run_id: Run hash.
    :returns: StreamingResponse (application/zip).
    :raises HTTPException: 404 if run not found or has no artifacts.
    """
    artifacts = await asyncio.to_thread(_fetch_artifacts_list, db, run_id)
    if artifacts is None:
        raise HTTPException(status_code=404)

    now = datetime.now(tz=UTC)

    def _member_files() -> Generator[tuple[Any, datetime, Literal[384], _NO_COMPRESSION_64_TYPE, Iterator[bytes]]]:
        for a in artifacts:
            yield a["path"], now, 0o600, NO_COMPRESSION_64, stream_blob(a["s3_key"])

    return StreamingResponse(
        stream_zip(_member_files()),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_id}-artifacts.zip"'},
    )


# ---------------------------------------------------------------------------
# Run Update / Delete
# ---------------------------------------------------------------------------


def _collect_run_updates(body: StructuredRunUpdateIn, db: FdbDb, run_id: str) -> dict:

    updates: dict = {}
    if body.name is not None:
        updates["name"] = body.name
    if body.description is not None:
        updates["description"] = body.description
    if body.archived is not None:
        updates["is_archived"] = body.archived
    if body.active is not None:
        updates["active"] = body.active
        if not body.active:
            meta = get_run_meta(db, run_id)
            if not meta.get("finalized_at"):
                updates["finalized_at"] = time.time()
        else:
            updates["finalized_at"] = None
    return updates


def _apply_run_update(db: Database, run_id: str, body: StructuredRunUpdateIn) -> bool:
    """Synchronous helper — apply run update to FDB. Returns False if run not found."""  # noqa: D401
    run = get_run(db, run_id)
    if not run:
        return False

    updates = _collect_run_updates(body, db, run_id)
    if updates:
        update_run_meta(db, run_id, **updates)

    if body.experiment is not None:
        exp = entities.get_experiment_by_name(db, body.experiment)
        exp_id = exp["id"] if exp else None
        set_run_experiment(db, run_id, exp_id)
        if exp_id:
            entities.set_run_experiment(db, run_id, exp_id)
    return True


@rest_router_runs.put("/{run_id}/", response_model=StructuredRunUpdateOut)
async def update_run_api(
    run_id: str,
    body: StructuredRunUpdateIn,
    db: FdbDb,
    producer: KafkaProducerDep,
) -> dict[str, str]:
    """Update run properties (name, description, archived, active, experiment).

    :param run_id: Run hash.
    :param body: Fields to update (only provided fields are changed).
    :returns: ``{"id": run_id, "status": "OK"}``.
    :raises HTTPException: 404 if run not found.
    """
    found = await asyncio.to_thread(_apply_run_update, db, run_id, body)
    if not found:
        raise HTTPException(status_code=404)

    if body.archived is not None:
        event_type = "run_archived" if body.archived else "run_unarchived"
        await emit_control_event(producer, event_type, run_id=run_id, archived=body.archived)

    return {"id": run_id, "status": "OK"}


def _mark_run_pending_deletion(db: Database, run_id: str) -> bool:
    """Synchronous helper — check run exists and mark for deletion."""  # noqa: D401
    run = get_run(db, run_id)
    if not run:
        return False
    mark_pending_deletion(db, run_id)
    return True


@rest_router_runs.delete("/{run_id}/")
async def delete_run_api(
    run_id: str,
    db: FdbDb,
    ingestion_producer: IngestionProducerDep,
) -> dict[str, str]:
    """Mark run as deleted in FDB and emit control event for async cleanup (e.g. S3).

    :param run_id: Run hash.
    :returns: ``{"id": run_id, "status": "OK"}``.
    :raises HTTPException: 404 if run not found.
    """
    found = await asyncio.to_thread(_mark_run_pending_deletion, db, run_id)
    if not found:
        raise HTTPException(status_code=404)
    await emit_delete_run(ingestion_producer, run_id)
    return {"id": run_id, "status": "OK"}


def _mark_runs_pending_deletion(db: Database, run_ids: list[str]) -> None:
    """Synchronous helper — mark multiple runs for deletion."""  # noqa: D401
    for run_id in run_ids:
        mark_pending_deletion(db, run_id)


@rest_router_runs.post("/delete-batch/")
async def delete_runs_batch_api(
    body: RunsBatchIn,
    db: FdbDb,
    ingestion_producer: IngestionProducerDep,
) -> dict[str, str]:
    """Mark multiple runs as deleted and emit control events for each.

    :param body: List of run hashes to delete.
    :returns: ``{"status": "OK"}``.
    """
    await asyncio.to_thread(_mark_runs_pending_deletion, db, list(body))
    for run_id in body:
        await emit_delete_run(ingestion_producer, run_id)
    return {"status": "OK"}


@rest_router_runs.post("/{run_id}/clear-tombstone/", status_code=204, response_model=None)
async def clear_run_tombstone_api(run_id: str, db: FdbDb) -> None:
    """Clear the tombstone for a run (e.g. after undo delete).

    :param run_id: Run hash.
    """
    await asyncio.to_thread(indexes.clear_run_tombstone, db, run_id)


def _finish_runs_batch(db: Database, run_ids: list[str]) -> None:
    """Synchronous helper — finalize multiple runs."""  # noqa: D401
    now = time.time()
    for run_id in run_ids:
        meta = get_run_meta(db, run_id)
        updates: dict = {"active": False}
        if not meta.get("finalized_at"):
            updates["finalized_at"] = now
        update_run_meta(db, run_id, **updates)


@rest_router_runs.post("/finish-batch/")
async def finish_runs_batch_api(body: RunsBatchIn, db: FdbDb) -> dict[str, str]:
    """Mark multiple runs as finalized (active=False, set finalized_at).

    :param body: List of run hashes to finish.
    :returns: ``{"status": "OK"}``.
    """
    await asyncio.to_thread(_finish_runs_batch, db, list(body))
    return {"status": "OK"}


def _archive_runs_batch(db: Database, run_ids: list[str], archive: bool) -> None:
    """Synchronous helper — archive/unarchive multiple runs."""  # noqa: D401
    for run_id in run_ids:
        update_run_meta(db, run_id, is_archived=archive)


@rest_router_runs.post("/archive-batch/", response_model=StructuredRunsArchivedOut)
async def archive_runs_batch_api(
    body: RunsBatchIn,
    db: FdbDb,
    producer: KafkaProducerDep,
    archive: bool = True,
) -> dict[str, str]:
    """Archive or unarchive multiple runs; emit control events for each.

    :param body: List of run hashes.
    :param archive: True to archive, False to unarchive.
    :returns: ``{"status": "OK"}``.
    """
    await asyncio.to_thread(_archive_runs_batch, db, list(body), archive)
    event_type = "run_archived" if archive else "run_unarchived"
    for run_id in body:
        await emit_control_event(producer, event_type, run_id=run_id, archived=archive)
    return {"status": "OK"}


# ---------------------------------------------------------------------------
# Run ↔ Tags
# ---------------------------------------------------------------------------


def _add_tag_to_run(db: Database, run_id: str, tag_name: str) -> tuple[str, str] | None:
    """Synchronous helper — find/create tag and associate with run."""  # noqa: D401
    run = get_run(db, run_id)
    if not run:
        return None
    tag = entities.get_tag_by_name(db, tag_name)
    if not tag:
        tag = entities.create_tag(db, tag_name)
    entities.add_tag_to_run(db, run_id, tag["id"])
    return run_id, tag["id"]


@rest_router_runs.post("/{run_id}/tags/new/", response_model=StructuredRunAddTagOut)
async def add_tag_to_run_api(run_id: str, body: StructuredRunAddTagIn, db: FdbDb) -> dict[str, str]:
    """Add a tag to a run (create tag if it does not exist).

    :param run_id: Run hash.
    :param body: Must include ``tag_name``.
    :returns: ``{"id": run_id, "tag_id": tag_id, "status": "OK"}``.
    :raises HTTPException: 404 if run not found.
    """
    result = await asyncio.to_thread(_add_tag_to_run, db, run_id, body.tag_name.strip())
    if result is None:
        raise HTTPException(status_code=404)
    rid, tag_id = result
    return {"id": rid, "tag_id": tag_id, "status": "OK"}


def _remove_tag_from_run(db: Database, run_id: str, tag_id: str) -> bool:
    """Synchronous helper — remove tag association from run."""  # noqa: D401
    run = get_run(db, run_id)
    if not run:
        return False
    entities.remove_tag_from_run(db, run_id, tag_id)
    return True


@rest_router_runs.delete("/{run_id}/tags/{tag_id}/", response_model=StructuredRunRemoveTagOut)
async def remove_tag_from_run_api(run_id: str, tag_id: str, db: FdbDb) -> dict[str, Any]:
    """Remove a tag association from a run.

    :param run_id: Run hash.
    :param tag_id: Tag ID to remove.
    :returns: ``{"id": run_id, "removed": True, "status": "OK"}``.
    :raises HTTPException: 404 if run not found.
    """
    found = await asyncio.to_thread(_remove_tag_from_run, db, run_id, tag_id)
    if not found:
        raise HTTPException(status_code=404)
    return {"id": run_id, "removed": True, "status": "OK"}


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


@rest_router_runs.get("/{run_id}/note/")
async def list_run_notes_api(run_id: str, db: FdbDb) -> list[dict]:
    """List all notes for a run.

    :param run_id: Run hash.
    :returns: List of note dicts (id, content, created_at, updated_at, etc.).
    """
    return await asyncio.to_thread(entities.list_notes_for_run, db, run_id)


@rest_router_runs.post("/{run_id}/note/", status_code=201)
async def create_run_note_api(run_id: str, note_in: NoteIn, db: FdbDb) -> dict[str, Any]:
    """Create a note attached to a run.

    :param run_id: Run hash.
    :param note_in: Body with ``content``.
    :returns: ``{"id": note_id, "created_at": ...}``.
    """
    note = await asyncio.to_thread(entities.create_note, db, note_in.content.strip(), run_hash=run_id)
    return {"id": note["id"], "created_at": note["created_at"]}


def _get_run_note(db: Database, run_id: str, _id: str) -> dict | None:
    """Synchronous helper — get note and verify ownership."""  # noqa: D401
    note = entities.get_note(db, _id)
    if not note or note.get("run_hash") != run_id:
        return None
    return note


@rest_router_runs.get("/{run_id}/note/{_id}/")
async def get_run_note_api(run_id: str, _id: str, db: FdbDb) -> dict[str, Any]:
    """Get a single note by ID (must belong to the run).

    :param run_id: Run hash.
    :param _id: Note ID.
    :returns: ``{"id", "content", "updated_at"}``.
    :raises HTTPException: 404 if note not found or not owned by run.
    """
    note = await asyncio.to_thread(_get_run_note, db, run_id, _id)
    if not note:
        raise HTTPException(status_code=404)
    return {"id": note["id"], "content": note.get("content", ""), "updated_at": note.get("updated_at")}


def _update_run_note(db: Database, run_id: str, _id: str, content: str) -> dict | None:
    """Synchronous helper — validate, update, and return the note."""  # noqa: D401
    note = entities.get_note(db, _id)
    if not note or note.get("run_hash") != run_id:
        return None
    entities.update_note(db, _id, content=content)
    return entities.get_note(db, _id)


@rest_router_runs.put("/{run_id}/note/{_id}/")
async def update_run_note_api(run_id: str, _id: str, note_in: NoteIn, db: FdbDb) -> dict[str, Any]:
    """Update a note's content (note must belong to the run).

    :param run_id: Run hash.
    :param _id: Note ID.
    :param note_in: Body with ``content``.
    :returns: ``{"id", "content", "updated_at"}``.
    :raises HTTPException: 404 if note not found or not owned by run.
    """
    updated = await asyncio.to_thread(_update_run_note, db, run_id, _id, note_in.content.strip())
    if not updated:
        raise HTTPException(status_code=404)
    return {"id": _id, "content": updated.get("content", ""), "updated_at": updated.get("updated_at")}


def _delete_run_note(db: Database, run_id: str, _id: str) -> bool:
    """Synchronous helper — validate ownership and delete note."""  # noqa: D401
    note = entities.get_note(db, _id)
    if not note or note.get("run_hash") != run_id:
        return False
    entities.delete_note(db, _id)
    return True


@rest_router_runs.delete("/{run_id}/note/{_id}/")
async def delete_run_note_api(run_id: str, _id: str, db: FdbDb) -> dict[str, str]:
    """Delete a note (must belong to the run).

    :param run_id: Run hash.
    :param _id: Note ID.
    :returns: ``{"status": "OK"}``.
    :raises HTTPException: 404 if note not found or not owned by run.
    """
    found = await asyncio.to_thread(_delete_run_note, db, run_id, _id)
    if not found:
        raise HTTPException(status_code=404)
    return {"status": "OK"}


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

_DEFAULT_LOG_TAIL = 200
_LOG_SEQ_NAME = "logs"
_LOG_RECORDS_SEQ_NAME = "__log_records"
_LOG_LEVEL_NAMES: dict[int, str] = {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}


def _fetch_run_logs(
    db: Database,
    run_id: str,
    rr_start: int | None,
    rr_stop: int | None,
) -> list[tuple[int, Any]]:
    """Synchronous helper — read log sequence entries from FDB."""  # noqa: D401
    bounds = get_sequence_step_bounds(db, run_id, 0, _LOG_SEQ_NAME)
    first_step, last_step = bounds
    if first_step is None or last_step is None:
        return []

    start = rr_start
    stop = rr_stop
    if start is None and stop is None:
        start = max(last_step - _DEFAULT_LOG_TAIL + 1, first_step)
        stop = last_step + 1
    if start is None:
        start = first_step
    if stop is None:
        stop = last_step + 1

    data = read_sequence(db, run_id, 0, _LOG_SEQ_NAME, start_step=start, end_step=stop - 1)
    return list(zip(data.get("steps", []), data.get("val", []), strict=True))


@rest_router_runs.get("/{run_id}/logs/")
async def get_run_logs_api(run_id: str, db: FdbDb, record_range: str = "") -> StreamingResponse:
    """Stream terminal log lines for a run (binary Aim codec).

    :param run_id: Run hash.
    :param record_range: Optional ``start:stop`` range (default: last 200 lines).
    :returns: StreamingResponse (application/octet-stream).
    """
    rr = parse_range(record_range)

    async def _streamer() -> AsyncGenerator[bytes]:
        pairs = await asyncio.to_thread(_fetch_run_logs, db, run_id, rr.start, rr.stop)
        for step, val in pairs:
            yield collect_streamable_data(encode_tree({step: val}))

    return StreamingResponse(_streamer(), media_type="application/octet-stream")


def _fetch_run_log_records(
    db: Database,
    run_id: str,
    rr_start: int | None,
    rr_stop: int | None,
) -> tuple[int, list[tuple[int, Any]]]:
    """Synchronous helper — read log record entries from FDB."""  # noqa: D401
    bounds = get_sequence_step_bounds(db, run_id, 0, _LOG_RECORDS_SEQ_NAME)
    first_step, last_step = bounds
    total_count = get_sequence_length(db, run_id, 0, _LOG_RECORDS_SEQ_NAME)

    if first_step is None or last_step is None:
        return total_count, []

    start = rr_start
    stop = rr_stop
    if start is None and stop is None:
        start = max(last_step - _DEFAULT_LOG_TAIL + 1, first_step)
        stop = last_step + 1
    if start is None:
        start = first_step
    if stop is None:
        stop = last_step + 1

    data = read_sequence(db, run_id, 0, _LOG_RECORDS_SEQ_NAME, start_step=start, end_step=stop - 1)
    return total_count, list(zip(data.get("steps", []), data.get("val", []), strict=True))


@rest_router_runs.get("/{run_id}/log-records/")
async def get_run_log_records_api(run_id: str, db: FdbDb, record_range: str = "") -> StreamingResponse:
    """Stream structured log records for a run (binary Aim codec).

    :param run_id: Run hash.
    :param record_range: Optional ``start:stop`` range (default: last 200).
    :returns: StreamingResponse (application/octet-stream); total count in header.
    """
    rr = parse_range(record_range)

    async def _streamer() -> AsyncGenerator[bytes]:
        total_count, pairs = await asyncio.to_thread(
            _fetch_run_log_records,
            db,
            run_id,
            rr.start,
            rr.stop,
        )
        yield collect_streamable_data(encode_tree({"log_records_count": total_count}))

        for step, val in pairs:
            record: dict = val if isinstance(val, dict) else {"message": str(val)}
            raw_level = record.get("log_level", 20)
            if isinstance(raw_level, int):
                record["log_level"] = _LOG_LEVEL_NAMES.get(raw_level, "INFO")
            record.setdefault("timestamp", 0)
            record.setdefault("args", None)
            record["is_notified"] = True
            yield collect_streamable_data(encode_tree({step: record}))

    return StreamingResponse(_streamer(), media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Register streaming + custom-object endpoints on this router
# ---------------------------------------------------------------------------

from . import _streaming as _streaming  # noqa: E402, PLC0414
from ._custom_objects import register_all_custom_object_endpoints  # noqa: E402

register_all_custom_object_endpoints(rest_router_runs)
