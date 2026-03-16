"""Experiment REST endpoints: CRUD, runs listing, notes, and activity.

All routes are under the ``/experiments`` prefix on :data:`rest_router_experiments`.
"""

from __future__ import annotations

import asyncio
import datetime
from collections import Counter
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from matyan_backend.deps import FdbDb, KafkaProducerDep  # noqa: TC001
from matyan_backend.kafka.producer import emit_control_event
from matyan_backend.storage import entities
from matyan_backend.storage.runs import get_run_meta

from ._pydantic_models import (
    ExperimentActivityApiOut,
    ExperimentCreateRequest,
    ExperimentGetOut,
    ExperimentGetRunsResponse,
    ExperimentListOut,
    ExperimentUpdateOut,
    ExperimentUpdateRequest,
)

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database

rest_router_experiments = APIRouter(prefix="/experiments")


class NoteIn(BaseModel):
    content: str


def _exp_to_out(exp: dict) -> dict:
    return {
        "id": exp["id"],
        "name": exp.get("name", ""),
        "description": exp.get("description", ""),
        "run_count": exp.get("run_count", 0),
        "archived": exp.get("is_archived", False),
        "creation_time": exp.get("created_at"),
    }


def _list_experiments(db: Database) -> list[dict]:
    return [_exp_to_out(exp) for exp in entities.list_experiments(db)]


@rest_router_experiments.get("/", response_model=ExperimentListOut)
async def get_experiments_list_api(db: FdbDb) -> list[dict]:
    """List all experiments."""
    return await asyncio.to_thread(_list_experiments, db)


def _search_experiments(db: Database, q: str | None) -> list[dict]:
    exps = entities.list_experiments(db)
    search_term = q.strip().lower() if q else ""
    if search_term:
        exps = [e for e in exps if search_term in e.get("name", "").lower()]
    return [_exp_to_out(exp) for exp in exps]


@rest_router_experiments.get("/search/", response_model=ExperimentListOut)
async def search_experiments_by_name_api(
    db: FdbDb,
    q: Annotated[str | None, Query()] = None,
) -> list[dict]:
    """Search experiments by name (case-insensitive substring)."""
    return await asyncio.to_thread(_search_experiments, db, q)


@rest_router_experiments.post("/", response_model=ExperimentUpdateOut)
async def create_experiment_api(request: ExperimentCreateRequest, db: FdbDb) -> dict[str, Any]:
    """Create a new experiment by name.

    :raises HTTPException: 400 if name is duplicate or invalid.
    """
    try:
        exp = await asyncio.to_thread(entities.create_experiment, db, name=request.name.strip())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"id": exp["id"], "status": "OK"}


@rest_router_experiments.get("/{uuid}/", response_model=ExperimentGetOut)
async def get_experiment_api(uuid: str, db: FdbDb) -> dict:
    """Get a single experiment by UUID.

    :raises HTTPException: 404 if not found.
    """
    exp = await asyncio.to_thread(entities.get_experiment, db, uuid)
    if not exp:
        raise HTTPException(status_code=404)
    return _exp_to_out(exp)


def _delete_experiment(db: Database, uuid: str) -> bool:
    exp = entities.get_experiment(db, uuid)
    if not exp:
        return False
    entities.delete_experiment(db, uuid)
    return True


@rest_router_experiments.delete("/{uuid}/")
async def delete_experiment_api(uuid: str, db: FdbDb, producer: KafkaProducerDep) -> dict[str, str]:
    """Delete an experiment and emit control event for async side effects.

    :raises HTTPException: 404 if not found.
    """
    found = await asyncio.to_thread(_delete_experiment, db, uuid)
    if not found:
        raise HTTPException(status_code=404)
    await emit_control_event(producer, "experiment_deleted", experiment_id=uuid)
    return {"status": "OK"}


def _update_experiment(db: Database, uuid: str, exp_in: ExperimentUpdateRequest) -> bool:
    exp = entities.get_experiment(db, uuid)
    if not exp:
        return False
    updates: dict = {}
    if exp_in.name:
        updates["name"] = exp_in.name.strip()
    if exp_in.description is not None:
        updates["description"] = exp_in.description
    if exp_in.archived is not None:
        updates["is_archived"] = exp_in.archived
    if updates:
        entities.update_experiment(db, uuid, **updates)
    return True


@rest_router_experiments.put("/{uuid}/", response_model=ExperimentUpdateOut)
async def update_experiment_properties_api(
    uuid: str,
    exp_in: ExperimentUpdateRequest,
    db: FdbDb,
) -> dict[str, str]:
    """Update experiment name, description, or archived state.

    :raises HTTPException: 404 if experiment not found.
    """
    found = await asyncio.to_thread(_update_experiment, db, uuid, exp_in)
    if not found:
        raise HTTPException(status_code=404)
    return {"id": uuid, "status": "OK"}


def _fetch_experiment_runs(
    db: Database,
    uuid: str,
    limit: int | None,
    offset: str | None,
) -> dict[str, Any] | None:
    """Synchronous helper — get experiment runs from FDB."""  # noqa: D401
    exp = entities.get_experiment(db, uuid)
    if not exp:
        return None

    run_hashes = entities.get_runs_for_experiment(db, uuid)
    offset_idx = 0
    if offset and offset in run_hashes:
        offset_idx = run_hashes.index(offset) + 1
    run_hashes = run_hashes[offset_idx : offset_idx + limit] if limit else run_hashes[offset_idx:]

    runs = []
    for rh in run_hashes:
        meta = get_run_meta(db, rh)
        if not meta or meta.get("pending_deletion"):
            continue
        runs.append(
            {
                "run_id": rh,
                "name": meta.get("name", ""),
                "creation_time": meta.get("created_at", 0),
                "end_time": meta.get("finalized_at"),
                "archived": meta.get("is_archived", False),
            },
        )
    return {"id": uuid, "runs": runs}


@rest_router_experiments.get("/{uuid}/runs/", response_model=ExperimentGetRunsResponse)
async def get_experiment_runs_api(
    uuid: str,
    db: FdbDb,
    limit: Annotated[int | None, Query()] = None,
    offset: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    """List runs belonging to the experiment (with optional limit/offset).

    :raises HTTPException: 404 if experiment not found.
    """
    data = await asyncio.to_thread(_fetch_experiment_runs, db, uuid, limit, offset)
    if data is None:
        raise HTTPException(status_code=404)
    return data


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def _list_experiment_notes(db: Database, exp_id: str) -> list[dict] | None:
    exp = entities.get_experiment(db, exp_id)
    if not exp:
        return None
    return entities.list_notes_for_experiment(db, exp_id)


@rest_router_experiments.get("/{exp_id}/note/")
async def list_note_api(exp_id: str, db: FdbDb) -> list[dict]:
    """List all notes for an experiment.

    :raises HTTPException: 404 if experiment not found.
    """
    result = await asyncio.to_thread(_list_experiment_notes, db, exp_id)
    if result is None:
        raise HTTPException(status_code=404)
    return result


def _create_experiment_note(db: Database, exp_id: str, content: str) -> dict | None:
    exp = entities.get_experiment(db, exp_id)
    if not exp:
        return None
    return entities.create_note(db, content, experiment_id=exp_id)


@rest_router_experiments.post("/{exp_id}/note/", status_code=201)
async def create_note_api(exp_id: str, note_in: NoteIn, db: FdbDb) -> dict[str, Any]:
    """Create a note attached to an experiment.

    :raises HTTPException: 404 if experiment not found.
    """
    note = await asyncio.to_thread(_create_experiment_note, db, exp_id, note_in.content.strip())
    if note is None:
        raise HTTPException(status_code=404)
    return {"id": note["id"], "created_at": note["created_at"]}


@rest_router_experiments.get("/{exp_id}/note/{_id}/")
async def get_note_api(exp_id: str, _id: str, db: FdbDb) -> dict[str, Any]:  # noqa: ARG001
    """Get a single note by ID.

    :raises HTTPException: 404 if note not found.
    """
    note = await asyncio.to_thread(entities.get_note, db, _id)
    if not note:
        raise HTTPException(status_code=404)
    return {"id": note["id"], "content": note.get("content", ""), "updated_at": note.get("updated_at")}


def _update_experiment_note(db: Database, _id: str, content: str) -> dict | None:
    note = entities.get_note(db, _id)
    if not note:
        return None
    entities.update_note(db, _id, content=content)
    return entities.get_note(db, _id)


@rest_router_experiments.put("/{exp_id}/note/{_id}/")
async def update_note_api(exp_id: str, _id: str, note_in: NoteIn, db: FdbDb) -> dict[str, Any]:  # noqa: ARG001
    """Update a note's content.

    :raises HTTPException: 404 if note not found.
    """
    updated = await asyncio.to_thread(_update_experiment_note, db, _id, note_in.content.strip())
    if not updated:
        raise HTTPException(status_code=404)
    return {"id": _id, "content": updated.get("content", ""), "updated_at": updated.get("updated_at")}


def _delete_experiment_note(db: Database, _id: str) -> bool:
    note = entities.get_note(db, _id)
    if not note:
        return False
    entities.delete_note(db, _id)
    return True


@rest_router_experiments.delete("/{exp_id}/note/{_id}/")
async def delete_note_api(exp_id: str, _id: str, db: FdbDb) -> dict[str, str]:  # noqa: ARG001
    """Delete a note.

    :raises HTTPException: 404 if note not found.
    """
    found = await asyncio.to_thread(_delete_experiment_note, db, _id)
    if not found:
        raise HTTPException(status_code=404)
    return {"status": "OK"}


def _fetch_experiment_activity(
    db: Database,
    exp_id: str,
    x_timezone_offset: int,
) -> dict[str, Any] | None:
    """Synchronous helper — compute experiment activity from FDB."""  # noqa: D401
    exp = entities.get_experiment(db, exp_id)
    if not exp:
        return None

    run_hashes = entities.get_runs_for_experiment(db, exp_id)

    num_runs = 0
    num_archived = 0
    num_active = 0
    activity_counter: Counter[str] = Counter()

    for rh in run_hashes:
        meta = get_run_meta(db, rh)
        if not meta or meta.get("pending_deletion"):
            continue
        num_runs += 1
        if meta.get("is_archived"):
            num_archived += 1
        if meta.get("active"):
            num_active += 1

        created = meta.get("created_at")
        if created:
            ts = created - x_timezone_offset * 60
            day = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC).strftime("%Y-%m-%dT%H:00:00")
            activity_counter[day] += 1

    return {
        "num_runs": num_runs,
        "num_archived_runs": num_archived,
        "num_active_runs": num_active,
        "activity_map": dict(activity_counter),
    }


@rest_router_experiments.get("/{exp_id}/activity/", response_model=ExperimentActivityApiOut)
async def experiment_runs_activity_api(
    exp_id: str,
    db: FdbDb,
    x_timezone_offset: Annotated[int, Header()] = 0,
) -> dict[str, Any]:
    """Return experiment activity (run counts, activity map by day).

    :param exp_id: Experiment UUID.
    :param x_timezone_offset: Timezone offset in minutes (header).
    :raises HTTPException: 404 if experiment not found.
    """
    data = await asyncio.to_thread(_fetch_experiment_activity, db, exp_id, x_timezone_offset)
    if data is None:
        raise HTTPException(status_code=404)
    return data
