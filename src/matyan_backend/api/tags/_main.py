"""Tag REST endpoints: CRUD and tagged runs listing.

All routes are under the ``/tags`` prefix on :data:`tags_router`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from matyan_backend.deps import FdbDb, KafkaProducerDep  # noqa: TC001
from matyan_backend.kafka.producer import emit_control_event
from matyan_backend.storage import entities
from matyan_backend.storage.runs import get_run_meta

from ._pydantic_models import (
    TagCreateIn,
    TagGetOut,
    TagGetRunsOut,
    TagListOut,
    TagUpdateIn,
    TagUpdateOut,
)

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database

tags_router = APIRouter(prefix="/tags")


def _tag_to_out(tag: dict) -> dict:
    return {
        "id": tag["id"],
        "name": tag.get("name", ""),
        "color": tag.get("color"),
        "description": tag.get("description"),
        "run_count": tag.get("run_count", 0),
        "archived": tag.get("is_archived", False),
    }


def _list_tags(db: FdbDb) -> list[dict]:
    return [_tag_to_out(t) for t in entities.list_tags(db)]


@tags_router.get("/", response_model=TagListOut)
async def get_tags_list_api(db: FdbDb) -> list[dict]:
    """List all tags."""
    return await asyncio.to_thread(_list_tags, db)


def _search_tags(db: Database, q: str) -> list[dict]:
    tags = entities.list_tags(db)
    query = q.strip().lower()
    if query:
        tags = [t for t in tags if query in t.get("name", "").lower()]
    return [_tag_to_out(t) for t in tags]


@tags_router.get("/search/", response_model=TagListOut)
async def search_tags_by_name_api(db: FdbDb, q: str = "") -> list[dict]:
    """Search tags by name (case-insensitive substring)."""
    return await asyncio.to_thread(_search_tags, db, q)


@tags_router.post("/", response_model=TagUpdateOut)
async def create_tag_api(tag_in: TagCreateIn, db: FdbDb) -> dict[str, str]:
    """Create a new tag (name, optional color and description).

    :raises HTTPException: 400 if name is duplicate or invalid.
    """
    try:
        tag = await asyncio.to_thread(
            entities.create_tag,
            db,
            tag_in.name.strip(),
            color=tag_in.color.strip() if tag_in.color else None,
            description=tag_in.description.strip() if tag_in.description else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"id": tag["id"], "status": "OK"}


@tags_router.get("/{tag_id}/", response_model=TagGetOut)
async def get_tag_api(tag_id: str, db: FdbDb) -> dict:
    """Get a single tag by ID.

    :raises HTTPException: 404 if not found.
    """
    tag = await asyncio.to_thread(entities.get_tag, db, tag_id)
    if not tag:
        raise HTTPException(status_code=404)
    return _tag_to_out(tag)


def _update_tag(db: Database, tag_id: str, tag_in: TagUpdateIn) -> bool:
    tag = entities.get_tag(db, tag_id)
    if not tag:
        return False
    updates: dict = {}
    if tag_in.name:
        updates["name"] = tag_in.name.strip()
    if tag_in.color is not None:
        updates["color"] = tag_in.color.strip()
    if tag_in.description is not None:
        updates["description"] = tag_in.description.strip()
    if tag_in.archived is not None:
        updates["is_archived"] = tag_in.archived
    if updates:
        entities.update_tag(db, tag_id, **updates)
    return True


@tags_router.put("/{tag_id}/", response_model=TagUpdateOut)
async def update_tag_properties_api(tag_id: str, tag_in: TagUpdateIn, db: FdbDb) -> dict[str, str]:
    """Update tag name, color, description, or archived state.

    :raises HTTPException: 404 if tag not found.
    """
    found = await asyncio.to_thread(_update_tag, db, tag_id, tag_in)
    if not found:
        raise HTTPException(status_code=404)
    return {"id": tag_id, "status": "OK"}


def _delete_tag(db: Database, tag_id: str) -> bool:
    tag = entities.get_tag(db, tag_id)
    if not tag:
        return False
    entities.delete_tag(db, tag_id)
    return True


@tags_router.delete("/{tag_id}/")
async def delete_tag_api(tag_id: str, db: FdbDb, producer: KafkaProducerDep) -> None:
    """Delete a tag and emit control event for async side effects.

    :raises HTTPException: 404 if tag not found.
    """
    found = await asyncio.to_thread(_delete_tag, db, tag_id)
    if not found:
        raise HTTPException(status_code=404)
    await emit_control_event(producer, "tag_deleted", tag_id=tag_id)


def _fetch_tagged_runs(db: Database, tag_id: str) -> dict | None:
    """Fetch tag and associated runs from FDB."""
    tag = entities.get_tag(db, tag_id)
    if not tag:
        return None
    run_hashes = entities.get_runs_for_tag(db, tag_id)
    tag_runs = []
    for rh in run_hashes:
        meta = get_run_meta(db, rh)
        if not meta or meta.get("pending_deletion"):
            continue
        exp_name: str | None = None
        exp_id = meta.get("experiment_id")
        if exp_id:
            exp = entities.get_experiment(db, exp_id)
            exp_name = exp.get("name") if exp else None
        tag_runs.append(
            {
                "run_id": rh,
                "name": meta.get("name", ""),
                "experiment": exp_name,
                "creation_time": meta.get("created_at", 0),
                "end_time": meta.get("finalized_at"),
            },
        )
    return {"id": tag_id, "runs": tag_runs}


@tags_router.get("/{tag_id}/runs/", response_model=TagGetRunsOut)
async def get_tagged_runs_api(tag_id: str, db: FdbDb) -> dict:
    """List runs that have the given tag (with run_id, name, experiment, times).

    :raises HTTPException: 404 if tag not found.
    """
    data = await asyncio.to_thread(_fetch_tagged_runs, db, tag_id)
    if data is None:
        raise HTTPException(status_code=404)
    return data
