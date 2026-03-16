"""Dashboard apps (explore state) REST endpoints: CRUD (mounted under ``/rest/apps``)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from matyan_backend.deps import FdbDb  # noqa: TC001
from matyan_backend.storage import entities

from .pydantic_models import (
    ExploreStateCreateIn,
    ExploreStateGetOut,
    ExploreStateListOut,
    ExploreStateUpdateIn,
)

dashboard_apps_router = APIRouter()


def _app_to_out(a: dict) -> dict:
    return {
        "id": a["id"],
        "type": a.get("type", ""),
        "updated_at": a.get("updated_at"),
        "created_at": a.get("created_at"),
        "state": a.get("state", {}),
    }


def _list_apps(db: FdbDb) -> list[dict]:
    return [_app_to_out(a) for a in entities.list_dashboard_apps(db)]


def _create_app(db: FdbDb, app_type: str, state: dict | None) -> dict:
    a = entities.create_dashboard_app(db, app_type, state=state)
    return _app_to_out(a)


def _get_app(db: FdbDb, app_id: str) -> dict | None:
    a = entities.get_dashboard_app(db, app_id)
    if not a:
        return None
    return _app_to_out(a)


def _update_app(db: FdbDb, app_id: str, app_type: str | None, state: dict | None) -> dict | None:
    a = entities.get_dashboard_app(db, app_id)
    if not a:
        return None
    updates: dict = {}
    if app_type is not None:
        updates["type"] = app_type
    if state is not None:
        updates["state"] = state
    if updates:
        entities.update_dashboard_app(db, app_id, **updates)
    updated = entities.get_dashboard_app(db, app_id)
    if not updated:
        return None
    return _app_to_out(updated)


def _delete_app(db: object, app_id: str) -> bool:
    a = entities.get_dashboard_app(db, app_id)
    if not a:
        return False
    entities.delete_dashboard_app(db, app_id)
    return True


@dashboard_apps_router.get("/", response_model=ExploreStateListOut)
async def get_apps_api(db: FdbDb) -> list[dict]:
    """List all dashboard apps (explore states)."""
    return await asyncio.to_thread(_list_apps, db)


@dashboard_apps_router.post("/", response_model=ExploreStateGetOut, status_code=201)
async def create_app_api(body: ExploreStateCreateIn, db: FdbDb) -> dict:
    """Create a new dashboard app (type and optional state)."""
    return await asyncio.to_thread(_create_app, db, body.type, body.state)


@dashboard_apps_router.get("/{app_id}/", response_model=ExploreStateGetOut)
async def get_app_api(app_id: str, db: FdbDb) -> dict:
    """Get a single dashboard app by ID.

    :raises HTTPException: 404 if not found.
    """
    result = await asyncio.to_thread(_get_app, db, app_id)
    if result is None:
        raise HTTPException(status_code=404)
    return result


@dashboard_apps_router.put("/{app_id}/", response_model=ExploreStateGetOut)
async def update_app_api(app_id: str, body: ExploreStateUpdateIn, db: FdbDb) -> dict:
    """Update dashboard app type and/or state.

    :raises HTTPException: 404 if not found.
    """
    result = await asyncio.to_thread(_update_app, db, app_id, body.type, body.state)
    if result is None:
        raise HTTPException(status_code=404)
    return result


@dashboard_apps_router.delete("/{app_id}/", status_code=204, response_model=None)
async def delete_app_api(app_id: str, db: FdbDb) -> None:
    """Delete a dashboard app.

    :raises HTTPException: 404 if not found.
    """
    found = await asyncio.to_thread(_delete_app, db, app_id)
    if not found:
        raise HTTPException(status_code=404)
