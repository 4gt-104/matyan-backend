"""Dashboard REST endpoints: CRUD for dashboards (mounted under ``/rest/dashboards``)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException

from matyan_backend.deps import FdbDb  # noqa: TC001
from matyan_backend.storage import entities

from .pydantic_models import DashboardCreateIn, DashboardOut, DashboardUpdateIn

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database

dashboards_router = APIRouter()


def _dash_to_out(d: dict, db: Database | None = None) -> dict[str, Any]:
    app_type: str | None = None
    app_id = d.get("app_id")
    if app_id and db is not None:
        app = entities.get_dashboard_app(db, app_id)
        if app:
            app_type = app.get("type")
    return {
        "id": d["id"],
        "name": d.get("name", ""),
        "description": d.get("description"),
        "app_id": app_id,
        "app_type": app_type,
        "updated_at": d.get("updated_at"),
        "created_at": d.get("created_at"),
    }


def _list_dashboards(db: FdbDb) -> list[dict]:
    dashboards = entities.list_dashboards(db)
    return [_dash_to_out(d, db) for d in dashboards if not d.get("is_archived")]


def _create_dashboard(db: FdbDb, name: str, description: str | None, app_id: str | None) -> dict:
    d = entities.create_dashboard(db, name, description=description, app_id=app_id)
    return _dash_to_out(d, db)


def _get_dashboard(db: FdbDb, dashboard_id: str) -> dict | None:
    d = entities.get_dashboard(db, dashboard_id)
    if not d:
        return None
    return _dash_to_out(d, db)


def _update_dashboard(db: FdbDb, dashboard_id: str, name: str | None, description: str | None) -> dict | None:
    d = entities.get_dashboard(db, dashboard_id)
    if not d:
        return None
    updates: dict = {}
    if name is not None:
        updates["name"] = name
    if description is not None:
        updates["description"] = description
    if updates:
        entities.update_dashboard(db, dashboard_id, **updates)
    updated = entities.get_dashboard(db, dashboard_id)
    if not updated:
        return None
    return _dash_to_out(updated, db)


def _delete_dashboard(db: FdbDb, dashboard_id: str) -> bool:
    d = entities.get_dashboard(db, dashboard_id)
    if not d:
        return False
    entities.update_dashboard(db, dashboard_id, is_archived=True)
    return True


@dashboards_router.get("/", response_model=list[DashboardOut])
async def get_dashboards_api(db: FdbDb) -> list[dict]:
    """List all non-archived dashboards."""
    return await asyncio.to_thread(_list_dashboards, db)


@dashboards_router.post("/", response_model=DashboardOut, status_code=201)
async def create_dashboard_api(body: DashboardCreateIn, db: FdbDb) -> dict:
    """Create a new dashboard (name, optional description and app_id)."""
    return await asyncio.to_thread(
        _create_dashboard,
        db,
        body.name,
        body.description,
        str(body.app_id) if body.app_id else None,
    )


@dashboards_router.get("/{dashboard_id}/", response_model=DashboardOut)
async def get_dashboard_api(dashboard_id: str, db: FdbDb) -> dict:
    """Get a single dashboard by ID.

    :raises HTTPException: 404 if not found.
    """
    result = await asyncio.to_thread(_get_dashboard, db, dashboard_id)
    if result is None:
        raise HTTPException(status_code=404)
    return result


@dashboards_router.put("/{dashboard_id}/", response_model=DashboardOut)
async def update_dashboard_api(dashboard_id: str, body: DashboardUpdateIn, db: FdbDb) -> dict:
    """Update dashboard name and/or description.

    :raises HTTPException: 404 if not found.
    """
    result = await asyncio.to_thread(_update_dashboard, db, dashboard_id, body.name, body.description)
    if result is None:
        raise HTTPException(status_code=404)
    return result


@dashboards_router.delete("/{dashboard_id}/", status_code=204, response_model=None)
async def delete_dashboard_api(dashboard_id: str, db: FdbDb) -> None:
    """Archive a dashboard (soft delete).

    :raises HTTPException: 404 if not found.
    """
    found = await asyncio.to_thread(_delete_dashboard, db, dashboard_id)
    if not found:
        raise HTTPException(status_code=404)
