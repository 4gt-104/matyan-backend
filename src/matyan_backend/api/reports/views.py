"""Report REST endpoints: CRUD for reports (mounted under ``/rest/reports``)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from matyan_backend.deps import FdbDb  # noqa: TC001
from matyan_backend.storage import entities

from .pydantic_models import ReportCreateIn, ReportListOut, ReportOut, ReportUpdateIn

reports_router = APIRouter()


def _report_to_out(r: dict) -> dict:
    return {
        "id": r["id"],
        "name": r.get("name", ""),
        "code": r.get("code"),
        "description": r.get("description"),
        "updated_at": r.get("updated_at"),
        "created_at": r.get("created_at"),
    }


def _list_reports(db: FdbDb) -> list[dict]:
    return [_report_to_out(r) for r in entities.list_reports(db)]


def _create_report(db: FdbDb, name: str, code: str | None, description: str | None) -> dict:
    r = entities.create_report(db, name, code=code, description=description)
    return _report_to_out(r)


def _get_report(db: FdbDb, report_id: str) -> dict | None:
    r = entities.get_report(db, report_id)
    if not r:
        return None
    return _report_to_out(r)


def _update_report(
    db: FdbDb,
    report_id: str,
    name: str | None,
    code: str | None,
    description: str | None,
) -> dict | None:
    r = entities.get_report(db, report_id)
    if not r:
        return None
    updates: dict = {}
    if name is not None:
        updates["name"] = name
    if code is not None:
        updates["code"] = code
    if description is not None:
        updates["description"] = description
    if updates:
        entities.update_report(db, report_id, **updates)
    updated = entities.get_report(db, report_id)
    if not updated:
        return None
    return _report_to_out(updated)


def _delete_report(db: object, report_id: str) -> bool:
    r = entities.get_report(db, report_id)
    if not r:
        return False
    entities.delete_report(db, report_id)
    return True


@reports_router.get("/", response_model=ReportListOut)
async def get_reports_api(db: FdbDb) -> list[dict]:
    """List all reports."""
    return await asyncio.to_thread(_list_reports, db)


@reports_router.post("/", response_model=ReportOut, status_code=201)
async def create_report_api(body: ReportCreateIn, db: FdbDb) -> dict:
    """Create a new report (name, optional code and description)."""
    return await asyncio.to_thread(_create_report, db, body.name, body.code, body.description)


@reports_router.get("/{report_id}/", response_model=ReportOut)
async def get_report_api(report_id: str, db: FdbDb) -> dict:
    """Get a single report by ID.

    :raises HTTPException: 404 if not found.
    """
    result = await asyncio.to_thread(_get_report, db, report_id)
    if result is None:
        raise HTTPException(status_code=404)
    return result


@reports_router.put("/{report_id}/", response_model=ReportOut)
async def update_report_api(report_id: str, body: ReportUpdateIn, db: FdbDb) -> dict:
    """Update report name, code, and/or description.

    :raises HTTPException: 404 if not found.
    """
    result = await asyncio.to_thread(_update_report, db, report_id, body.name, body.code, body.description)
    if result is None:
        raise HTTPException(status_code=404)
    return result


@reports_router.delete("/{report_id}/", status_code=204, response_model=None)
async def delete_report_api(report_id: str, db: FdbDb) -> None:
    """Delete a report.

    :raises HTTPException: 404 if not found.
    """
    found = await asyncio.to_thread(_delete_report, db, report_id)
    if not found:
        raise HTTPException(status_code=404)
