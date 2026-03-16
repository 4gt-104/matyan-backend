"""Project REST endpoints: info, activity, params aggregation, pinned sequences.

All routes are under the ``/projects`` prefix on :data:`rest_router_projects`.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from fastapi import APIRouter, Header

from matyan_backend.deps import FdbDb  # noqa: TC001
from matyan_backend.storage import project as project_store

from ._pydantic_models import (
    ProjectActivityApiResponse,
    ProjectApiResponse,
    ProjectParamsOut,
    ProjectPinnedSequencesApiIn,
    ProjectPinnedSequencesApiResponse,
)

rest_router_projects = APIRouter(prefix="/projects")


def _get_project_info(db: FdbDb) -> dict:
    return project_store.get_project_info(db)


def _get_project_activity(db: FdbDb, tz_offset: int) -> dict:
    return project_store.get_project_activity(db, tz_offset=tz_offset)


def _get_project_params(db: FdbDb, sequence: tuple[str, ...], exclude_params: bool) -> dict:
    result = project_store.get_project_params_cached(
        db,
        sequence_types=sequence,
        exclude_params=exclude_params,
    )
    if exclude_params:
        result = {k: v for k, v in result.items() if k != "params"}
    return result


def _get_pinned_sequences(db: FdbDb) -> dict[str, list]:
    return {"sequences": project_store.get_pinned_sequences(db)}


def _update_pinned_sequences(db: FdbDb, raw: list) -> dict[str, list]:
    saved = project_store.set_pinned_sequences(db, raw)
    return {"sequences": saved}


@rest_router_projects.get("/", response_model=ProjectApiResponse)
async def get_project_api(db: FdbDb) -> dict:
    """Return project-level info (name, path, etc.)."""
    return await asyncio.to_thread(_get_project_info, db)


@rest_router_projects.get("/activity/", response_model=ProjectActivityApiResponse)
async def get_project_activity_api(
    db: FdbDb,
    x_timezone_offset: Annotated[int, Header()] = 0,
) -> dict:
    """Return project activity (e.g. run counts over time); uses timezone offset header."""
    return await asyncio.to_thread(_get_project_activity, db, x_timezone_offset)


@rest_router_projects.get(
    "/params/",
    response_model=ProjectParamsOut,
    response_model_exclude_none=True,
)
async def get_project_params_api(
    db: FdbDb,
    sequence: tuple[str, ...] = (),
    exclude_params: bool = False,
) -> dict:
    """Return aggregated params (and optionally sequence info) across runs."""
    return await asyncio.to_thread(_get_project_params, db, sequence, exclude_params)


@rest_router_projects.get("/pinned-sequences/", response_model=ProjectPinnedSequencesApiResponse)
async def get_pinned_sequences_api(db: FdbDb) -> dict[str, list]:
    """Return the list of pinned sequences for the project."""
    return await asyncio.to_thread(_get_pinned_sequences, db)


@rest_router_projects.post("/pinned-sequences/", response_model=ProjectPinnedSequencesApiResponse)
async def update_pinned_sequences_api(body: ProjectPinnedSequencesApiIn, db: FdbDb) -> dict[str, list]:
    """Update the list of pinned sequences."""
    raw = [s.model_dump() for s in body.sequences]
    return await asyncio.to_thread(_update_pinned_sequences, db, raw)


@rest_router_projects.get("/status/")
async def get_project_status_api() -> Literal["up-to-date"]:
    """Return project status (always ``up-to-date`` for this backend)."""
    return "up-to-date"
