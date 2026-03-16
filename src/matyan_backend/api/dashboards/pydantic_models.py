from __future__ import annotations

from pydantic import BaseModel


class DashboardOut(BaseModel):
    id: str
    name: str
    description: str | None = None
    app_id: str | None = None
    app_type: str | None = None
    updated_at: float | None = None
    created_at: float | None = None


class DashboardUpdateIn(BaseModel):
    name: str | None = None
    description: str | None = None


class DashboardCreateIn(BaseModel):
    name: str
    description: str | None = None
    app_id: str | None = None
