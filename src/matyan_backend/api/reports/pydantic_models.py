from __future__ import annotations

from pydantic import BaseModel


class ReportOut(BaseModel):
    id: str
    name: str
    code: str | None = None
    description: str | None = None
    updated_at: float | None = None
    created_at: float | None = None


class ReportUpdateIn(BaseModel):
    name: str | None = None
    code: str | None = None
    description: str | None = None


class ReportCreateIn(BaseModel):
    name: str
    code: str | None = None
    description: str | None = None


ReportListOut = list[ReportOut]
