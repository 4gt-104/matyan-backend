from __future__ import annotations

from uuid import UUID  # noqa: TC003

from pydantic import BaseModel


class TagCreateIn(BaseModel):
    name: str
    color: str | None = ""
    description: str | None = ""


class TagUpdateIn(BaseModel):
    name: str | None = ""
    color: str | None = None
    description: str | None = None
    archived: bool | None = None


class TagUpdateOut(BaseModel):
    id: UUID
    status: str = "OK"


class TagGetOut(BaseModel):
    id: UUID
    name: str
    color: str | None = None
    description: str | None = None
    run_count: int = 0
    archived: bool


TagListOut = list[TagGetOut]


class TagGetRunsOut(BaseModel):
    class Run(BaseModel):
        run_id: str
        name: str
        experiment: str | None = None
        creation_time: float
        end_time: float | None

    id: UUID
    runs: list[Run]
