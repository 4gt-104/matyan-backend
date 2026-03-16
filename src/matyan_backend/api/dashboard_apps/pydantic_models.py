from __future__ import annotations

from pydantic import BaseModel


class ExploreStateCreateIn(BaseModel):
    type: str
    state: dict


class ExploreStateUpdateIn(BaseModel):
    type: str | None = None
    state: dict | None = None


class ExploreStateGetOut(BaseModel):
    id: str
    type: str
    updated_at: float | None = None
    created_at: float | None = None
    state: dict


ExploreStateListOut = list[ExploreStateGetOut]
