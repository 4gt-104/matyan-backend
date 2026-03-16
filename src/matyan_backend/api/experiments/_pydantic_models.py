from __future__ import annotations  # noqa: I001


from pydantic import BaseModel

from uuid import UUID  # noqa: TC003


class ExperimentCreateRequest(BaseModel):
    name: str


class ExperimentUpdateRequest(BaseModel):
    name: str | None = ""
    description: str | None = ""
    archived: bool | None = None


class ExperimentGetOut(BaseModel):
    id: UUID
    name: str
    description: str | None = ""
    run_count: int
    archived: bool
    creation_time: float | None = None


ExperimentListOut = list[ExperimentGetOut]


class ExperimentUpdateOut(BaseModel):
    id: UUID
    status: str = "OK"


class ExperimentGetRunsResponse(BaseModel):
    class Run(BaseModel):
        run_id: str
        name: str
        creation_time: float
        end_time: float | None
        archived: bool

    id: UUID
    runs: list[Run]


class ExperimentActivityApiOut(BaseModel):
    num_runs: int
    num_archived_runs: int
    num_active_runs: int
    activity_map: dict[str, int] = {"2021-01-01": 54}
