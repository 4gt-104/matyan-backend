from __future__ import annotations

from pydantic import BaseModel


class ProjectApiResponse(BaseModel):
    name: str
    path: str
    description: str
    telemetry_enabled: int
    warn_index: bool | None = False
    warn_runs: bool | None = False


class ProjectParamsOut(BaseModel):
    params: dict | None = None
    metric: dict[str, list] | None = None
    images: dict[str, list] | None = None
    texts: dict[str, list] | None = None
    figures: dict[str, list] | None = None
    distributions: dict[str, list] | None = None
    audios: dict[str, list] | None = None


class ProjectActivityApiResponse(BaseModel):
    num_experiments: int
    num_runs: int
    num_archived_runs: int
    num_active_runs: int
    activity_map: dict[str, int] = {"2021-01-01": 54}


class Sequence(BaseModel):
    name: str
    context: dict


class ProjectPinnedSequencesApiResponse(BaseModel):
    sequences: list[Sequence] = []


class ProjectPinnedSequencesApiIn(BaseModel):
    sequences: list[Sequence]
