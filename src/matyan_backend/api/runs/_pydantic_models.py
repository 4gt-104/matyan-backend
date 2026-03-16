from __future__ import annotations

from uuid import UUID  # noqa: TC003

from pydantic import BaseModel


# response models
class EncodedNumpyArray(BaseModel):
    type: str = "numpy"
    shape: int = 0
    dtype: str = "float64"
    blob: bytes = b""


class TraceBase(BaseModel):
    context: dict
    name: str


class TraceOverview(TraceBase):
    last_value: float = 0.1


class TraceBaseView(TraceBase):
    iters: list[int]


class MetricsBaseView(TraceBaseView):
    values: list[float]


RunMetricsBatchApiOut = list[MetricsBaseView]


class TraceAlignedView(TraceBase):
    x_axis_values: EncodedNumpyArray | None = None
    x_axis_iters: EncodedNumpyArray | None = None


RunMetricCustomAlignApiOut = dict[str, list[TraceAlignedView]]


class TraceFullView(TraceAlignedView):
    slice: tuple[int, int, int]
    values: EncodedNumpyArray | None = None
    iters: EncodedNumpyArray | None = None
    epochs: EncodedNumpyArray | None = None
    timestamps: EncodedNumpyArray | None = None


class PropsView(BaseModel):
    class Tag(BaseModel):
        id: UUID
        name: str
        color: str | None = None
        description: str | None = None

    class Experiment(BaseModel):
        id: UUID | None = None
        name: str | None = None
        description: str | None = None

    name: str | None = None
    description: str | None = None
    experiment: Experiment = Experiment()
    tags: list[Tag] | None = []
    creation_time: float
    end_time: float | None
    archived: bool
    active: bool


class MetricSearchRunView(BaseModel):
    params: dict
    traces: list[TraceFullView]
    props: PropsView


class ArtifactInfo(BaseModel):
    name: str
    path: str
    uri: str


class RunInfoOut(BaseModel):
    params: dict
    traces: dict[str, list[TraceOverview]]
    props: PropsView
    artifacts: list[ArtifactInfo]


RunMetricSearchApiOut = dict[str, MetricSearchRunView]


class RunSearchRunView(BaseModel):
    params: dict | None
    traces: list[TraceOverview] | None
    props: PropsView


RunSearchApiOut = dict[str, RunSearchRunView]


class RunActiveOut(BaseModel):
    traces: dict[str, list[TraceOverview]]
    props: PropsView


# request models
class AlignedTraceIn(BaseModel):
    context: dict
    name: str
    slice: tuple[int, int, int]


class AlignedRunIn(BaseModel):
    run_id: str
    traces: list[AlignedTraceIn]


class MetricAlignApiIn(BaseModel):
    align_by: str
    runs: list[AlignedRunIn]


RunTracesBatchApiIn = list[TraceBase]


# structured run models
class StructuredRunUpdateIn(BaseModel):
    name: str | None = None
    description: str | None = None
    archived: bool | None = None
    experiment: str | None = None
    active: bool | None = None


class StructuredRunUpdateOut(BaseModel):
    id: str
    status: str = "OK"


class StructuredRunsArchivedOut(BaseModel):
    status: str = "OK"


class StructuredRunAddTagIn(BaseModel):
    tag_name: str


class StructuredRunAddTagOut(BaseModel):
    id: str
    tag_id: UUID
    status: str = "OK"


class StructuredRunRemoveTagOut(BaseModel):
    id: str
    removed: bool
    status: str = "OK"


class QuerySyntaxErrorOut(BaseModel):
    class SE(BaseModel):
        name: str
        statement: str
        line: int
        offset: int

    detail: SE


URIBatchIn = list[str]
RunsBatchIn = list[str]


# Custom object Models "Fully Generic"


class BaseRangeInfo(BaseModel):
    record_range_used: tuple[int, int]
    record_range_total: tuple[int, int]
    index_range_used: tuple[int, int] | None
    index_range_total: tuple[int, int] | None


class ObjectSequenceBaseView(BaseRangeInfo, TraceBaseView):
    values: list


class ObjectSequenceFullView(TraceBaseView):
    values: list
    iters: list[int]
    epochs: list[int]
    timestamps: list[float]


class ObjectSearchRunView(BaseModel):
    params: dict
    traces: list[ObjectSequenceFullView]
    ranges: BaseRangeInfo
    props: PropsView


# Custom objects
class ImageInfo(BaseModel):
    caption: str
    width: int
    height: int
    blob_uri: str
    index: int


class TextInfo(BaseModel):
    data: str
    index: int


class AudioInfo(BaseModel):
    caption: str
    blob_uri: str
    index: int


class DistributionInfo(BaseModel):
    data: EncodedNumpyArray
    bin_count: int
    range: tuple[int | float, int | float]


class FigureInfo(BaseModel):
    blob_uri: str


class NoteIn(BaseModel):
    content: str


ImageList = list[ImageInfo]
TextList = list[TextInfo]
AudioList = list[AudioInfo]
