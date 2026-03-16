"""Proxy objects for MatyanQL query evaluation.

``RunView`` and ``SequenceView`` provide lazy, attribute-style access to
run / metric data stored in FDB, satisfying the interface expected by
``RestrictedPythonQuery.check(run=..., metric=...)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database

from matyan_backend.storage import entities
from matyan_backend.storage.runs import (
    get_run_attrs,
    get_run_meta,
    get_run_traces_info,
)


class AimObjectProxy:
    """Wraps a dict so attribute access works like item access.

    Enables ``run['hparams'].lr`` or ``metric.context.subset``.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Any) -> None:  # noqa: ANN401
        self._data = data

    def __getattr__(self, item: str) -> Any:  # noqa: ANN401
        if item.startswith("_"):
            raise AttributeError(item)
        try:
            val = self._data[item]
        except (KeyError, TypeError, IndexError):
            return None
        if isinstance(val, dict):
            return AimObjectProxy(val)
        return val

    def __getitem__(self, key: Any) -> Any:  # noqa: ANN401
        val = self._data[key]
        if isinstance(val, dict):
            return AimObjectProxy(val)
        return val

    def __contains__(self, item: Any) -> bool:  # noqa: ANN401
        return item in self._data

    def __repr__(self) -> str:
        return f"AimObjectProxy({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AimObjectProxy):
            return self._data == other._data
        return self._data == other

    __hash__ = None  # mutable proxy — not hashable

    def __bool__(self) -> bool:
        return bool(self._data)


class MetricsView:
    """Proxy for ``run.metrics['loss']`` or ``run.metrics['loss', ctx]``."""

    def __init__(self, db: Database, run_hash: str) -> None:
        self._db = db
        self._run_hash = run_hash
        self._traces_info: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self._traces_info is None:
            self._traces_info = get_run_traces_info(self._db, self._run_hash)
        return self._traces_info

    def __getitem__(self, key: Any) -> _SingleMetricView:  # noqa: ANN401
        if isinstance(key, tuple):
            name, ctx = key
        else:
            name, ctx = key, {}
        traces = self._load()
        for t in traces:
            if t.get("name") != name:
                continue
            if ctx and t.get("context", {}) != ctx:
                continue
            return _SingleMetricView(t)
        return _SingleMetricView({})


class _SingleMetricView:
    """Proxy for one metric trace, providing ``.last``, ``.first``, etc."""

    __slots__ = ("_info",)

    def __init__(self, info: dict) -> None:
        self._info = info

    @property
    def last(self) -> Any:  # noqa: ANN401
        return self._info.get("last")

    @property
    def last_step(self) -> int | None:
        return self._info.get("last_step")

    def __bool__(self) -> bool:
        return bool(self._info)


class RunView:
    """Proxy for run data consumed by MatyanQL ``RestrictedPythonQuery.check(run=...)``.

    Provides both attribute access (``run.name``, ``run.experiment``) and
    item access (``run['hparams']``).  Attribute values are loaded lazily
    from FDB.
    """

    def __init__(
        self,
        db: Database,
        run_hash: str,
        meta: dict | None = None,
        *,
        tz_offset: int = 0,
    ) -> None:
        self.hash = run_hash
        self._db = db
        self._meta = meta or get_run_meta(db, run_hash)
        self._attrs: dict | None = None
        self._tz_offset = tz_offset

    # ---- structured properties ----

    @property
    def name(self) -> str:
        return self._meta.get("name", "")

    @property
    def description(self) -> str:
        return self._meta.get("description", "")

    @property
    def experiment(self) -> str | None:
        exp_id = self._meta.get("experiment_id")
        if not exp_id:
            return None
        exp = entities.get_experiment(self._db, exp_id)
        return exp.get("name") if exp else None

    @property
    def tags(self) -> _TagsProxy:
        return _TagsProxy(self._db, self.hash)

    @property
    def created_at(self) -> float:
        return self._meta.get("created_at", 0)

    @property
    def creation_time(self) -> float:
        return self.created_at

    @property
    def end_time(self) -> float | None:
        return self._meta.get("finalized_at")

    @property
    def duration(self) -> float:
        e = self.end_time
        if e is None:
            return 0.0
        return e - self.created_at

    @property
    def active(self) -> bool:
        return self._meta.get("active", False)

    @property
    def archived(self) -> bool:
        return self._meta.get("is_archived", False)

    @property
    def is_archived(self) -> bool:
        return self.archived

    @property
    def metrics(self) -> MetricsView:
        return MetricsView(self._db, self.hash)

    # ---- item access  (run['hparams']) ----

    def _load_attrs(self) -> dict:
        if self._attrs is None:
            val = get_run_attrs(self._db, self.hash)
            self._attrs = val if isinstance(val, dict) else {}
        return self._attrs

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        attrs = self._load_attrs()
        val = attrs.get(key)
        if isinstance(val, dict):
            return AimObjectProxy(val)
        return val

    def __getattr__(self, item: str) -> Any:  # noqa: ANN401
        if item.startswith("_") or item in {
            "hash",
            "name",
            "description",
            "experiment",
            "tags",
            "created_at",
            "creation_time",
            "end_time",
            "duration",
            "active",
            "archived",
            "is_archived",
            "metrics",
        }:
            raise AttributeError(item)
        attrs = self._load_attrs()
        val = attrs.get(item)
        if isinstance(val, dict):
            return AimObjectProxy(val)
        return val


class _TagsProxy:
    """Supports ``run.tags`` queries like ``'production' in run.tags``."""

    def __init__(self, db: Database, run_hash: str) -> None:
        self._db = db
        self._run_hash = run_hash
        self._names: list[str] | None = None

    def _load(self) -> list[str]:
        if self._names is None:
            tags = entities.get_tags_for_run(self._db, self._run_hash)
            self._names = [t.get("name", "") for t in tags]
        return self._names

    def __contains__(self, item: str) -> bool:
        return item in self._load()

    def contains(self, name: str) -> bool:
        return name in self


class SequenceView:
    """Proxy for metric/sequence data used in MatyanQL queries.

    Satisfies ``metric.name``, ``metric.context.subset``, ``metric.last``.
    """

    def __init__(self, name: str, context: dict, run_view: RunView, trace_info: dict | None = None) -> None:
        self.name = name
        self._context = context
        self.run = run_view
        self._trace_info = trace_info or {}

    @property
    def context(self) -> AimObjectProxy:
        return AimObjectProxy(self._context)

    @property
    def last(self) -> Any:  # noqa: ANN401
        return self._trace_info.get("last")

    @property
    def last_step(self) -> int | None:
        return self._trace_info.get("last_step")


def build_props_dict(meta: dict, db: Database) -> dict:
    """Build a ``PropsView``-compatible dict from run meta + db lookups."""
    run_hash = meta.get("hash", "")
    tags_raw = entities.get_tags_for_run(db, run_hash)
    exp: dict | None = None
    exp_id = meta.get("experiment_id")
    if exp_id:
        exp = entities.get_experiment(db, exp_id)
    return build_props_from_bundle(meta, tags_raw, exp)


def build_props_from_bundle(meta: dict, tags_raw: list[dict], experiment: dict | None) -> dict:
    """Build a ``PropsView``-compatible dict from pre-fetched data (no FDB calls)."""
    tags = [
        {"id": t["id"], "name": t.get("name", ""), "color": t.get("color"), "description": t.get("description")}
        for t in tags_raw
    ]
    exp_dict: dict = {"id": None, "name": None, "description": None}
    if experiment:
        exp_dict = {
            "id": experiment["id"],
            "name": experiment.get("name", ""),
            "description": experiment.get("description", ""),
        }

    return {
        "name": meta.get("name"),
        "description": meta.get("description"),
        "experiment": exp_dict,
        "tags": tags,
        "creation_time": meta.get("created_at", 0),
        "end_time": meta.get("finalized_at"),
        "archived": meta.get("is_archived", False),
        "active": meta.get("active", False),
    }
