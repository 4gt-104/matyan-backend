"""Tests for api/runs/_views.py — proxy objects for MatyanQL evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from matyan_backend.api.runs._views import (
    AimObjectProxy,
    MetricsView,
    RunView,
    SequenceView,
    _SingleMetricView,
    _TagsProxy,
    build_props_dict,
)
from matyan_backend.storage import entities, runs

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


class TestAimObjectProxy:
    def test_getattr(self) -> None:
        proxy = AimObjectProxy({"lr": 0.01})
        assert proxy.lr == 0.01

    def test_getattr_nested(self) -> None:
        proxy = AimObjectProxy({"model": {"type": "cnn"}})
        assert isinstance(proxy.model, AimObjectProxy)
        assert proxy.model.type == "cnn"

    def test_getattr_missing(self) -> None:
        proxy = AimObjectProxy({"a": 1})
        assert proxy.b is None

    def test_getattr_private_raises(self) -> None:
        proxy = AimObjectProxy({"a": 1})
        with pytest.raises(AttributeError):
            _ = proxy._private  # noqa: SLF001

    def test_getitem(self) -> None:
        proxy = AimObjectProxy({"key": "val"})
        assert proxy["key"] == "val"

    def test_getitem_dict_returns_proxy(self) -> None:
        proxy = AimObjectProxy({"nested": {"x": 1}})
        result = proxy["nested"]
        assert isinstance(result, AimObjectProxy)
        assert result.x == 1

    def test_contains(self) -> None:
        proxy = AimObjectProxy({"a": 1, "b": 2})
        assert "a" in proxy
        assert "c" not in proxy

    def test_repr(self) -> None:
        proxy = AimObjectProxy(42)
        assert "42" in repr(proxy)

    def test_eq_proxy(self) -> None:
        a = AimObjectProxy({"x": 1})
        b = AimObjectProxy({"x": 1})
        assert a == b

    def test_eq_raw(self) -> None:
        proxy = AimObjectProxy(42)
        assert proxy == 42

    def test_bool_truthy(self) -> None:
        assert bool(AimObjectProxy({"a": 1})) is True

    def test_bool_falsy(self) -> None:
        assert bool(AimObjectProxy({})) is False
        assert bool(AimObjectProxy(None)) is False
        assert bool(AimObjectProxy(0)) is False


class TestSingleMetricView:
    def test_with_data(self) -> None:
        mv = _SingleMetricView({"last": 0.05, "last_step": 100})
        assert mv.last == 0.05
        assert mv.last_step == 100
        assert bool(mv) is True

    def test_empty(self) -> None:
        mv = _SingleMetricView({})
        assert mv.last is None
        assert mv.last_step is None
        assert bool(mv) is False


class TestMetricsView:
    def test_getitem_by_name(self, db: Database) -> None:
        runs.create_run(db, "mv1")
        runs.set_trace_info(db, "mv1", 0, "loss", dtype="float", last=0.1, last_step=50)

        view = MetricsView(db, "mv1")
        metric = view["loss"]
        assert isinstance(metric, _SingleMetricView)
        assert metric.last == 0.1

    def test_getitem_by_name_and_context(self, db: Database) -> None:
        runs.create_run(db, "mv2")
        runs.set_context(db, "mv2", 0, {})
        runs.set_trace_info(db, "mv2", 0, "loss", dtype="float", last=0.2)

        view = MetricsView(db, "mv2")
        metric = view["loss", {}]
        assert metric.last == 0.2

    def test_getitem_missing(self, db: Database) -> None:
        runs.create_run(db, "mv3")
        view = MetricsView(db, "mv3")
        metric = view["nonexistent"]
        assert bool(metric) is False

    def test_getitem_skips_wrong_name(self, db: Database) -> None:
        runs.create_run(db, "mv4")
        runs.set_trace_info(db, "mv4", 0, "acc", dtype="float", last=0.9)
        runs.set_trace_info(db, "mv4", 0, "loss", dtype="float", last=0.1)
        view = MetricsView(db, "mv4")
        metric = view["loss"]
        assert metric.last == 0.1

    def test_getitem_with_context_skips_mismatched(self, db: Database) -> None:
        """When context is specified but trace dict doesn't have a matching context,
        the lookup skips those traces and returns an empty result.
        """
        runs.create_run(db, "mv5")
        runs.set_trace_info(db, "mv5", 0, "loss", dtype="float", last=0.2)
        view = MetricsView(db, "mv5")
        metric = view["loss", {"subset": "nonexistent"}]
        assert bool(metric) is False


class TestRunView:
    def test_name_and_description(self, db: Database) -> None:
        runs.create_run(db, "rv1", name="Test Run", description="A test")
        meta = runs.get_run_meta(db, "rv1")
        rv = RunView(db, "rv1", meta)
        assert rv.name == "Test Run"
        assert rv.description == "A test"

    def test_experiment_property(self, db: Database) -> None:
        exp = entities.create_experiment(db, "exp-rv")
        runs.create_run(db, "rv2", experiment_id=exp["id"])
        meta = runs.get_run_meta(db, "rv2")
        rv = RunView(db, "rv2", meta)
        assert rv.experiment == "exp-rv"

    def test_experiment_none(self, db: Database) -> None:
        runs.create_run(db, "rv3")
        meta = runs.get_run_meta(db, "rv3")
        rv = RunView(db, "rv3", meta)
        assert rv.experiment is None

    def test_tags_proxy(self, db: Database) -> None:
        runs.create_run(db, "rv4")
        tag = entities.create_tag(db, "my-tag")
        entities.add_tag_to_run(db, "rv4", tag["id"])

        meta = runs.get_run_meta(db, "rv4")
        rv = RunView(db, "rv4", meta)
        assert "my-tag" in rv.tags

    def test_created_at(self, db: Database) -> None:
        runs.create_run(db, "rv5")
        meta = runs.get_run_meta(db, "rv5")
        rv = RunView(db, "rv5", meta)
        assert rv.created_at > 0
        assert rv.creation_time == rv.created_at

    def test_active_and_archived(self, db: Database) -> None:
        runs.create_run(db, "rv6")
        meta = runs.get_run_meta(db, "rv6")
        rv = RunView(db, "rv6", meta)
        assert rv.active is True
        assert rv.archived is False
        assert rv.is_archived is False

    def test_end_time_and_duration(self, db: Database) -> None:
        runs.create_run(db, "rv7")
        meta = runs.get_run_meta(db, "rv7")
        finalized = meta["created_at"] + 60.0
        runs.update_run_meta(db, "rv7", finalized_at=finalized)
        meta = runs.get_run_meta(db, "rv7")
        rv = RunView(db, "rv7", meta)
        assert rv.end_time == finalized
        assert abs(rv.duration - 60.0) < 1.0

    def test_duration_no_end_time(self, db: Database) -> None:
        runs.create_run(db, "rv8")
        meta = runs.get_run_meta(db, "rv8")
        rv = RunView(db, "rv8", meta)
        assert rv.duration == 0.0

    def test_getitem(self, db: Database) -> None:
        runs.create_run(db, "rv9")
        runs.set_run_attrs(db, "rv9", ("hparams",), {"lr": 0.01})
        meta = runs.get_run_meta(db, "rv9")
        rv = RunView(db, "rv9", meta)
        result = rv["hparams"]
        assert isinstance(result, AimObjectProxy)
        assert result.lr == 0.01

    def test_getitem_scalar(self, db: Database) -> None:
        runs.create_run(db, "rv9b")
        runs.set_run_attrs(db, "rv9b", (), {"scalar_key": 42})
        meta = runs.get_run_meta(db, "rv9b")
        rv = RunView(db, "rv9b", meta)
        assert rv["scalar_key"] == 42

    def test_getattr_delegates_to_attrs(self, db: Database) -> None:
        runs.create_run(db, "rv10")
        runs.set_run_attrs(db, "rv10", (), {"custom_field": "custom_value"})
        meta = runs.get_run_meta(db, "rv10")
        rv = RunView(db, "rv10", meta)
        assert rv.custom_field == "custom_value"

    def test_getattr_returns_none_for_missing(self, db: Database) -> None:
        runs.create_run(db, "rv10b")
        runs.set_run_attrs(db, "rv10b", (), {"a": 1})
        meta = runs.get_run_meta(db, "rv10b")
        rv = RunView(db, "rv10b", meta)
        assert rv.nonexistent_field is None

    def test_getattr_known_property_raises(self, db: Database) -> None:
        runs.create_run(db, "rv11")
        meta = runs.get_run_meta(db, "rv11")
        rv = RunView(db, "rv11", meta)
        with pytest.raises(AttributeError):
            _ = rv._private  # noqa: SLF001

    def test_metrics_property(self, db: Database) -> None:
        runs.create_run(db, "rv12")
        meta = runs.get_run_meta(db, "rv12")
        rv = RunView(db, "rv12", meta)
        assert isinstance(rv.metrics, MetricsView)

    def test_loads_meta_lazily(self, db: Database) -> None:
        runs.create_run(db, "rv13")
        rv = RunView(db, "rv13")
        assert rv.name is not None


class TestTagsProxy:
    def test_contains(self, db: Database) -> None:
        runs.create_run(db, "tp1")
        tag = entities.create_tag(db, "alpha")
        entities.add_tag_to_run(db, "tp1", tag["id"])

        proxy = _TagsProxy(db, "tp1")
        assert "alpha" in proxy
        assert "beta" not in proxy

    def test_contains_method(self, db: Database) -> None:
        runs.create_run(db, "tp2")
        tag = entities.create_tag(db, "beta-tag")
        entities.add_tag_to_run(db, "tp2", tag["id"])

        proxy = _TagsProxy(db, "tp2")
        assert proxy.contains("beta-tag") is True
        assert proxy.contains("gamma") is False


class TestSequenceView:
    def test_properties(self) -> None:
        rv = MagicMock()
        trace_info = {"last": 0.5, "last_step": 10, "context_id": 0}
        sv = SequenceView("loss", {"subset": "train"}, rv, trace_info=trace_info)
        assert sv.name == "loss"
        assert sv.last == 0.5
        assert sv.last_step == 10
        assert isinstance(sv.context, AimObjectProxy)
        assert sv.context.subset == "train"

    def test_no_trace_info(self) -> None:
        rv = MagicMock()
        sv = SequenceView("acc", {}, rv)
        assert sv.last is None
        assert sv.last_step is None


class TestBuildPropsDict:
    def test_basic(self, db: Database) -> None:
        runs.create_run(db, "bpd1", name="My Run")
        meta = runs.get_run_meta(db, "bpd1")
        meta["hash"] = "bpd1"
        props = build_props_dict(meta, db)
        assert props["name"] == "My Run"
        assert props["archived"] is False
        assert props["active"] is True
        assert props["experiment"]["id"] is None
        assert props["tags"] == []

    def test_with_experiment_and_tags(self, db: Database) -> None:
        exp = entities.create_experiment(db, "bpd-exp")
        runs.create_run(db, "bpd2", experiment_id=exp["id"])
        tag = entities.create_tag(db, "bpd-tag")
        entities.add_tag_to_run(db, "bpd2", tag["id"])

        meta = runs.get_run_meta(db, "bpd2")
        meta["hash"] = "bpd2"
        props = build_props_dict(meta, db)
        assert props["experiment"]["name"] == "bpd-exp"
        assert len(props["tags"]) == 1
        assert props["tags"][0]["name"] == "bpd-tag"
