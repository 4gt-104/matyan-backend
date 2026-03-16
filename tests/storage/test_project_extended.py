"""Extended tests for storage/project.py — params aggregation, project info, activity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.storage import runs
from matyan_backend.storage.project import (
    get_project_activity,
    get_project_info,
    get_project_params,
    set_project_info,
)

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


class TestSetProjectInfo:
    def test_set_and_get_name(self, db: Database) -> None:
        set_project_info(db, name="Test Project", description="A test")
        info = get_project_info(db)
        assert info["name"] == "Test Project"
        assert info["description"] == "A test"

    def test_defaults(self, db: Database) -> None:
        info = get_project_info(db)
        assert info["name"] == "My project"
        assert info["path"] == ""
        assert info["telemetry_enabled"] == 0


class TestGetProjectActivity:
    def test_empty(self, db: Database) -> None:
        activity = get_project_activity(db)
        assert activity["num_runs"] == 0
        assert activity["num_archived_runs"] == 0
        assert activity["num_active_runs"] == 0

    def test_with_runs(self, db: Database) -> None:
        runs.create_run(db, "act1")
        runs.create_run(db, "act2")
        runs.update_run_meta(db, "act2", is_archived=True)

        activity = get_project_activity(db)
        assert activity["num_runs"] == 2
        assert activity["num_archived_runs"] == 1
        assert activity["num_active_runs"] == 2
        assert isinstance(activity["activity_map"], dict)

    def test_with_timezone_offset(self, db: Database) -> None:
        runs.create_run(db, "act_tz")
        activity = get_project_activity(db, tz_offset=300)
        assert activity["num_runs"] >= 1


class TestGetProjectParams:
    def test_empty(self, db: Database) -> None:
        result = get_project_params(db)
        assert result["params"] == {}
        assert result["metric"] == {}

    def test_with_metric_traces(self, db: Database) -> None:
        runs.create_run(db, "pp1")
        runs.set_run_attrs(db, "pp1", (), {"lr": 0.01})
        runs.set_context(db, "pp1", 0, {})
        runs.set_trace_info(db, "pp1", 0, "loss", dtype="float", last=0.1)

        result = get_project_params(db)
        assert "lr" in result["params"]
        assert "loss" in result["metric"]

    def test_custom_object_dtypes_bucketed(self, db: Database) -> None:
        runs.create_run(db, "pp2")
        runs.set_context(db, "pp2", 0, {})
        runs.set_trace_info(db, "pp2", 0, "train_images", dtype="image")
        runs.set_trace_info(db, "pp2", 0, "predictions", dtype="text")
        runs.set_trace_info(db, "pp2", 0, "hist", dtype="distribution")
        runs.set_trace_info(db, "pp2", 0, "sounds", dtype="audio")
        runs.set_trace_info(db, "pp2", 0, "plots", dtype="figure")

        result = get_project_params(db)
        assert "train_images" in result["images"]
        assert "predictions" in result["texts"]
        assert "hist" in result["distributions"]
        assert "sounds" in result["audios"]
        assert "plots" in result["figures"]

    def test_exclude_params_skips_attrs(self, db: Database) -> None:
        runs.create_run(db, "pp3")
        runs.set_run_attrs(db, "pp3", (), {"lr": 0.01, "batch_size": 32})
        runs.set_context(db, "pp3", 0, {})
        runs.set_trace_info(db, "pp3", 0, "loss", dtype="float", last=0.1)
        runs.set_trace_info(db, "pp3", 0, "my_img", dtype="image")

        result = get_project_params(db, exclude_params=True)
        assert result["params"] == {}
        assert "loss" in result["metric"]
        assert "my_img" in result["images"]
