"""Tests for query timing instrumentation in _collections.py."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from matyan_backend.api.runs._collections import (
    _TIMING_PREFIX,
    iter_matching_runs,
    iter_matching_sequences,
)
from matyan_backend.api.runs._planner import PlanResult
from matyan_backend.config import SETTINGS
from matyan_backend.storage import runs, sequences

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database


BASE = "/api/v1/rest"


@pytest.fixture
def _enable_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SETTINGS, "query_timing_enabled", True)


@pytest.fixture
def _disable_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SETTINGS, "query_timing_enabled", False)


def _collect_timing_lines(mock_info: Mock) -> list[str]:
    return [
        call.args[0] if call.args else str(call)
        for call in mock_info.call_args_list
        if call.args and isinstance(call.args[0], str) and _TIMING_PREFIX in call.args[0]
    ]


class TestRunSearchTimingEnabled:
    """Timing logs should appear for superset and lazy paths when enabled."""

    @pytest.mark.usefixtures("_enable_timing")
    def test_lazy_path_emits_timing(self, db: Database) -> None:
        runs.create_run(db, "qt_lazy1", name="lazy-run")
        lazy_result = PlanResult(candidates=None, exact=True)
        with (
            patch("matyan_backend.api.runs._collections.plan_query", return_value=lazy_result),
            patch("matyan_backend.api.runs._collections.logger") as mock_logger,
        ):
            list(iter_matching_runs(db, 'run.name == "lazy-run"'))

        lines = _collect_timing_lines(mock_logger.info)
        steps = [ln.split()[1] for ln in lines]
        step_names = [s.split("=", 1)[1] for s in steps]

        assert "plan_query" in step_names
        assert "fetch_meta" in step_names
        assert "build_run_view" in step_names
        assert "check" in step_names
        assert "summary" in step_names
        for ln in lines:
            assert "path=lazy" in ln

    @pytest.mark.usefixtures("_enable_timing")
    def test_superset_path_emits_timing(self, db: Database) -> None:
        runs.create_run(db, "qt_ss1", name="ss-run")
        runs.update_run_meta(db, "qt_ss1", is_archived=False)
        with patch("matyan_backend.api.runs._collections.logger") as mock_logger:
            list(iter_matching_runs(db, 'run.name == "ss-run"'))

        lines = _collect_timing_lines(mock_logger.info)
        step_names = [ln.split()[1].split("=", 1)[1] for ln in lines]

        assert "plan_query" in step_names
        assert "summary" in step_names

    @pytest.mark.usefixtures("_enable_timing")
    def test_summary_has_counts(self, db: Database) -> None:
        runs.create_run(db, "qt_cnt1")
        with patch("matyan_backend.api.runs._collections.logger") as mock_logger:
            list(iter_matching_runs(db, ""))

        lines = _collect_timing_lines(mock_logger.info)
        summary = [ln for ln in lines if "step=summary" in ln]
        assert len(summary) <= 1  # exact path has no summary


class TestSequenceSearchTimingEnabled:
    """Timing logs for iter_matching_sequences."""

    @pytest.mark.usefixtures("_enable_timing")
    def test_emits_all_steps(self, db: Database) -> None:
        runs.create_run(db, "qt_seq1")
        runs.set_context(db, "qt_seq1", 0, {})
        runs.set_trace_info(db, "qt_seq1", 0, "loss", dtype="float", last=0.5)

        with patch("matyan_backend.api.runs._collections.logger") as mock_logger:
            list(iter_matching_sequences(db, "", seq_type="metric"))

        lines = _collect_timing_lines(mock_logger.info)
        step_names = [ln.split()[1].split("=", 1)[1] for ln in lines]

        assert "plan_query" in step_names
        assert "fetch_meta" in step_names
        assert "build_run_view" in step_names
        assert "fetch_traces" in step_names
        assert "fetch_contexts" in step_names
        assert "trace_loop" in step_names
        assert "summary" in step_names

    @pytest.mark.usefixtures("_enable_timing")
    def test_summary_has_trace_counts(self, db: Database) -> None:
        runs.create_run(db, "qt_seq2")
        runs.set_context(db, "qt_seq2", 0, {})
        runs.set_trace_info(db, "qt_seq2", 0, "loss", dtype="float", last=0.5)
        runs.set_trace_info(db, "qt_seq2", 0, "acc", dtype="float", last=0.9)

        with patch("matyan_backend.api.runs._collections.logger") as mock_logger:
            list(iter_matching_sequences(db, "", seq_type="metric"))

        lines = _collect_timing_lines(mock_logger.info)
        summary = [ln for ln in lines if "step=summary" in ln]
        assert len(summary) == 1
        assert "traces_checked=" in summary[0]
        assert "runs_checked=" in summary[0]


class TestMetricCandidateSearchTimingEnabled:
    """Timing logs should appear for the metric search candidate path when enabled."""

    @pytest.mark.usefixtures("_enable_timing")
    def test_candidate_path_emits_timing(self, db: Database, client: TestClient) -> None:
        run_hash = "qt_metric_candidate1"
        runs.create_run(db, run_hash)
        runs.set_context(db, run_hash, 0, {})
        runs.set_trace_info(db, run_hash, 0, "loss", dtype="float", last=0.5, last_step=2)
        for step in range(3):
            sequences.write_sequence_step(
                db,
                run_hash,
                0,
                "loss",
                step,
                0.5 - (step * 0.1),
                epoch=0,
                timestamp=1000.0 + step,
            )

        with patch("matyan_backend.api.runs._streaming._log_timing") as mock_log_timing:
            resp = client.get(
                f"{BASE}/runs/search/metric/",
                params={"q": "", "report_progress": "false"},
            )
            content = resp.content

        assert resp.status_code == 200
        assert len(content) > 0

        step_names = [call.args[0] for call in mock_log_timing.call_args_list]

        assert "plan_query" in step_names
        assert "fetch_bundles" in step_names
        assert "build_run_view_and_check" in step_names
        assert "collect_traces" in step_names
        assert "summary" in step_names

        candidate_calls = [call for call in mock_log_timing.call_args_list if call.kwargs.get("path") == "candidate"]
        assert candidate_calls
        for call in candidate_calls:
            assert call.kwargs.get("endpoint") == "metric_search"

        summary_calls = [call for call in mock_log_timing.call_args_list if call.args[0] == "summary"]
        assert len(summary_calls) == 1
        assert "runs_checked=1" in summary_calls[0].kwargs["extra"]
        assert "runs_matched=1" in summary_calls[0].kwargs["extra"]


class TestTimingDisabled:
    """No timing logs when query_timing_enabled is False."""

    @pytest.mark.usefixtures("_disable_timing")
    def test_run_search_no_timing(self, db: Database) -> None:
        runs.create_run(db, "qt_off1")
        with patch("matyan_backend.api.runs._collections.logger") as mock_logger:
            list(iter_matching_runs(db, ""))

        lines = _collect_timing_lines(mock_logger.info)
        assert len(lines) == 0

    @pytest.mark.usefixtures("_disable_timing")
    def test_sequence_search_no_timing(self, db: Database) -> None:
        runs.create_run(db, "qt_off2")
        runs.set_context(db, "qt_off2", 0, {})
        runs.set_trace_info(db, "qt_off2", 0, "loss", dtype="float", last=0.1)

        with patch("matyan_backend.api.runs._collections.logger") as mock_logger:
            list(iter_matching_sequences(db, "", seq_type="metric"))

        lines = _collect_timing_lines(mock_logger.info)
        assert len(lines) == 0


class TestTimingFormat:
    """Timing lines use the expected machine-friendly format."""

    @pytest.mark.usefixtures("_enable_timing")
    def test_key_value_format(self, db: Database) -> None:
        runs.create_run(db, "qt_fmt1")
        with patch("matyan_backend.api.runs._collections.logger") as mock_logger:
            list(iter_matching_runs(db, 'run.name == "qt_fmt1"'))

        lines = _collect_timing_lines(mock_logger.info)
        for ln in lines:
            assert ln.startswith(_TIMING_PREFIX)
            assert "step=" in ln
            assert "path=" in ln
            assert "duration_sec=" in ln
