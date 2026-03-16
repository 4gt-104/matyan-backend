"""Extended tests for _collections.py — seq_type filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from matyan_backend.api.runs._collections import RunHashRef, iter_matching_runs, iter_matching_sequences
from matyan_backend.storage import runs

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database

# ruff: noqa: SLF001


class TestIterMatchingRuns:
    def test_basic(self, db: Database) -> None:
        runs.create_run(db, "imr1")
        results = list(iter_matching_runs(db, ""))
        assert any(rv.hash == "imr1" for rv, _ in results)

    def test_with_query_filter(self, db: Database) -> None:
        runs.create_run(db, "imr2")
        runs.update_run_meta(db, "imr2", is_archived=True)
        results = list(iter_matching_runs(db, ""))
        hashes = [rv.hash for rv, _ in results]
        assert "imr2" not in hashes

    def test_candidate_path_yields_hash_ref(self, db: Database) -> None:
        """When the planner returns candidates, iterator yields RunHashRef."""
        runs.create_run(db, "imr3")
        results = list(iter_matching_runs(db, ""))
        refs = [rv for rv, _ in results if rv.hash == "imr3"]
        assert len(refs) == 1
        assert isinstance(refs[0], RunHashRef)

    def test_candidate_path_skips_get_run_meta(self, db: Database) -> None:
        """Candidate-list path should not call get_run_meta at all."""
        runs.create_run(db, "imr4")
        with patch(
            "matyan_backend.api.runs._collections.get_run_meta",
            wraps=runs.get_run_meta,
        ) as mock_meta:
            list(iter_matching_runs(db, ""))
            assert mock_meta.call_count == 0


class TestIterMatchingSequences:
    def test_seq_type_filter(self, db: Database) -> None:
        runs.create_run(db, "ims1")
        runs.set_context(db, "ims1", 0, {})
        runs.set_trace_info(db, "ims1", 0, "loss", dtype="float", last=0.1)
        runs.set_trace_info(db, "ims1", 0, "my_images", dtype="image")

        float_results = list(iter_matching_sequences(db, "", seq_type="metric"))
        for _, sv, _ in float_results:
            trace = sv._trace_info
            assert trace.get("dtype") != "image"

    def test_image_seq_type(self, db: Database) -> None:
        runs.create_run(db, "ims2")
        runs.set_context(db, "ims2", 0, {})
        runs.set_trace_info(db, "ims2", 0, "my_images", dtype="image")

        results = list(iter_matching_sequences(db, "", seq_type="images"))
        names = [sv.name for _, sv, _ in results]
        assert "my_images" in names

    def test_text_seq_type(self, db: Database) -> None:
        runs.create_run(db, "ims3")
        runs.set_context(db, "ims3", 0, {})
        runs.set_trace_info(db, "ims3", 0, "my_texts", dtype="text")

        results = list(iter_matching_sequences(db, "", seq_type="texts"))
        names = [sv.name for _, sv, _ in results]
        assert "my_texts" in names

    def test_metric_excludes_logs(self, db: Database) -> None:
        runs.create_run(db, "ims4")
        runs.set_context(db, "ims4", 0, {})
        runs.set_trace_info(db, "ims4", 0, "my_logs", dtype="logs")

        results = list(iter_matching_sequences(db, "", seq_type="metric"))
        names = [sv.name for _, sv, _ in results]
        assert "my_logs" not in names

    def test_distribution_seq_type(self, db: Database) -> None:
        runs.create_run(db, "ims5")
        runs.set_context(db, "ims5", 0, {})
        runs.set_trace_info(db, "ims5", 0, "hist", dtype="distribution")

        results = list(iter_matching_sequences(db, "", seq_type="distributions"))
        names = [sv.name for _, sv, _ in results]
        assert "hist" in names


class TestLazyIteration:
    """Verify that ``iter_matching_runs`` uses lazy iteration so callers
    that break early don't trigger ``get_run_meta`` for every run in the DB.
    """

    def test_lazy_stops_early(self, db: Database) -> None:
        num_runs = 10
        for i in range(num_runs):
            runs.create_run(db, f"lazy{i:03d}")

        consume_count = 2
        with patch(
            "matyan_backend.api.runs._collections.get_run_meta",
            wraps=runs.get_run_meta,
        ) as mock_meta:
            gen = iter_matching_runs(db, "")
            for _ in range(consume_count):
                next(gen)
            del gen

            assert mock_meta.call_count < num_runs
