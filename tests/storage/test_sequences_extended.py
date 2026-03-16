"""Extended tests for storage/sequences.py — edge cases and composite operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matyan_backend.storage import runs, sequences

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


class TestReadAndSampleSequence:
    def test_basic_downsample(self, db: Database) -> None:
        runs.create_run(db, "ras1")
        for i in range(20):
            sequences.write_sequence_step(db, "ras1", 0, "loss", i, float(i))
        result = sequences.read_and_sample_sequence(db, "ras1", 0, "loss", density=5)
        assert len(result["steps"]) == 5

    def test_no_downsample_when_density_exceeds_data(self, db: Database) -> None:
        runs.create_run(db, "ras2")
        for i in range(3):
            sequences.write_sequence_step(db, "ras2", 0, "loss", i, float(i))
        result = sequences.read_and_sample_sequence(db, "ras2", 0, "loss", density=100)
        assert len(result["steps"]) == 3

    def test_with_range_and_density(self, db: Database) -> None:
        runs.create_run(db, "ras3")
        for i in range(100):
            sequences.write_sequence_step(db, "ras3", 0, "m", i, float(i))
        result = sequences.read_and_sample_sequence(db, "ras3", 0, "m", start_step=10, end_step=50, density=5)
        assert len(result["steps"]) == 5
        assert all(10 <= s <= 50 for s in result["steps"])

    def test_density_none_returns_all(self, db: Database) -> None:
        runs.create_run(db, "ras4")
        for i in range(10):
            sequences.write_sequence_step(db, "ras4", 0, "m", i, float(i))
        result = sequences.read_and_sample_sequence(db, "ras4", 0, "m", density=None)
        assert len(result["steps"]) == 10

    def test_density_zero_returns_all(self, db: Database) -> None:
        runs.create_run(db, "ras5")
        for i in range(10):
            sequences.write_sequence_step(db, "ras5", 0, "m", i, float(i))
        result = sequences.read_and_sample_sequence(db, "ras5", 0, "m", density=0)
        assert len(result["steps"]) == 10


class TestReadAndSampleNonContiguous:
    """read_and_sample_sequence with non-contiguous steps and density."""

    def test_non_contiguous_with_density(self, db: Database) -> None:
        runs.create_run(db, "rasnc1")
        actual_steps = [0, 10, 50, 100, 200, 500, 999]
        for s in actual_steps:
            sequences.write_sequence_step(db, "rasnc1", 0, "m", s, float(s))

        result = sequences.read_and_sample_sequence(db, "rasnc1", 0, "m", density=4)
        assert len(result["steps"]) == 4
        assert result["steps"][0] == 0
        assert result["steps"][-1] == 999
        for s in result["steps"]:
            assert s in actual_steps

    def test_single_step_with_density(self, db: Database) -> None:
        runs.create_run(db, "rasnc2")
        sequences.write_sequence_step(db, "rasnc2", 0, "m", 77, 1.5)

        result = sequences.read_and_sample_sequence(db, "rasnc2", 0, "m", density=10)
        assert result["steps"] == [77]
        assert result["val"] == [1.5]

    def test_range_with_density_non_contiguous(self, db: Database) -> None:
        runs.create_run(db, "rasnc3")
        for s in range(0, 200, 7):
            sequences.write_sequence_step(db, "rasnc3", 0, "m", s, float(s))

        result = sequences.read_and_sample_sequence(
            db, "rasnc3", 0, "m", start_step=20, end_step=150, density=5,
        )
        assert len(result["steps"]) == 5
        assert all(20 <= s <= 150 for s in result["steps"])

    def test_density_exceeds_available_non_contiguous(self, db: Database) -> None:
        runs.create_run(db, "rasnc4")
        actual_steps = [5, 50, 500]
        for s in actual_steps:
            sequences.write_sequence_step(db, "rasnc4", 0, "m", s, float(s))

        result = sequences.read_and_sample_sequence(db, "rasnc4", 0, "m", density=100)
        assert sorted(result["steps"]) == actual_steps

    def test_empty_with_density(self, db: Database) -> None:
        runs.create_run(db, "rasnc5")
        result = sequences.read_and_sample_sequence(db, "rasnc5", 0, "m", density=10)
        assert result["steps"] == []
        assert result["val"] == []


class TestGetSequenceStepBounds:
    def test_empty_sequence(self, db: Database) -> None:
        runs.create_run(db, "bounds1")
        first, last = sequences.get_sequence_step_bounds(db, "bounds1", 0, "missing")
        assert first is None
        assert last is None

    def test_single_step(self, db: Database) -> None:
        runs.create_run(db, "bounds2")
        sequences.write_sequence_step(db, "bounds2", 0, "m", 5, 1.0)
        first, last = sequences.get_sequence_step_bounds(db, "bounds2", 0, "m")
        assert first == 5
        assert last == 5

    def test_multiple_steps(self, db: Database) -> None:
        runs.create_run(db, "bounds3")
        for i in [3, 7, 15]:
            sequences.write_sequence_step(db, "bounds3", 0, "m", i, float(i))
        first, last = sequences.get_sequence_step_bounds(db, "bounds3", 0, "m")
        assert first == 3
        assert last == 15


class TestWriteSequenceBatch:
    def test_batch_write_and_read(self, db: Database) -> None:
        runs.create_run(db, "batch1")
        steps_data = [
            {"step": 0, "value": 1.0, "epoch": 0, "timestamp": 100.0},
            {"step": 1, "value": 2.0, "epoch": 0, "timestamp": 101.0},
            {"step": 2, "value": 3.0},
        ]
        sequences.write_sequence_batch(db, "batch1", 0, "loss", steps_data)
        result = sequences.read_sequence(db, "batch1", 0, "loss", columns=("val", "epoch", "time"))
        assert result["steps"] == [0, 1, 2]
        assert result["val"] == [1.0, 2.0, 3.0]
        assert result["epoch"][0] == 0
        assert result["time"][0] == 100.0
        assert result["epoch"][2] is None
