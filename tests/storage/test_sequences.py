"""Tests for sequences.py — write/read/sample metric sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from matyan_backend.storage import sequences
from matyan_backend.storage.runs import create_run

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


class TestWriteRead:
    def test_write_and_read_single_step(self, db: Database) -> None:
        create_run(db, "seq1")
        sequences.write_sequence_step(db, "seq1", 0, "loss", step=0, value=1.5, epoch=0, timestamp=1000.0)

        result = sequences.read_sequence(db, "seq1", 0, "loss", columns=("val", "epoch", "time"))
        assert result["steps"] == [0]
        assert result["val"] == [1.5]
        assert result["epoch"] == [0]
        assert result["time"] == [1000.0]

    def test_write_batch_and_read(self, db: Database) -> None:
        create_run(db, "seq2")
        batch = [
            {"step": 0, "value": 1.0, "epoch": 0, "timestamp": 100.0},
            {"step": 1, "value": 0.8, "epoch": 0, "timestamp": 101.0},
            {"step": 2, "value": 0.6, "epoch": 1, "timestamp": 102.0},
        ]
        sequences.write_sequence_batch(db, "seq2", 0, "loss", batch)

        result = sequences.read_sequence(db, "seq2", 0, "loss")
        assert result["steps"] == [0, 1, 2]
        assert result["val"] == [1.0, 0.8, 0.6]

    def test_read_step_range(self, db: Database) -> None:
        create_run(db, "seq3")
        for i in range(10):
            sequences.write_sequence_step(db, "seq3", 0, "metric", step=i, value=float(i))

        result = sequences.read_sequence(db, "seq3", 0, "metric", start_step=3, end_step=6)
        assert result["steps"] == [3, 4, 5, 6]
        assert result["val"] == [3.0, 4.0, 5.0, 6.0]

    def test_read_empty_sequence(self, db: Database) -> None:
        create_run(db, "seq4")
        result = sequences.read_sequence(db, "seq4", 0, "nonexistent")
        assert result["steps"] == []
        assert result["val"] == []


class TestSample:
    def test_sample_fewer_than_available(self, db: Database) -> None:
        create_run(db, "samp1")
        for i in range(100):
            sequences.write_sequence_step(db, "samp1", 0, "loss", step=i, value=float(i))

        result = sequences.sample_sequence(db, "samp1", 0, "loss", num_points=5)
        assert len(result["steps"]) == 5
        assert result["steps"][0] == 0
        assert result["steps"][-1] == 99

    def test_sample_more_than_available(self, db: Database) -> None:
        create_run(db, "samp2")
        for i in range(3):
            sequences.write_sequence_step(db, "samp2", 0, "loss", step=i, value=float(i))

        result = sequences.sample_sequence(db, "samp2", 0, "loss", num_points=10)
        assert len(result["steps"]) == 3

    def test_sample_empty(self, db: Database) -> None:
        create_run(db, "samp3")
        result = sequences.sample_sequence(db, "samp3", 0, "loss", num_points=5)
        assert result["steps"] == []

    def test_sample_with_extra_columns(self, db: Database) -> None:
        create_run(db, "samp4")
        for i in range(10):
            sequences.write_sequence_step(db, "samp4", 0, "m", step=i, value=float(i), epoch=i // 5)

        result = sequences.sample_sequence(db, "samp4", 0, "m", num_points=3, columns=("val", "epoch"))
        assert len(result["steps"]) == 3
        assert len(result["val"]) == 3
        assert len(result["epoch"]) == 3


class TestSampleNonContiguous:
    """Point-read sampling with gaps between step numbers."""

    def test_non_contiguous_steps(self, db: Database) -> None:
        create_run(db, "ncg1")
        actual_steps = [0, 10, 50, 100]
        for s in actual_steps:
            sequences.write_sequence_step(db, "ncg1", 0, "m", step=s, value=float(s))

        result = sequences.sample_sequence(db, "ncg1", 0, "m", num_points=3)
        assert len(result["steps"]) == 3
        assert result["steps"][0] == 0
        assert result["steps"][-1] == 100
        for s in result["steps"]:
            assert s in actual_steps

    def test_non_contiguous_all_returned_when_few(self, db: Database) -> None:
        create_run(db, "ncg2")
        actual_steps = [0, 50, 200, 999]
        for s in actual_steps:
            sequences.write_sequence_step(db, "ncg2", 0, "m", step=s, value=float(s))

        result = sequences.sample_sequence(db, "ncg2", 0, "m", num_points=10)
        assert sorted(result["steps"]) == actual_steps

    def test_single_step_sequence(self, db: Database) -> None:
        create_run(db, "ncg3")
        sequences.write_sequence_step(db, "ncg3", 0, "m", step=42, value=3.14)

        result = sequences.sample_sequence(db, "ncg3", 0, "m", num_points=5)
        assert result["steps"] == [42]
        assert result["val"] == [3.14]

    def test_num_points_one(self, db: Database) -> None:
        create_run(db, "ncg4")
        for i in range(20):
            sequences.write_sequence_step(db, "ncg4", 0, "m", step=i, value=float(i))

        result = sequences.sample_sequence(db, "ncg4", 0, "m", num_points=1)
        assert len(result["steps"]) == 1
        assert result["steps"][0] == 0

    def test_sample_includes_endpoints(self, db: Database) -> None:
        create_run(db, "ncg5")
        for i in range(1000):
            sequences.write_sequence_step(db, "ncg5", 0, "m", step=i, value=float(i))

        result = sequences.sample_sequence(db, "ncg5", 0, "m", num_points=50)
        assert result["steps"][0] == 0
        assert result["steps"][-1] == 999
        assert len(result["steps"]) == 50

    def test_sample_values_match_steps(self, db: Database) -> None:
        create_run(db, "ncg6")
        for s in [0, 3, 7, 15, 31, 63]:
            sequences.write_sequence_step(db, "ncg6", 0, "m", step=s, value=float(s * 10))

        result = sequences.sample_sequence(db, "ncg6", 0, "m", num_points=4)
        for step, val in zip(result["steps"], result["val"], strict=True):
            assert val == float(step * 10)


class TestSampleSequencesBatch:
    """Tests for the batched multi-sequence sampling function."""

    def test_batch_same_as_single(self, db: Database) -> None:
        create_run(db, "bsb1")
        for i in range(20):
            sequences.write_sequence_step(
                db,
                "bsb1",
                0,
                "loss",
                step=i,
                value=float(i),
                epoch=i // 5,
                timestamp=1000.0 + i,
            )
            sequences.write_sequence_step(
                db,
                "bsb1",
                0,
                "acc",
                step=i,
                value=float(i * 2),
                epoch=i // 5,
                timestamp=1000.0 + i,
            )

        main, x_axis = sequences.sample_sequences_batch(
            db,
            "bsb1",
            [(0, "loss"), (0, "acc")],
            num_points=5,
        )
        assert x_axis is None
        assert (0, "loss") in main
        assert (0, "acc") in main

        single_loss = sequences.sample_sequence(db, "bsb1", 0, "loss", 5, columns=("val", "epoch", "time"))
        single_acc = sequences.sample_sequence(db, "bsb1", 0, "acc", 5, columns=("val", "epoch", "time"))

        assert main[(0, "loss")]["steps"] == single_loss["steps"]
        assert main[(0, "loss")]["val"] == single_loss["val"]
        assert main[(0, "acc")]["steps"] == single_acc["steps"]
        assert main[(0, "acc")]["val"] == single_acc["val"]

    def test_batch_with_x_axis(self, db: Database) -> None:
        create_run(db, "bsb2")
        for i in range(15):
            sequences.write_sequence_step(db, "bsb2", 0, "loss", step=i, value=float(i))
            sequences.write_sequence_step(db, "bsb2", 0, "x_metric", step=i, value=float(i * 3))

        main, x_axis = sequences.sample_sequences_batch(
            db,
            "bsb2",
            [(0, "loss")],
            num_points=5,
            x_axis_name="x_metric",
        )
        assert (0, "loss") in main
        assert len(main[(0, "loss")]["steps"]) == 5
        assert x_axis is not None
        assert 0 in x_axis
        assert len(x_axis[0]["steps"]) == 5

    def test_batch_empty_requests(self, db: Database) -> None:
        create_run(db, "bsb3")
        main, x_axis = sequences.sample_sequences_batch(
            db,
            "bsb3",
            [],
            num_points=5,
        )
        assert main == {}
        assert x_axis is None

    def test_batch_dedup(self, db: Database) -> None:
        create_run(db, "bsb4")
        for i in range(10):
            sequences.write_sequence_step(db, "bsb4", 0, "loss", step=i, value=float(i))

        main, _ = sequences.sample_sequences_batch(
            db,
            "bsb4",
            [(0, "loss"), (0, "loss"), (0, "loss")],
            num_points=3,
        )
        assert len(main) == 1
        assert (0, "loss") in main

    def test_batch_multiple_contexts(self, db: Database) -> None:
        create_run(db, "bsb5")
        for i in range(10):
            sequences.write_sequence_step(db, "bsb5", 0, "loss", step=i, value=float(i))
            sequences.write_sequence_step(db, "bsb5", 1, "loss", step=i, value=float(i * 10))

        main, _ = sequences.sample_sequences_batch(
            db,
            "bsb5",
            [(0, "loss"), (1, "loss")],
            num_points=3,
        )
        assert (0, "loss") in main
        assert (1, "loss") in main
        assert main[(0, "loss")]["val"] != main[(1, "loss")]["val"]

    def test_batch_x_axis_dedup_contexts(self, db: Database) -> None:
        """x_axis is sampled once per distinct ctx_id, not once per trace."""
        create_run(db, "bsb6")
        for i in range(10):
            sequences.write_sequence_step(db, "bsb6", 0, "loss", step=i, value=float(i))
            sequences.write_sequence_step(db, "bsb6", 0, "acc", step=i, value=float(i * 2))
            sequences.write_sequence_step(db, "bsb6", 0, "x_m", step=i, value=float(i * 5))

        main, x_axis = sequences.sample_sequences_batch(
            db,
            "bsb6",
            [(0, "loss"), (0, "acc")],
            num_points=3,
            x_axis_name="x_m",
        )
        assert len(main) == 2
        assert x_axis is not None
        assert len(x_axis) == 1
        assert 0 in x_axis

    def test_batch_uses_stream_scan_for_large_num_points(self, db: Database) -> None:
        create_run(db, "bsb7")
        for i in range(1000):
            sequences.write_sequence_step(
                db,
                "bsb7",
                0,
                "loss",
                step=i,
                value=float(i),
                epoch=i // 10,
                timestamp=1000.0 + i,
            )

        sentinel = {
            "steps": [0, 500, 999],
            "val": [0.0, 500.0, 999.0],
            "epoch": [0, 50, 99],
            "time": [1000.0, 1500.0, 1999.0],
        }

        with patch(
            "matyan_backend.storage.sequences._stream_scan_sample",
            return_value=sentinel,
        ) as mock_stream_scan:
            main, x_axis = sequences.sample_sequences_batch(
                db,
                "bsb7",
                [(0, "loss")],
                num_points=500,
            )

        assert x_axis is None
        assert main[(0, "loss")] == sentinel
        assert mock_stream_scan.call_count == 1

    def test_batch_uses_stream_scan_for_x_axis_too(self, db: Database) -> None:
        create_run(db, "bsb8")
        for i in range(1000):
            sequences.write_sequence_step(db, "bsb8", 0, "loss", step=i, value=float(i))
            sequences.write_sequence_step(db, "bsb8", 0, "acc", step=i, value=float(i * 2))

        sentinel = {"steps": [0, 500, 999], "val": [0.0, 500.0, 999.0]}

        with patch(
            "matyan_backend.storage.sequences._stream_scan_sample",
            return_value=sentinel,
        ) as mock_stream_scan:
            main, x_axis = sequences.sample_sequences_batch(
                db,
                "bsb8",
                [(0, "loss")],
                num_points=500,
                x_axis_name="acc",
            )

        assert main[(0, "loss")] == sentinel
        assert x_axis == {0: sentinel}
        assert mock_stream_scan.call_count == 2


class TestSequenceLastStep:
    def test_last_step(self, db: Database) -> None:
        create_run(db, "ls1")
        for i in range(7):
            sequences.write_sequence_step(db, "ls1", 0, "loss", step=i, value=0.0)

        assert sequences.get_sequence_last_step(db, "ls1", 0, "loss") == 6

    def test_last_step_empty(self, db: Database) -> None:
        create_run(db, "ls2")
        assert sequences.get_sequence_last_step(db, "ls2", 0, "loss") is None

    def test_last_step_non_contiguous(self, db: Database) -> None:
        create_run(db, "ls3")
        for step in (0, 5, 10, 20):
            sequences.write_sequence_step(db, "ls3", 0, "loss", step=step, value=0.0)

        assert sequences.get_sequence_last_step(db, "ls3", 0, "loss") == 20


class TestSequenceLength:
    def test_length(self, db: Database) -> None:
        create_run(db, "len1")
        for i in range(7):
            sequences.write_sequence_step(db, "len1", 0, "loss", step=i, value=0.0)

        assert sequences.get_sequence_length(db, "len1", 0, "loss") == 7

    def test_length_empty(self, db: Database) -> None:
        create_run(db, "len2")
        assert sequences.get_sequence_length(db, "len2", 0, "loss") == 0
