"""Tests for workers/control.py handler functions.

S3 is mocked; FDB is not used by the handlers directly (they only do S3 cleanup).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from matyan_api_models.kafka import ControlEvent

from matyan_backend.workers.control import (
    _HANDLERS,
    ControlWorker,
    _handle_experiment_deleted,
    _handle_run_archived,
    _handle_run_deleted,
    _handle_tag_deleted,
    delete_s3_prefix,
)

# ruff: noqa: SLF001


def _event(event_type: str, **payload: object) -> ControlEvent:
    return ControlEvent(
        type=event_type,
        timestamp=datetime.now(UTC),
        payload=dict(payload),
    )


class TestHandleRunDeleted:
    def test_with_explicit_blob_keys(self) -> None:
        s3 = MagicMock()
        s3.delete_objects.return_value = {"Errors": []}
        db = MagicMock()

        _handle_run_deleted(
            db,
            s3,
            _event("run_deleted", run_id="r1", blob_keys=["r1/a.png", "r1/b.wav"]),
        )
        s3.delete_objects.assert_called_once()
        call_args = s3.delete_objects.call_args
        objects = call_args[1]["Delete"]["Objects"]
        assert len(objects) == 2

    def test_with_prefix_listing(self) -> None:
        s3 = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "r2/file1.pt"}, {"Key": "r2/file2.pt"}]},
        ]
        s3.get_paginator.return_value = paginator
        s3.delete_objects.return_value = {"Errors": []}
        db = MagicMock()

        _handle_run_deleted(db, s3, _event("run_deleted", run_id="r2"))
        s3.delete_objects.assert_called_once()

    def test_empty_prefix_listing(self) -> None:
        s3 = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        s3.get_paginator.return_value = paginator
        db = MagicMock()

        _handle_run_deleted(db, s3, _event("run_deleted", run_id="r3"))
        s3.delete_objects.assert_not_called()

    def test_missing_run_id(self) -> None:
        s3 = MagicMock()
        db = MagicMock()
        _handle_run_deleted(db, s3, _event("run_deleted"))
        s3.delete_objects.assert_not_called()


class TestHandleNoOpEvents:
    def test_experiment_deleted(self) -> None:
        _handle_experiment_deleted(MagicMock(), MagicMock(), _event("experiment_deleted", experiment_id="e1"))

    def test_tag_deleted(self) -> None:
        _handle_tag_deleted(MagicMock(), MagicMock(), _event("tag_deleted", tag_id="t1"))

    def test_run_archived(self) -> None:
        _handle_run_archived(MagicMock(), MagicMock(), _event("run_archived", run_id="r1"))


class TestDeleteS3Prefix:
    def test_deletes_objects_under_prefix(self) -> None:
        s3 = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "run1/a.pt"}, {"Key": "run1/b.pt"}]},
            {"Contents": [{"Key": "run1/c.pt"}]},
        ]
        s3.get_paginator.return_value = paginator
        s3.delete_objects.return_value = {"Errors": []}

        count = delete_s3_prefix(s3, "my-bucket", "run1/")
        assert count == 3
        assert s3.delete_objects.call_count == 2

    def test_empty_prefix(self) -> None:
        s3 = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        s3.get_paginator.return_value = paginator

        count = delete_s3_prefix(s3, "my-bucket", "empty_run/")
        assert count == 0
        s3.delete_objects.assert_not_called()

    def test_no_pages(self) -> None:
        s3 = MagicMock()
        paginator = MagicMock()
        paginator.paginate.return_value = []
        s3.get_paginator.return_value = paginator

        count = delete_s3_prefix(s3, "my-bucket", "missing/")
        assert count == 0


class TestControlWorkerDispatch:
    def test_unknown_event_type(self) -> None:
        worker = ControlWorker()
        worker._db = MagicMock()
        worker._s3 = MagicMock()
        worker._handle_event(_event("completely_unknown"))

    def test_dispatches_run_deleted(self) -> None:
        mock_handler = MagicMock()
        original = _HANDLERS["run_deleted"]
        _HANDLERS["run_deleted"] = mock_handler
        try:
            worker = ControlWorker()
            worker._db = MagicMock()
            worker._s3 = MagicMock()
            event = _event("run_deleted", run_id="dispatch1")
            worker._handle_event(event)
            mock_handler.assert_called_once_with(worker._db, worker._s3, event)
        finally:
            _HANDLERS["run_deleted"] = original
