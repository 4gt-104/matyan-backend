"""Tests for workers/control.py handler functions.

S3 is mocked; FDB is not used by the handlers directly (they only do S3 cleanup).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from matyan_api_models.kafka import ControlEvent

from matyan_backend.workers.control import (
    _HANDLERS,
    ControlWorker,
    _handle_experiment_deleted,
    _handle_run_archived,
    _handle_run_deleted,
    _handle_tag_deleted,
)

# ruff: noqa: SLF001


def _event(event_type: str, **payload: object) -> ControlEvent:
    return ControlEvent(
        type=event_type,
        timestamp=datetime.now(UTC),
        payload=dict(payload),
    )


class TestHandleRunDeleted:
    @patch("matyan_backend.workers.control.blob.delete_blobs")
    def test_with_explicit_blob_keys(self, mock_delete_blobs: MagicMock) -> None:
        db = MagicMock()
        _handle_run_deleted(db, _event("run_deleted", run_id="r1", blob_keys=["r1/a.png", "r1/b.wav"]))
        mock_delete_blobs.assert_called_once_with(["r1/a.png", "r1/b.wav"])

    @patch("matyan_backend.workers.control.blob.delete_blob_prefix")
    def test_with_prefix_listing(self, mock_delete_blob_prefix: MagicMock) -> None:
        db = MagicMock()
        _handle_run_deleted(db, _event("run_deleted", run_id="r2"))
        mock_delete_blob_prefix.assert_called_once_with("r2/")

    @patch("matyan_backend.workers.control.blob.delete_blob_prefix")
    @patch("matyan_backend.workers.control.blob.delete_blobs")
    def test_missing_run_id(self, mock_delete_blobs: MagicMock, mock_delete_blob_prefix: MagicMock) -> None:
        db = MagicMock()
        _handle_run_deleted(db, _event("run_deleted"))
        mock_delete_blobs.assert_not_called()
        mock_delete_blob_prefix.assert_not_called()


class TestHandleNoOpEvents:
    def test_experiment_deleted(self) -> None:
        _handle_experiment_deleted(MagicMock(), _event("experiment_deleted", experiment_id="e1"))

    def test_tag_deleted(self) -> None:
        _handle_tag_deleted(MagicMock(), _event("tag_deleted", tag_id="t1"))

    def test_run_archived(self) -> None:
        _handle_run_archived(MagicMock(), _event("run_archived", run_id="r1"))


class TestControlWorkerDispatch:
    def test_unknown_event_type(self) -> None:
        worker = ControlWorker()
        worker._db = MagicMock()
        worker._handle_event(_event("completely_unknown"))

    def test_dispatches_run_deleted(self) -> None:
        mock_handler = MagicMock()
        original = _HANDLERS["run_deleted"]
        _HANDLERS["run_deleted"] = mock_handler
        try:
            worker = ControlWorker()
            worker._db = MagicMock()
            event = _event("run_deleted", run_id="dispatch1")
            worker._handle_event(event)
            mock_handler.assert_called_once_with(worker._db, event)
        finally:
            _HANDLERS["run_deleted"] = original
