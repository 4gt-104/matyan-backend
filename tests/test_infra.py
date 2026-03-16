"""Infrastructure edge case tests — s3_client, fdb_client, deps, collections filtering."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from matyan_backend.deps import fdb_db, fdb_dirs, kafka_producer
from matyan_backend.storage import fdb_client
from matyan_backend.storage.s3_client import get_blob

# ruff: noqa: SLF001


class TestS3Client:
    @patch("matyan_backend.storage.s3_client._client", None)
    @patch("matyan_backend.storage.s3_client.boto3")
    def test_get_blob(self, mock_boto3: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.get_object.return_value = {"Body": MagicMock(read=lambda: b"blob-data")}
        mock_boto3.client.return_value = mock_client

        data = get_blob("test-key")
        assert data == b"blob-data"
        mock_client.get_object.assert_called_once()


class TestFdbClient:
    def test_get_db_before_init_raises(self) -> None:
        original = fdb_client._db
        try:
            fdb_client._db = None
            with pytest.raises(RuntimeError, match="FDB not initialized"):
                fdb_client.get_db()
        finally:
            fdb_client._db = original

    def test_get_directories_before_init_raises(self) -> None:
        original = fdb_client._directories
        try:
            fdb_client._directories = None
            with pytest.raises(RuntimeError, match="Directories not initialized"):
                fdb_client.get_directories()
        finally:
            fdb_client._directories = original

    def test_init_fdb_idempotent(self) -> None:
        db1 = fdb_client.init_fdb()
        db2 = fdb_client.init_fdb()
        assert db1 is not None
        assert db2 is not None

    def test_ensure_directories(self) -> None:
        fdb_client.init_fdb()
        dirs = fdb_client.ensure_directories()
        assert dirs.runs is not None
        assert dirs.indexes is not None
        assert dirs.system is not None


class TestDeps:
    def test_fdb_db(self) -> None:
        fdb_client.init_fdb()
        fdb_client.ensure_directories()
        db = fdb_db()
        assert db is not None

    def test_fdb_dirs(self) -> None:
        fdb_client.init_fdb()
        fdb_client.ensure_directories()
        dirs = fdb_dirs()
        assert dirs.runs is not None

    def test_kafka_producer(self) -> None:
        p = kafka_producer()
        assert p is not None
