"""Tests for jobs/lock.py — FDB-based distributed lock."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import fdb

from matyan_backend.jobs.lock import release, try_acquire
from matyan_backend.storage.fdb_client import get_directories

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


class TestTryAcquire:
    def test_acquire_when_no_key(self, db: Database) -> None:
        assert try_acquire(db, "test_job_1", ttl_seconds=60) is True

    def test_cannot_acquire_when_held(self, db: Database) -> None:
        assert try_acquire(db, "test_job_2", ttl_seconds=60) is True
        assert try_acquire(db, "test_job_2", ttl_seconds=60) is False

    def test_acquire_after_expiry(self, db: Database) -> None:
        assert try_acquire(db, "test_job_3", ttl_seconds=1) is True
        with patch("matyan_backend.jobs.lock.time") as mock_time:
            mock_time.time.return_value = time.time() + 10
            assert try_acquire(db, "test_job_3", ttl_seconds=60) is True

    def test_acquire_with_corrupt_value(self, db: Database) -> None:

        sys_dir = get_directories().system

        @fdb.transactional
        def write_corrupt(tr: object) -> None:
            tr[sys_dir.pack(("job_lock", "corrupt_job"))] = b"\xff\xff"  # ty:ignore[invalid-assignment]

        write_corrupt(db)
        assert try_acquire(db, "corrupt_job", ttl_seconds=60) is True


class TestRelease:
    def test_release_existing_lock(self, db: Database) -> None:
        assert try_acquire(db, "rel_job_1", ttl_seconds=60) is True
        release(db, "rel_job_1")
        assert try_acquire(db, "rel_job_1", ttl_seconds=60) is True

    def test_release_nonexistent_key(self, db: Database) -> None:
        release(db, "nonexistent_lock")


class TestConcurrentAccess:
    def test_second_caller_blocked_while_first_holds(self, db: Database) -> None:
        assert try_acquire(db, "concurrent_job", ttl_seconds=300) is True
        assert try_acquire(db, "concurrent_job", ttl_seconds=300) is False
        release(db, "concurrent_job")
        assert try_acquire(db, "concurrent_job", ttl_seconds=300) is True
