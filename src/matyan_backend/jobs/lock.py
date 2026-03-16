"""FDB-based distributed lock for periodic maintenance jobs.

Uses a single FDB key per job name with an expiry timestamp as value.
The check-and-set runs in one FDB transaction for atomicity.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import fdb

from matyan_backend.storage import encoding
from matyan_backend.storage.fdb_client import get_directories

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


def try_acquire(db: Database, job_name: str, ttl_seconds: int) -> bool:
    """Attempt to acquire the lock for *job_name*.

    Returns ``True`` if the lock was acquired (key absent or expired).
    Returns ``False`` if another holder still owns the lock.
    """

    @fdb.transactional
    def _txn(tr: object) -> bool:
        sys_dir = get_directories().system
        key = sys_dir.pack(("job_lock", job_name))
        raw = tr[key]  # type: ignore[index]
        if raw.present():
            try:
                expiry = encoding.decode_value(bytes(raw))
                if isinstance(expiry, (int, float)) and expiry > time.time():
                    return False
            except Exception:  # noqa: BLE001, S110
                pass
        tr[key] = encoding.encode_value(time.time() + ttl_seconds)  # type: ignore[index]
        return True

    return _txn(db)


def release(db: Database, job_name: str) -> None:
    """Release the lock for *job_name*. Idempotent — missing key is a no-op."""

    @fdb.transactional
    def _txn(tr: object) -> None:
        sys_dir = get_directories().system
        key = sys_dir.pack(("job_lock", job_name))
        del tr[key]  # type: ignore[arg-type]

    _txn(db)
