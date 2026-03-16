"""Shared fixtures for backup tests.

Requires a running FDB instance (docker compose up -d) for tests that use
the ``db`` fixture. Manifest-only tests can run without FDB.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from fdb.directory_impl import directory as _fdb_directory

from matyan_backend.storage import fdb_client

if TYPE_CHECKING:
    from collections.abc import Generator

    from matyan_backend.fdb_types import Database
    from matyan_backend.storage.fdb_client import Directories

_fdb_initialized = False


def _ensure_fdb() -> None:
    global _fdb_initialized  # noqa: PLW0603
    if not _fdb_initialized:
        fdb_client.init_fdb()
        _fdb_initialized = True


@pytest.fixture
def fdb_dirs(monkeypatch: pytest.MonkeyPatch) -> Generator[Directories, None, None]:
    """Give the test its own FDB directories. Only used by tests needing FDB."""
    _ensure_fdb()
    db = fdb_client.get_db()
    test_id = uuid.uuid4().hex[:12]
    prefix = ("_test", test_id)

    runs = _fdb_directory.create_or_open(db, (*prefix, "data", "runs"))
    indexes = _fdb_directory.create_or_open(db, (*prefix, "data", "indexes"))
    system = _fdb_directory.create_or_open(db, (*prefix, "system"))
    dirs = fdb_client.Directories(runs=runs, indexes=indexes, system=system)

    monkeypatch.setattr(fdb_client, "_directories", dirs)

    yield dirs

    if _fdb_directory.exists(db, prefix):
        _fdb_directory.remove(db, prefix)


@pytest.fixture
def db(fdb_dirs: Directories) -> Database:  # noqa: ARG001
    """Return the FDB database handle (implicitly initializes isolated dirs)."""
    return fdb_client.get_db()
