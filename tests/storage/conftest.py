"""Shared fixtures for storage tests.

Requires a running FDB instance (docker compose up -d).
Each test gets isolated via a unique directory prefix that is cleaned up after.
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


@pytest.fixture(scope="session", autouse=True)
def _init_fdb() -> None:
    """Initialize the FDB connection once for the entire test session."""
    fdb_client.init_fdb()


@pytest.fixture(autouse=True)
def _isolated_directories(monkeypatch: pytest.MonkeyPatch) -> Generator[Directories, None, None]:
    """Give each test its own FDB directories so tests don't interfere.

    Creates a unique directory prefix per test, patches ``get_directories``
    to return it, and cleans up after the test.
    """
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
def db() -> Database:
    """Return the FDB database handle."""
    return fdb_client.get_db()
