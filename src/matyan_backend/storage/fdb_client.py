from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import fdb
from fdb.directory_impl import directory as _fdb_directory

from matyan_backend.config import SETTINGS

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database, DirectorySubspace, Transaction

# Must be called before any @transactional decorator is evaluated at
# import time in downstream modules (runs.py, sequences.py, entities.py).
fdb.api_version(SETTINGS.fdb_api_version)

_db: Database | None = None
_directories: Directories | None = None


class Directories(NamedTuple):
    runs: DirectorySubspace
    indexes: DirectorySubspace
    system: DirectorySubspace


def init_fdb(cluster_file: str | None = None) -> Database:
    """Open the FDB connection. Must be called once at application startup."""
    global _db  # noqa: PLW0603
    _db = fdb.open(cluster_file or SETTINGS.fdb_cluster_file)
    return _db


def get_db() -> Database:
    if _db is None:
        msg = "FDB not initialized. Call init_fdb() first."
        raise RuntimeError(msg)
    return _db


def ensure_directories(db: Database | None = None) -> Directories:
    """Create or open the top-level FDB directories. Caches the result.

    Each ``create_or_open`` runs in its own transaction internally.
    Directory creation is idempotent, so no single-transaction guarantee needed.
    """
    global _directories  # noqa: PLW0603
    target = db or get_db()
    runs = _fdb_directory.create_or_open(target, ("data", "runs"))
    indexes = _fdb_directory.create_or_open(target, ("data", "indexes"))
    system = _fdb_directory.create_or_open(target, ("system",))
    _directories = Directories(runs=runs, indexes=indexes, system=system)
    return _directories


def get_directories() -> Directories:
    if _directories is None:
        msg = "Directories not initialized. Call ensure_directories() first."
        raise RuntimeError(msg)
    return _directories


def ping(db: Database | None = None) -> bool:
    """Run a minimal FDB read transaction to verify connectivity.

    Returns ``True`` on success, raises on failure (timeout, network error, etc.).
    """
    target = db or get_db()
    dirs = get_directories()

    @fdb.transactional
    def _read(tr: Transaction) -> None:
        tr[dirs.system.pack(("__ping__",))]  # type: ignore[index]

    _read(target)
    return True
