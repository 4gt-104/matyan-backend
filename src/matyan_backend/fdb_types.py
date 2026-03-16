"""Type stubs and typed wrappers for FoundationDB Python bindings.

The ``foundationdb`` package (v7.3.x) ships without ``py.typed`` or ``.pyi``
files, so type checkers cannot resolve its internal types.  This module
re-exports the concrete runtime classes under a single import and provides
``Protocol`` definitions for the handful of duck-typed objects (``Value``,
``KeyValue``) that the storage layer touches.

Usage::

    from __future__ import annotations
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from matyan_backend.fdb_types import Database, DirectorySubspace, Transaction

At runtime these are the real FDB classes; the protocols below exist only to
give type checkers something to work with when the runtime classes have no
stubs.
"""

from __future__ import annotations

import functools
import random
import time
from typing import TYPE_CHECKING, Any, Concatenate, Protocol, runtime_checkable

import fdb as _fdb

# ---------------------------------------------------------------------------
# Re-exports of concrete runtime types (usable in TYPE_CHECKING blocks)
# ---------------------------------------------------------------------------
# These imports will resolve at type-check time because the fdb package IS
# installed — the problem is that it lacks stubs, not that it's missing.
from fdb.directory_impl import DirectoryLayer, DirectorySubspace
from fdb.impl import Database, FDBError, Transaction
from loguru import logger

from matyan_backend.config import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# ---------------------------------------------------------------------------
# FDB retry helpers
# ---------------------------------------------------------------------------

FDB_TRANSACTION_TOO_LARGE: int = 2101

RETRYABLE_FDB_CODES: frozenset[int] = frozenset(
    {
        1004,  # timed_out
        1007,  # transaction_too_old
        1020,  # not_committed
        1021,  # commit_unknown_result
        1031,  # transaction_timed_out
        1037,  # process_behind
        1051,  # batch_transaction_throttled
        1213,  # tag_throttled
    },
)


def is_retryable_fdb_error(exc: BaseException) -> bool:
    """Return True if *exc* is an FDBError with a retryable error code."""
    return isinstance(exc, FDBError) and exc.code in RETRYABLE_FDB_CODES


def run_with_retry[R](
    fn: Callable[[], R],
    *,
    max_attempts: int = 5,
    initial_delay: float = 0.05,
    max_delay: float = 2.0,
    on_retry: Callable[[], None] | None = None,
) -> R:
    """Call *fn()* with exponential-backoff retry on retryable ``FDBError``.

    Non-retryable ``FDBError`` and any other exception are re-raised
    immediately.  After *max_attempts* consecutive retryable failures the
    last error is re-raised.

    *on_retry* is called (if provided) each time a retryable error triggers
    a retry, before sleeping.  Use it to increment a Prometheus counter.
    """
    last_err: FDBError | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except FDBError as exc:
            if not is_retryable_fdb_error(exc):
                raise
            last_err = exc
            if attempt + 1 < max_attempts:
                if on_retry is not None:
                    on_retry()
                jitter = random.uniform(0, initial_delay * 0.25)
                delay = min(initial_delay * (2**attempt) + jitter, max_delay)
                logger.warning(
                    "Retryable FDB error {} (code {}), attempt {}/{}, sleeping {:.3f}s",
                    exc,
                    exc.code,
                    attempt + 1,
                    max_attempts,
                    delay,
                )
                time.sleep(delay)
    raise last_err


# ---------------------------------------------------------------------------
# Typed wrapper for fdb.transactional (with retry)
# ---------------------------------------------------------------------------


def transactional[**P, R](
    func: Callable[Concatenate[Transaction, P], R],
) -> Callable[Concatenate[Database | Transaction, P], R]:
    """Type-preserving wrapper around ``fdb.transactional`` with retry.

    The real decorator replaces the first ``Transaction`` parameter so that
    callers may pass a ``Database`` (the decorator creates and commits the
    transaction automatically) or a ``Transaction`` (passed through as-is).

    When the first argument is a ``Database``, the call is wrapped in
    :func:`run_with_retry` so transient FDB errors are retried with
    exponential backoff.  When a ``Transaction`` is passed (nested call),
    no retry is applied — the caller owns the transaction lifecycle.
    """
    inner = _fdb.transactional(func)

    @functools.wraps(func)
    def wrapper(db_or_tr: Database | Transaction, *args: P.args, **kwargs: P.kwargs) -> R:
        if isinstance(db_or_tr, Database):
            return run_with_retry(
                lambda: inner(db_or_tr, *args, **kwargs),
                max_attempts=SETTINGS.fdb_retry_max_attempts,
                initial_delay=SETTINGS.fdb_retry_initial_delay_sec,
                max_delay=SETTINGS.fdb_retry_max_delay_sec,
            )
        return inner(db_or_tr, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Protocols for duck-typed FDB objects
# ---------------------------------------------------------------------------


@runtime_checkable
class Value(Protocol):
    """Result of ``tr[key]`` — a lazy future that resolves to bytes."""

    def present(self) -> bool: ...
    def __bytes__(self) -> bytes: ...


@runtime_checkable
class KeyValue(Protocol):
    """Element yielded by ``tr.get_range()``."""

    key: bytes
    value: bytes


class FDBRange(Protocol):
    """Return type of ``subspace.range()`` — a ``slice``-like object."""

    start: bytes
    stop: bytes


class Subspace(Protocol):
    """Minimal protocol for FDB Subspace / DirectorySubspace key operations."""

    def pack(self, t: tuple = ()) -> bytes: ...
    def unpack(self, key: bytes) -> tuple: ...
    def range(self, t: tuple = ()) -> FDBRange: ...
    def key(self) -> bytes: ...
    def contains(self, key: bytes) -> bool: ...


class TransactionRead(Protocol):
    """Read-side of an FDB transaction (both Transaction and snapshot)."""

    def get(self, key: bytes) -> Value: ...
    def get_range(
        self,
        begin: bytes,
        end: bytes,
        limit: int = 0,
        reverse: bool = False,
        streaming_mode: int = -1,
    ) -> Iterator[KeyValue]: ...
    def __getitem__(self, key: bytes | slice) -> Any: ...  # noqa: ANN401


__all__ = [
    "RETRYABLE_FDB_CODES",
    "Database",
    "DirectoryLayer",
    "DirectorySubspace",
    "FDBError",
    "FDBRange",
    "KeyValue",
    "Subspace",
    "Transaction",
    "TransactionRead",
    "Value",
    "is_retryable_fdb_error",
    "run_with_retry",
    "transactional",
]
