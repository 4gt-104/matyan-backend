"""Secondary index maintenance and lookup.

All index entries live in ``indexes_dir``.  Key layout::

    Tier 1 (structured fields):
    ("archived",   <bool>,      <run_hash>) -> b""
    ("active",     <bool>,      <run_hash>) -> b""
    ("experiment", <exp_name>,   <run_hash>) -> b""
    ("created_at", <timestamp>,  <run_hash>) -> b""
    ("tag",        <tag_name>,   <run_hash>) -> b""

    Tier 2 (hyperparameters — top-level scalars only):
    ("hparam",     <param_name>, <value>,    <run_hash>) -> b""

    Tier 3 (metric trace names):
    ("trace",      <metric_name>, <run_hash>) -> b""

    Reverse index (for O(1) per-run deindexing):
    ("_rev",       <run_hash>,   <forward_key_elements>...) -> b""

Values are always empty bytes — the run hash embedded in the key *is* the
payload.  Range scans on a prefix like ``("archived", False)`` yield all
non-archived run hashes.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from matyan_backend.fdb_types import transactional

from . import encoding, entities, runs, tree
from .fdb_client import get_directories

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matyan_backend.fdb_types import Database, DirectorySubspace, Transaction

_EMPTY = b""

# Index field name constants — Tier 1
_ARCHIVED = "archived"
_ACTIVE = "active"
_EXPERIMENT = "experiment"
_CREATED_AT = "created_at"
_TAG = "tag"

# Index field name constants — Tier 2
_HPARAM = "hparam"

# Index field name constants — Tier 3
_TRACE = "trace"

# Deletion tombstones — prevent ghost runs from being recreated by late
# ingestion messages after a run has been deleted via the API.
_DELETED = "_deleted"

_INDEXABLE_SCALAR_TYPES = (int, float, str, bool)

# Reverse index prefix for per-run deindexing
_REV = "_rev"


def _idx_dir() -> DirectorySubspace:
    return get_directories().indexes


def _write_fwd_rev(tr: Transaction, idx: DirectorySubspace, forward_key: tuple, run_hash: str) -> None:
    """Write a forward index key and its corresponding reverse entry."""
    tr[idx.pack(forward_key)] = _EMPTY
    tr[idx.pack((_REV, run_hash, *forward_key))] = _EMPTY


def _delete_fwd_rev(tr: Transaction, idx: DirectorySubspace, forward_key: tuple, run_hash: str) -> None:
    """Delete a forward index key and its corresponding reverse entry."""
    del tr[idx.pack(forward_key)]
    rev_key = idx.pack((_REV, run_hash, *forward_key))
    if tr[rev_key].present():
        del tr[rev_key]


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


@transactional
def index_run(
    tr: Transaction,
    run_hash: str,
    *,
    is_archived: bool = False,
    active: bool = True,
    created_at: float = 0.0,
    experiment_name: str | None = None,
    tag_names: list[str] | None = None,
) -> None:
    """Write all Tier 1 index entries for a run (idempotent)."""
    idx = _idx_dir()
    _write_fwd_rev(tr, idx, (_ARCHIVED, is_archived, run_hash), run_hash)
    _write_fwd_rev(tr, idx, (_ACTIVE, active, run_hash), run_hash)
    _write_fwd_rev(tr, idx, (_CREATED_AT, created_at, run_hash), run_hash)
    if experiment_name is not None:
        _write_fwd_rev(tr, idx, (_EXPERIMENT, experiment_name, run_hash), run_hash)
    for tn in tag_names or []:
        _write_fwd_rev(tr, idx, (_TAG, tn, run_hash), run_hash)


@transactional
def deindex_run(tr: Transaction, run_hash: str) -> None:
    """Remove *all* index entries (Tier 1 + Tier 2) for a run.

    Uses the reverse index ``("_rev", run_hash, ...)`` so only entries
    belonging to this run are scanned — O(entries_for_run) instead of
    O(total_entries).
    """
    idx = _idx_dir()
    r = idx.range((_REV, run_hash))
    for kv in tr.get_range(r.start, r.stop):
        rev_tuple = idx.unpack(kv.key)
        forward_key = rev_tuple[2:]
        fwd_packed = idx.pack(forward_key)
        if tr[fwd_packed].present():
            del tr[fwd_packed]
        del tr[kv.key]


@transactional
def update_index_field(
    tr: Transaction,
    run_hash: str,
    field: str,
    old_value: object,
    new_value: object,
) -> None:
    """Atomically swap an index entry: delete old, write new.

    If *new_value* is ``None`` the entry is simply removed (no replacement).
    """
    idx = _idx_dir()
    if old_value is not None:
        _delete_fwd_rev(tr, idx, (field, old_value, run_hash), run_hash)
    if new_value is not None:
        _write_fwd_rev(tr, idx, (field, new_value, run_hash), run_hash)


@transactional
def add_tag_index(tr: Transaction, run_hash: str, tag_name: str) -> None:
    idx = _idx_dir()
    _write_fwd_rev(tr, idx, (_TAG, tag_name, run_hash), run_hash)


@transactional
def remove_tag_index(tr: Transaction, run_hash: str, tag_name: str) -> None:
    idx = _idx_dir()
    _delete_fwd_rev(tr, idx, (_TAG, tag_name, run_hash), run_hash)


@transactional
def remove_all_tag_indexes_for_tag(tr: Transaction, tag_name: str) -> None:
    """Remove all ``("tag", tag_name, *)`` index entries and their reverse entries."""
    idx = _idx_dir()
    r = idx.range((_TAG, tag_name))
    for kv in tr.get_range(r.start, r.stop):
        fwd_tuple = idx.unpack(kv.key)
        run_hash = fwd_tuple[-1]
        rev_key = idx.pack((_REV, run_hash, *fwd_tuple))
        if tr[rev_key].present():
            del tr[rev_key]
    del tr[r.start : r.stop]


# ---------------------------------------------------------------------------
# Deletion tombstones
# ---------------------------------------------------------------------------


@transactional
def mark_run_deleted(tr: Transaction, run_hash: str) -> None:
    """Write a tombstone so that late-arriving ingestion messages are skipped."""
    idx = _idx_dir()

    tr[idx.pack((_DELETED, run_hash))] = encoding.encode_value(time.time())


@transactional
def is_run_deleted(tr: Transaction, run_hash: str) -> bool:
    """Return ``True`` if a deletion tombstone exists for *run_hash*."""
    idx = _idx_dir()
    return tr[idx.pack((_DELETED, run_hash))].present()


@transactional
def clear_run_tombstone(tr: Transaction, run_hash: str) -> None:
    """Remove the deletion tombstone (e.g. for periodic cleanup)."""
    idx = _idx_dir()
    del tr[idx.pack((_DELETED, run_hash))]


@transactional
def list_tombstones(tr: Transaction) -> list[tuple[str, float]]:
    """Return ``(run_hash, deleted_at_timestamp)`` for every tombstone.

    Values are msgpack-encoded floats written by :func:`mark_run_deleted`.
    Corrupt or undecodable values are silently treated as timestamp ``0.0``
    so the caller can still process them.
    """
    idx = _idx_dir()
    r = idx.range((_DELETED,))
    result: list[tuple[str, float]] = []
    for kv in tr.get_range(r.start, r.stop):
        run_hash: str = idx.unpack(kv.key)[1]
        try:
            ts = encoding.decode_value(kv.value)
            if not isinstance(ts, (int, float)):
                ts = 0.0
        except Exception:  # noqa: BLE001
            ts = 0.0
        result.append((run_hash, float(ts)))
    return result


@transactional
def rename_experiment_index(tr: Transaction, old_name: str, new_name: str) -> None:
    """Move all ``("experiment", old_name, *)`` entries to ``new_name``."""
    idx = _idx_dir()
    r = idx.range((_EXPERIMENT, old_name))
    run_hashes = [idx.unpack(kv.key)[2] for kv in tr.get_range(r.start, r.stop)]
    for rh in run_hashes:
        _delete_fwd_rev(tr, idx, (_EXPERIMENT, old_name, rh), rh)
        _write_fwd_rev(tr, idx, (_EXPERIMENT, new_name, rh), rh)


# ---------------------------------------------------------------------------
# Tier 2 — hparam index write helpers
# ---------------------------------------------------------------------------


@transactional
def index_hparams(tr: Transaction, run_hash: str, hparams: dict[str, Any]) -> None:
    """Write hparam index entries for all top-level scalar values (idempotent)."""
    idx = _idx_dir()
    for key, val in hparams.items():
        if isinstance(val, _INDEXABLE_SCALAR_TYPES):
            _write_fwd_rev(tr, idx, (_HPARAM, key, val, run_hash), run_hash)


@transactional
def deindex_hparams(tr: Transaction, run_hash: str) -> None:
    """Remove all hparam index entries for a run.

    Uses the reverse index to find only this run's hparam entries.
    """
    idx = _idx_dir()
    r = idx.range((_REV, run_hash))
    to_delete: list[tuple] = []
    for kv in tr.get_range(r.start, r.stop):
        rev_tuple = idx.unpack(kv.key)
        forward_key = rev_tuple[2:]
        if forward_key and forward_key[0] == _HPARAM:
            to_delete.append((forward_key, kv.key))
    for forward_key, rev_packed in to_delete:
        fwd_packed = idx.pack(forward_key)
        if tr[fwd_packed].present():
            del tr[fwd_packed]
        del tr[rev_packed]


# ---------------------------------------------------------------------------
# Tier 3 — metric trace name index write helpers
# ---------------------------------------------------------------------------


@transactional
def index_trace(tr: Transaction, run_hash: str, metric_name: str) -> None:
    """Write a trace-name index entry (idempotent)."""
    idx = _idx_dir()
    _write_fwd_rev(tr, idx, (_TRACE, metric_name, run_hash), run_hash)


@transactional
def deindex_traces(tr: Transaction, run_hash: str) -> None:
    """Remove all trace-name index entries for a run via reverse scan."""
    idx = _idx_dir()
    r = idx.range((_REV, run_hash))
    to_delete: list[tuple] = []
    for kv in tr.get_range(r.start, r.stop):
        rev_tuple = idx.unpack(kv.key)
        forward_key = rev_tuple[2:]
        if forward_key and forward_key[0] == _TRACE:
            to_delete.append((forward_key, kv.key))
    for forward_key, rev_packed in to_delete:
        fwd_packed = idx.pack(forward_key)
        if tr[fwd_packed].present():
            del tr[fwd_packed]
        del tr[rev_packed]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


@transactional
def lookup_by_archived(tr: Transaction, archived: bool) -> list[str]:
    idx = _idx_dir()
    r = idx.range((_ARCHIVED, archived))
    return [idx.unpack(kv.key)[2] for kv in tr.get_range(r.start, r.stop)]


@transactional
def lookup_by_active(tr: Transaction, active: bool) -> list[str]:
    idx = _idx_dir()
    r = idx.range((_ACTIVE, active))
    return [idx.unpack(kv.key)[2] for kv in tr.get_range(r.start, r.stop)]


@transactional
def lookup_by_experiment(tr: Transaction, experiment_name: str) -> list[str]:
    idx = _idx_dir()
    r = idx.range((_EXPERIMENT, experiment_name))
    return [idx.unpack(kv.key)[2] for kv in tr.get_range(r.start, r.stop)]


@transactional
def lookup_by_created_at(
    tr: Transaction,
    start: float | None = None,
    end: float | None = None,
) -> list[str]:
    idx = _idx_dir()
    if start is None and end is None:
        r = idx.range((_CREATED_AT,))
        return [idx.unpack(kv.key)[2] for kv in tr.get_range(r.start, r.stop)]

    begin_key = idx.pack((_CREATED_AT, start or 0.0))
    end_key = idx.pack((_CREATED_AT, end)) if end is not None else idx.range((_CREATED_AT,)).stop
    return [idx.unpack(kv.key)[2] for kv in tr.get_range(begin_key, end_key)]


@transactional
def lookup_by_tag(tr: Transaction, tag_name: str) -> list[str]:
    idx = _idx_dir()
    r = idx.range((_TAG, tag_name))
    return [idx.unpack(kv.key)[2] for kv in tr.get_range(r.start, r.stop)]


# ---------------------------------------------------------------------------
# Tier 3 — metric trace name lookup helpers
# ---------------------------------------------------------------------------


@transactional
def lookup_by_trace_name(tr: Transaction, metric_name: str) -> list[str]:
    """Return run hashes that have a metric trace named *metric_name*."""
    idx = _idx_dir()
    r = idx.range((_TRACE, metric_name))
    return [idx.unpack(kv.key)[2] for kv in tr.get_range(r.start, r.stop)]


@transactional
def lookup_all_run_hashes(tr: Transaction) -> list[str]:
    """Return all indexed run hashes (ordered by creation time)."""
    idx = _idx_dir()
    r = idx.range((_CREATED_AT,))
    return [idx.unpack(kv.key)[2] for kv in tr.get_range(r.start, r.stop)]


def iter_run_hashes_from_index(db: Database) -> Iterator[str]:
    """Yield run hashes lazily from the ``created_at`` index.

    Unlike :func:`lookup_all_run_hashes` this does **not** materialise the
    full list, so callers that ``break`` early avoid reading the entire index.

    Uses an explicit transaction (``fdb.transactional`` does not support
    generators).  The transaction stays open while the caller iterates and
    is cleaned up when the generator is garbage-collected.
    """
    tr = db.create_transaction()
    idx = _idx_dir()
    r = idx.range((_CREATED_AT,))
    for kv in tr.get_range(r.start, r.stop):
        yield idx.unpack(kv.key)[2]


@transactional
def iter_created_at_timestamps(tr: Transaction) -> list[float]:
    """Return all ``created_at`` timestamps from the index (no per-run FDB reads)."""
    idx = _idx_dir()
    r = idx.range((_CREATED_AT,))
    return [idx.unpack(kv.key)[1] for kv in tr.get_range(r.start, r.stop)]


@transactional
def count_by_archived(tr: Transaction, archived: bool) -> int:
    idx = _idx_dir()
    r = idx.range((_ARCHIVED, archived))
    count = 0
    for _ in tr.get_range(r.start, r.stop):
        count += 1
    return count


@transactional
def count_by_active(tr: Transaction, active: bool) -> int:
    idx = _idx_dir()
    r = idx.range((_ACTIVE, active))
    count = 0
    for _ in tr.get_range(r.start, r.stop):
        count += 1
    return count


# ---------------------------------------------------------------------------
# Tier 2 — hparam lookup helpers
# ---------------------------------------------------------------------------


@transactional
def lookup_by_hparam_eq(tr: Transaction, param_name: str, value: Any) -> list[str]:  # noqa: ANN401
    """Exact-match lookup: return run hashes where ``hparams[param_name] == value``."""
    idx = _idx_dir()
    r = idx.range((_HPARAM, param_name, value))
    return [idx.unpack(kv.key)[3] for kv in tr.get_range(r.start, r.stop)]


@transactional
def lookup_by_hparam_range(
    tr: Transaction,
    param_name: str,
    lo: Any = None,  # noqa: ANN401
    hi: Any = None,  # noqa: ANN401
) -> list[str]:
    """Range lookup on a hparam value.  Bounds are ``[lo, hi)`` (inclusive low, exclusive high).

    Pass ``None`` for an open bound.
    """
    idx = _idx_dir()
    full_range = idx.range((_HPARAM, param_name))
    begin_key = idx.pack((_HPARAM, param_name, lo)) if lo is not None else full_range.start
    end_key = idx.pack((_HPARAM, param_name, hi)) if hi is not None else full_range.stop
    return [idx.unpack(kv.key)[3] for kv in tr.get_range(begin_key, end_key)]


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------


@transactional
def _clear_all_indexes(tr: Transaction) -> None:
    idx = _idx_dir()
    for prefix in (_ARCHIVED, _ACTIVE, _EXPERIMENT, _CREATED_AT, _TAG, _HPARAM, _TRACE, _REV):
        r = idx.range((prefix,))
        del tr[r.start : r.stop]


@transactional
def _purge_ghost_run(
    tr: Transaction,
    rd: DirectorySubspace,
    run_hash: str,
) -> None:
    tree.tree_delete(tr, rd, (run_hash,))
    entities.remove_run_associations(tr, run_hash)


def rebuild_indexes(db: Database) -> tuple[int, int]:
    """Drop and rebuild all indexes (Tier 1 + Tier 2 + reverse) from run data.

    Also removes ghost runs — run data left behind by the ingestion worker
    after a run was deleted (identifiable by missing ``created_at``), along
    with their orphaned entity associations (experiment_runs, run_tags, etc.).

    After indexing, recomputes ``run_count`` on every experiment and tag
    entity and writes the ``("run_experiment", run_hash)`` reverse keys.

    Returns ``(indexed_count, ghost_count)``.
    """
    _clear_all_indexes(db)

    all_hashes = runs.list_run_hashes(db)
    count = 0
    ghost_count = 0

    exp_run_counts: dict[str, int] = {}
    tag_run_counts: dict[str, int] = {}

    for rh in all_hashes:
        result = _reindex_single_run(db, rh)
        if result is None:
            ghost_count += 1
            continue
        exp_id, tag_ids = result
        if exp_id:
            exp_run_counts[exp_id] = exp_run_counts.get(exp_id, 0) + 1
        for tid in tag_ids:
            tag_run_counts[tid] = tag_run_counts.get(tid, 0) + 1
        count += 1

    _sync_entity_run_counts(db, exp_run_counts, tag_run_counts)
    return count, ghost_count


@transactional
def _reindex_single_run(
    tr: Transaction,
    run_hash: str,
) -> tuple[str | None, list[str]] | None:
    """Read all data and write all index entries for one run in a single transaction.

    Returns ``(experiment_id, [tag_ids])`` on success, or ``None`` if the
    run is a ghost and was purged.
    """
    rd = runs._runs_dir()  # noqa: SLF001
    meta = runs.get_run_meta(tr, run_hash)
    if not meta or not meta.get("created_at") or meta.get("pending_deletion"):
        _purge_ghost_run(tr, rd, run_hash)
        return None

    exp_name: str | None = None
    exp_id: str | None = meta.get("experiment_id")
    if exp_id:
        exp = entities.get_experiment(tr, exp_id)
        if exp:
            exp_name = exp.get("name")
        sd = entities.sys_dir()
        tr[sd.pack(("run_experiment", run_hash))] = encoding.encode_value(exp_id)

    tag_names: list[str] = []
    tag_ids: list[str] = []
    tag_dicts = entities.get_tags_for_run(tr, run_hash)
    for td in tag_dicts:
        n = td.get("name")
        if n:
            tag_names.append(n)
        tid = td.get("id")
        if tid:
            tag_ids.append(tid)

    index_run(
        tr,
        run_hash,
        is_archived=meta.get("is_archived", False),
        active=meta.get("active", True),
        created_at=meta.get("created_at", 0.0),
        experiment_name=exp_name,
        tag_names=tag_names,
    )

    hparams = runs.get_run_attrs(tr, run_hash, ("hparams",))
    if isinstance(hparams, dict):
        index_hparams(tr, run_hash, hparams)

    for trace in runs.get_run_traces_info(tr, run_hash):
        name = trace.get("name")
        if name:
            index_trace(tr, run_hash, name)

    return exp_id, tag_ids


@transactional
def _sync_entity_run_counts(
    tr: Transaction,
    exp_counts: dict[str, int],
    tag_counts: dict[str, int],
) -> None:
    """Write ``run_count`` on every experiment and tag entity."""
    sd = entities.sys_dir()

    for exp in entities.list_experiments(tr):
        uid = exp["id"]
        tr[sd.pack(("experiments", uid, "run_count"))] = encoding.encode_value(
            exp_counts.get(uid, 0),
        )

    for tag in entities.list_tags(tr):
        uid = tag["id"]
        tr[sd.pack(("tags", uid, "run_count"))] = encoding.encode_value(
            tag_counts.get(uid, 0),
        )
