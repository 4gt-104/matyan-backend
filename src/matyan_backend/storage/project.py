"""Project-level aggregation queries.

These functions scan FDB to produce summary statistics consumed by the
``/projects/*`` API endpoints.  Project settings (name, pinned sequences, etc.)
are stored under the ``system`` directory.
"""

from __future__ import annotations

import datetime
import threading
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from cachetools import TTLCache

from matyan_backend.fdb_types import Database, transactional

from . import encoding, entities, indexes, runs, tree

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Transaction

_PROJECT_PREFIX = "project"


@transactional
def get_project_info(tr: Transaction) -> dict:
    sd = entities.sys_dir()
    r = sd.range((_PROJECT_PREFIX,))
    fields: dict[str, Any] = {}
    for kv in tr.get_range(r.start, r.stop):
        key_tuple = sd.unpack(kv.key)
        if len(key_tuple) >= 2:
            fields[key_tuple[1]] = encoding.decode_value(kv.value)
    return {
        "name": fields.get("name", "My project"),
        "path": "",
        "description": fields.get("description", ""),
        "telemetry_enabled": 0,
    }


@transactional
def set_project_info(tr: Transaction, **fields: Any) -> None:  # noqa: ANN401
    sd = entities.sys_dir()
    for k, v in fields.items():
        tr[sd.pack((_PROJECT_PREFIX, k, tree.LEAF_SENTINEL))] = encoding.encode_value(v)


@transactional
def get_project_activity(tr: Transaction, tz_offset: int = 0) -> dict:
    """Compute aggregate activity stats across all runs.

    Uses secondary indexes instead of loading per-run metadata.  Counts
    and the activity map are derived from lightweight key-only scans.
    """
    timestamps = indexes.iter_created_at_timestamps(tr)
    num_runs = len(timestamps)
    num_archived = indexes.count_by_archived(tr, True)
    num_active = indexes.count_by_active(tr, True)
    num_experiments = len(entities.list_experiments(tr))

    activity_map: dict[str, int] = defaultdict(int)
    for created in timestamps:
        ts = created - tz_offset * 60
        day = datetime.datetime.fromtimestamp(ts, tz=datetime.UTC).strftime("%Y-%m-%dT00:00:00")
        activity_map[day] += 1

    return {
        "num_experiments": num_experiments,
        "num_runs": num_runs,
        "num_archived_runs": num_archived,
        "num_active_runs": num_active,
        "activity_map": dict(activity_map),
    }


_INTERNAL_ATTR_KEYS = frozenset({"__system_params", "__blobs__"})
_NON_METRIC_DTYPES = frozenset(
    {
        "image",
        "text",
        "distribution",
        "audio",
        "figure",
        "logs",
        "log_records",
    },
)

# ---------------------------------------------------------------------------
# Project params with bounded TTL cache
# ---------------------------------------------------------------------------

_params_cache: TTLCache[tuple, dict] = TTLCache(maxsize=32, ttl=30)
_params_cache_lock = threading.Lock()


def init_params_cache(maxsize: int, ttl: int) -> None:
    """Re-initialise the params cache with the given bounds.

    Called once at application startup so that the cache picks up
    ``Settings.project_params_cache_maxsize`` and
    ``Settings.project_params_cache_ttl``.
    """
    global _params_cache  # noqa: PLW0603
    with _params_cache_lock:
        _params_cache = TTLCache(maxsize=max(maxsize, 1), ttl=max(ttl, 1))


def invalidate_project_params_cache() -> None:
    """Clear the in-memory project params cache."""
    with _params_cache_lock:
        _params_cache.clear()


def get_project_params_cached(
    db: Database,
    sequence_types: tuple[str, ...] = (),
    *,
    exclude_params: bool = False,
) -> dict:
    """Return project params, served from cache if fresh enough."""
    cache_key = (tuple(sorted(sequence_types)), exclude_params)
    with _params_cache_lock:
        cached = _params_cache.get(cache_key)
    if cached is not None:
        return cached

    result = _compute_project_params(db, sequence_types, exclude_params=exclude_params)

    with _params_cache_lock:
        _params_cache[cache_key] = result
    return result


_DTYPE_TO_BUCKET = {
    "image": "images",
    "text": "texts",
    "distribution": "distributions",
    "audio": "audios",
    "figure": "figures",
}

_EMPTY_BUCKETS: dict[str, dict[str, list[dict]]] = {
    "metric": {},
    "images": {},
    "texts": {},
    "figures": {},
    "distributions": {},
    "audios": {},
}


def _collect_trace_buckets(
    tr: Transaction,
    run_hash: str,
    wanted: set[str],
    buckets: dict[str, dict[str, list[dict]]],
) -> None:
    """Merge a single run's traces into *buckets* (mutated in place)."""
    traces = runs.get_run_traces_info(tr, run_hash)
    for t in traces:
        dtype = t.get("dtype", "")
        if dtype in _NON_METRIC_DTYPES and dtype not in _DTYPE_TO_BUCKET:
            continue
        bucket = _DTYPE_TO_BUCKET.get(dtype, "metric")
        if bucket not in wanted:
            continue
        name = t.get("name", "")
        ctx_id = t.get("context_id", 0)
        ctx = runs.get_context(tr, run_hash, ctx_id) or {}
        if name not in buckets[bucket]:
            buckets[bucket][name] = []
        if ctx not in buckets[bucket][name]:
            buckets[bucket][name].append(ctx)


@transactional
def _compute_project_params(
    tr: Transaction,
    sequence_types: tuple[str, ...] = (),
    *,
    exclude_params: bool = False,
) -> dict:
    """Aggregate distinct hparams and trace names across all runs."""
    hashes = indexes.lookup_all_run_hashes(tr)

    all_params: dict[str, Any] = {}
    buckets: dict[str, dict[str, list[dict]]] = {k: {} for k in _EMPTY_BUCKETS}
    wanted = set(sequence_types) if sequence_types else set(buckets)

    for h in hashes:
        meta = runs.get_run_meta(tr, h)
        if not meta or meta.get("pending_deletion"):
            continue

        if not exclude_params:
            attrs = runs.get_run_attrs(tr, h)
            if isinstance(attrs, dict):
                _merge_dict(all_params, attrs)

        _collect_trace_buckets(tr, h, wanted, buckets)

    for key in _INTERNAL_ATTR_KEYS:
        all_params.pop(key, None)

    return {"params": all_params, **buckets}


@transactional
def get_project_params(
    tr: Transaction,
    sequence_types: tuple[str, ...] = (),
    *,
    exclude_params: bool = False,
) -> dict:
    """Aggregate distinct hparams and trace names across all runs.

    Delegates to ``_compute_project_params``.  For cached access from API
    endpoints use ``get_project_params_cached`` instead.
    """
    return _compute_project_params(tr, sequence_types, exclude_params=exclude_params)


def _merge_dict(target: dict, source: dict) -> None:
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _merge_dict(target[k], v)
        else:
            target[k] = v


# ---------------------------------------------------------------------------
# Pinned sequences
# ---------------------------------------------------------------------------

_PINNED = "pinned_sequences"


@transactional
def get_pinned_sequences(tr: Transaction) -> list:
    sd = entities.sys_dir()
    raw = tr[sd.pack((_PINNED, tree.LEAF_SENTINEL))]
    if not raw.present():
        return []
    return encoding.decode_value(raw)


@transactional
def set_pinned_sequences(tr: Transaction, sequences: list) -> list:
    sd = entities.sys_dir()
    tr[sd.pack((_PINNED, tree.LEAF_SENTINEL))] = encoding.encode_value(sequences)
    return sequences
