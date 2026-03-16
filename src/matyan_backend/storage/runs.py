"""Run data storage: CRUD on run metadata, attributes, trace info, and contexts."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from matyan_backend.fdb_types import transactional

from . import encoding, entities, indexes, tree
from .fdb_client import get_directories

if TYPE_CHECKING:
    from matyan_backend.fdb_types import DirectorySubspace, Transaction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runs_dir() -> DirectorySubspace:
    return get_directories().runs


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Create / Get / Update / Delete
# ---------------------------------------------------------------------------


@transactional
def create_run(
    tr: Transaction,
    run_hash: str,
    *,
    name: str | None = None,
    experiment_id: str | None = None,
    description: str | None = None,
) -> dict:
    rd = _runs_dir()
    now = _now()
    meta = {
        "name": name or f"Run: {run_hash}",
        "description": description or "",
        "created_at": now,
        "updated_at": now,
        "finalized_at": None,
        "is_archived": False,
        "active": True,
        "experiment_id": experiment_id,
        "client_start_ts": None,
        "duration": None,
    }
    tree.tree_set(tr, rd, (run_hash, "meta"), meta)

    exp_name: str | None = None
    if experiment_id:
        exp = entities.get_experiment(tr, experiment_id)
        if exp:
            exp_name = exp.get("name")

    indexes.index_run(
        tr,
        run_hash,
        is_archived=False,
        active=True,
        created_at=now,
        experiment_name=exp_name,
    )
    return {"hash": run_hash, **meta}


@transactional
def resume_run(tr: Transaction, run_hash: str) -> dict | None:
    """Re-activate an existing run without overwriting its metadata.

    Returns the updated run dict, or ``None`` if the run does not exist
    (caller should fall back to ``create_run``).
    """
    rd = _runs_dir()
    meta = tree.tree_get(tr, rd, (run_hash, "meta"))
    if meta is None:
        return None

    update_run_meta(tr, run_hash, active=True, finalized_at=None)
    meta["active"] = True
    meta["finalized_at"] = None
    return {"hash": run_hash, **meta}


@transactional
def get_run(tr: Transaction, run_hash: str) -> dict | None:
    rd = _runs_dir()
    meta = tree.tree_get(tr, rd, (run_hash, "meta"))
    if meta is None:
        return None
    return {"hash": run_hash, **meta}


@transactional
def get_run_meta(tr: Transaction, run_hash: str) -> dict:
    rd = _runs_dir()
    meta = tree.tree_get(tr, rd, (run_hash, "meta"))
    return meta or {}


_INDEXED_META_FIELDS = {"is_archived": "archived", "active": "active"}


@transactional
def update_run_meta(tr: Transaction, run_hash: str, **fields: Any) -> None:  # noqa: ANN401
    rd = _runs_dir()

    for meta_key, idx_field in _INDEXED_META_FIELDS.items():
        if meta_key in fields:
            raw_old = tr[rd.pack((run_hash, "meta", meta_key, tree.LEAF_SENTINEL))]
            old_val = encoding.decode_value(raw_old) if raw_old.present() else None
            new_val = fields[meta_key]
            if old_val != new_val:
                indexes.update_index_field(tr, run_hash, idx_field, old_val, new_val)

    fields["updated_at"] = _now()
    for key, val in fields.items():
        tr[rd.pack((run_hash, "meta", key, tree.LEAF_SENTINEL))] = encoding.encode_value(val)


@transactional
def delete_run(tr: Transaction, run_hash: str) -> None:
    indexes.deindex_run(tr, run_hash)
    entities.remove_run_associations(tr, run_hash)
    rd = _runs_dir()
    r = rd.range((run_hash,))
    del tr[r.start : r.stop]
    indexes.mark_run_deleted(tr, run_hash)


@transactional
def mark_pending_deletion(tr: Transaction, run_hash: str) -> None:
    """Flag a run so it is hidden from read paths immediately."""
    rd = _runs_dir()
    tr[rd.pack((run_hash, "meta", "pending_deletion", tree.LEAF_SENTINEL))] = encoding.encode_value(True)


@transactional
def is_pending_deletion(tr: Transaction, run_hash: str) -> bool:
    rd = _runs_dir()
    raw = tr[rd.pack((run_hash, "meta", "pending_deletion", tree.LEAF_SENTINEL))]
    if raw.present():
        return bool(encoding.decode_value(raw))
    return False


@transactional
def list_run_hashes(tr: Transaction) -> list[str]:
    rd = _runs_dir()
    return [str(k) for k in tree.tree_keys(tr, rd, ())]


# ---------------------------------------------------------------------------
# Attributes (nested user data — hparams etc.)
# ---------------------------------------------------------------------------


@transactional
def get_run_attrs(tr: Transaction, run_hash: str, path: tuple = ()) -> Any:  # noqa: ANN401
    rd = _runs_dir()
    full_path = (run_hash, "attrs", *path)
    return tree.tree_get(tr, rd, full_path)


@transactional
def get_run_artifacts(tr: Transaction, run_hash: str) -> list[dict]:
    """Return raw file-artifact metadata stored under ``__blobs__``.

    Excludes custom-object blobs (images, audio) which are stored under
    ``seq/`` prefixed paths by the client's ``_upload_blob_if_needed``.
    Each entry has ``name``, ``path``, ``s3_key``, and ``content_type``.
    """
    rd = _runs_dir()
    blobs = tree.tree_get(tr, rd, (run_hash, "attrs", "__blobs__"))
    if not isinstance(blobs, dict):
        return []
    result: list[dict] = []
    for artifact_path, meta in blobs.items():
        if not isinstance(meta, dict):
            continue
        if artifact_path.startswith("seq/"):
            continue
        result.append(
            {
                "name": artifact_path.rsplit("/", 1)[-1] if "/" in artifact_path else artifact_path,
                "path": artifact_path,
                "s3_key": meta.get("s3_key", ""),
                "content_type": meta.get("content_type", "application/octet-stream"),
            },
        )
    return result


@transactional
def set_run_attrs(tr: Transaction, run_hash: str, path: tuple, value: Any) -> None:  # noqa: ANN401
    rd = _runs_dir()
    full_path = (run_hash, "attrs", *path)
    tree.tree_set(tr, rd, full_path, value)
    tr[rd.pack((run_hash, "meta", "updated_at", tree.LEAF_SENTINEL))] = encoding.encode_value(_now())

    if path == ("hparams",) and isinstance(value, dict):
        indexes.deindex_hparams(tr, run_hash)
        indexes.index_hparams(tr, run_hash, value)
    elif len(path) >= 1 and path[0] == "hparams":
        hparams = tree.tree_get(tr, rd, (run_hash, "attrs", "hparams"))
        if isinstance(hparams, dict):
            indexes.deindex_hparams(tr, run_hash)
            indexes.index_hparams(tr, run_hash, hparams)


# ---------------------------------------------------------------------------
# Traces metadata
# ---------------------------------------------------------------------------


@transactional
def set_trace_info(
    tr: Transaction,
    run_hash: str,
    ctx_id: int,
    name: str,
    *,
    dtype: str,
    last: Any = None,  # noqa: ANN401
    last_step: int | None = None,
) -> None:
    rd = _runs_dir()
    base = (run_hash, "traces", ctx_id, name)
    tr[rd.pack((*base, "dtype"))] = encoding.encode_value(dtype)
    if last is not None:
        tr[rd.pack((*base, "last"))] = encoding.encode_value(last)
    if last_step is not None:
        tr[rd.pack((*base, "last_step"))] = encoding.encode_value(last_step)
    indexes.index_trace(tr, run_hash, name)


@transactional
def get_run_traces_info(tr: Transaction, run_hash: str) -> list[dict]:
    rd = _runs_dir()
    r = rd.range((run_hash, "traces"))
    traces: dict[tuple[int, str], dict] = {}

    for kv in tr.get_range(r.start, r.stop):
        full_key = rd.unpack(kv.key)
        ctx_id = full_key[2]
        metric_name = full_key[3]
        field = full_key[4]
        ident = (ctx_id, metric_name)
        if ident not in traces:
            traces[ident] = {"name": metric_name, "context_id": ctx_id}
        traces[ident][field] = encoding.decode_value(kv.value)

    return list(traces.values())


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------


@transactional
def set_context(tr: Transaction, run_hash: str, ctx_id: int, context: dict) -> None:
    rd = _runs_dir()
    tr[rd.pack((run_hash, "contexts", ctx_id))] = encoding.encode_value(context)


@transactional
def get_context(tr: Transaction, run_hash: str, ctx_id: int) -> dict | None:
    rd = _runs_dir()
    raw = tr[rd.pack((run_hash, "contexts", ctx_id))]
    if raw.present():
        return encoding.decode_value(raw)
    return None


@transactional
def get_all_contexts(tr: Transaction, run_hash: str) -> dict[int, dict]:
    rd = _runs_dir()
    r = rd.range((run_hash, "contexts"))
    result: dict[int, dict] = {}
    for kv in tr.get_range(r.start, r.stop):
        ctx_id = rd.unpack(kv.key)[2]
        result[ctx_id] = encoding.decode_value(kv.value)
    return result


# ---------------------------------------------------------------------------
# Run ↔ Tag associations (stored in runs_dir for co-location)
# ---------------------------------------------------------------------------


@transactional
def add_tag_to_run(tr: Transaction, run_hash: str, tag_uuid: str) -> None:
    rd = _runs_dir()
    tr[rd.pack((run_hash, "tags", tag_uuid))] = encoding.encode_value(True)


@transactional
def remove_tag_from_run(tr: Transaction, run_hash: str, tag_uuid: str) -> None:
    rd = _runs_dir()
    del tr[rd.pack((run_hash, "tags", tag_uuid))]


@transactional
def get_run_tag_uuids(tr: Transaction, run_hash: str) -> list[str]:
    rd = _runs_dir()
    r = rd.range((run_hash, "tags"))
    return [rd.unpack(kv.key)[2] for kv in tr.get_range(r.start, r.stop)]


@transactional
def set_run_experiment(tr: Transaction, run_hash: str, experiment_id: str | None) -> None:
    rd = _runs_dir()
    tr[rd.pack((run_hash, "meta", "experiment_id", tree.LEAF_SENTINEL))] = encoding.encode_value(experiment_id)
    tr[rd.pack((run_hash, "meta", "updated_at", tree.LEAF_SENTINEL))] = encoding.encode_value(_now())


def _get_run_bundle_impl(
    tr: Transaction,
    run_hash: str,
    *,
    include_attrs: bool = True,
    include_traces: bool = True,
) -> dict | None:
    """Shared implementation for reading a single run bundle inside a transaction.

    Callable from both ``get_run_bundle`` (single-run) and ``get_run_bundles``
    (batched).  The caller must supply an open FDB transaction as *tr*.
    """
    rd = _runs_dir()
    meta = tree.tree_get(tr, rd, (run_hash, "meta"))
    if not meta or meta.get("pending_deletion"):
        return None

    attrs = None
    if include_attrs:
        attrs = tree.tree_get(tr, rd, (run_hash, "attrs"))

    traces_list: list[dict] = []
    contexts: dict[int, dict] = {}
    if include_traces:
        raw_traces: dict[tuple[int, str], dict] = {}
        r = rd.range((run_hash, "traces"))
        for kv in tr.get_range(r.start, r.stop):
            full_key = rd.unpack(kv.key)
            ctx_id = full_key[2]
            metric_name = full_key[3]
            field = full_key[4]
            ident = (ctx_id, metric_name)
            if ident not in raw_traces:
                raw_traces[ident] = {"name": metric_name, "context_id": ctx_id}
            raw_traces[ident][field] = encoding.decode_value(kv.value)
        traces_list = list(raw_traces.values())

        r = rd.range((run_hash, "contexts"))
        for kv in tr.get_range(r.start, r.stop):
            ctx_id = rd.unpack(kv.key)[2]
            contexts[ctx_id] = encoding.decode_value(kv.value)

    sd = entities.sys_dir()
    tags_raw: list[dict] = []
    r = sd.range(("run_tags", run_hash))
    for kv in tr.get_range(r.start, r.stop):
        tag_uuid = sd.unpack(kv.key)[2]
        tag = entities.read_entity(tr, "tags", tag_uuid)
        if tag:
            tags_raw.append(tag)

    experiment: dict | None = None
    exp_id = meta.get("experiment_id")
    if exp_id:
        experiment = entities.read_entity(tr, "experiments", exp_id)

    return {
        "meta": meta,
        "attrs": attrs,
        "traces": traces_list,
        "contexts": contexts,
        "tags": tags_raw,
        "experiment": experiment,
    }


@transactional
def get_run_bundle(
    tr: Transaction,
    run_hash: str,
    *,
    include_attrs: bool = True,
    include_traces: bool = True,
) -> dict | None:
    """Read all run data needed by search endpoints in a single FDB transaction.

    Combines meta, attrs, traces, contexts, tags, and experiment reads into
    one round-trip instead of 4-6 separate transactions.

    Returns ``None`` if the run has no meta or is pending deletion.
    """
    return _get_run_bundle_impl(
        tr,
        run_hash,
        include_attrs=include_attrs,
        include_traces=include_traces,
    )


@transactional
def get_metric_search_bundle(tr: Transaction, run_hash: str) -> dict | None:
    """Read the bundle shape needed by metric-search candidate streaming.

    Keeps the response-compatible shape of :func:`get_run_bundle` while giving
    metric-search code a dedicated single-run helper for exact fast paths.
    """
    return _get_run_bundle_impl(
        tr,
        run_hash,
        include_attrs=True,
        include_traces=True,
    )


@transactional
def get_run_bundles(
    tr: Transaction,
    run_hashes: list[str],
    *,
    include_attrs: bool = True,
    include_traces: bool = True,
) -> list[dict | None]:
    """Read bundles for multiple runs in a single FDB transaction.

    Returns a list of the same length as *run_hashes*.  Each element is
    either a bundle dict (same shape as ``get_run_bundle``) or ``None``
    for missing/deleted runs.
    """
    return [
        _get_run_bundle_impl(
            tr,
            h,
            include_attrs=include_attrs,
            include_traces=include_traces,
        )
        for h in run_hashes
    ]


@transactional
def list_runs_meta(tr: Transaction) -> list[dict]:
    """Return metadata dicts for all runs (used by project-level aggregations)."""
    hashes = list_run_hashes(tr)
    results: list[dict] = []
    for h in hashes:
        meta = get_run_meta(tr, h)
        if meta:
            meta["hash"] = h
            results.append(meta)
    return results
