"""Structured entity storage: experiments, tags, dashboards, dashboard apps, reports, notes.

All entities live under the ``system`` FDB directory.  Each entity type uses a
pair of key prefixes: one for the records and one (optional) unique-name index.

Key layout
----------
('experiments', uuid, field)           → value
('experiments_by_name', name)          → uuid
('tags', uuid, field)                  → value
('tags_by_name', name)                 → uuid
('dashboards', uuid, field)            → value
('dashboard_apps', uuid, field)        → value
('reports', uuid, field)               → value
('notes', uuid, field)                 → value
('run_tags', run_hash, tag_uuid)       → True
('tag_runs', tag_uuid, run_hash)       → True
('experiment_runs', exp_uuid, run_hash)→ True
"""

from __future__ import annotations

import time
import uuid as _uuid
from typing import TYPE_CHECKING, Any

from matyan_backend.fdb_types import transactional

from . import encoding, indexes, runs
from .fdb_client import get_directories

if TYPE_CHECKING:
    from matyan_backend.fdb_types import DirectorySubspace, Transaction


def sys_dir() -> DirectorySubspace:
    return get_directories().system


def _now() -> float:
    return time.time()


def _new_uuid() -> str:
    return str(_uuid.uuid4())


# ===================================================================
# Generic helpers
# ===================================================================


def _write_entity(tr: Transaction, prefix: str, entity_uuid: str, data: dict) -> None:
    sd = sys_dir()
    for field, val in data.items():
        tr[sd.pack((prefix, entity_uuid, field))] = encoding.encode_value(val)


def read_entity(tr: Transaction, prefix: str, entity_uuid: str) -> dict | None:
    sd = sys_dir()
    r = sd.range((prefix, entity_uuid))
    fields: dict[str, Any] = {}
    for kv in tr.get_range(r.start, r.stop):
        key_tuple = sd.unpack(kv.key)
        fields[key_tuple[2]] = encoding.decode_value(kv.value)
    if not fields:
        return None
    fields["id"] = entity_uuid
    return fields


def _list_entities(tr: Transaction, prefix: str) -> list[dict[str, Any]]:
    sd = sys_dir()
    r = sd.range((prefix,))
    entities: dict[str, dict[str, Any]] = {}
    for kv in tr.get_range(r.start, r.stop):
        key_tuple = sd.unpack(kv.key)
        entity_uuid = key_tuple[1]
        field = key_tuple[2]
        if entity_uuid not in entities:
            entities[entity_uuid] = {"id": entity_uuid}
        entities[entity_uuid][field] = encoding.decode_value(kv.value)
    return list(entities.values())


def _delete_entity(tr: Transaction, prefix: str, entity_uuid: str) -> None:
    sd = sys_dir()
    r = sd.range((prefix, entity_uuid))
    del tr[r.start : r.stop]


def _update_entity(tr: Transaction, prefix: str, entity_uuid: str, **fields: Any) -> None:  # noqa: ANN401
    sd = sys_dir()
    fields["updated_at"] = _now()
    for field, val in fields.items():
        tr[sd.pack((prefix, entity_uuid, field))] = encoding.encode_value(val)


def _adjust_run_count(tr: Transaction, prefix: str, entity_uuid: str, delta: int) -> None:
    """Atomically increment or decrement the ``run_count`` field on an entity."""
    sd = sys_dir()
    key = sd.pack((prefix, entity_uuid, "run_count"))
    raw = tr[key]
    current = encoding.decode_value(raw) if raw.present() else 0
    tr[key] = encoding.encode_value(max(0, current + delta))


# ===================================================================
# Experiments
# ===================================================================

_EXP = "experiments"
_EXP_BY_NAME = "experiments_by_name"
_EXP_RUNS = "experiment_runs"
_RUN_EXP = "run_experiment"


@transactional
def create_experiment(tr: Transaction, name: str, *, description: str | None = None) -> dict:
    sd = sys_dir()

    existing = tr[sd.pack((_EXP_BY_NAME, name))]
    if existing.present():
        msg = f"Experiment with name '{name}' already exists"
        raise ValueError(msg)

    uid = _new_uuid()
    now = _now()
    data = {
        "name": name,
        "description": description or "",
        "is_archived": False,
        "run_count": 0,
        "created_at": now,
        "updated_at": now,
    }
    _write_entity(tr, _EXP, uid, data)
    tr[sd.pack((_EXP_BY_NAME, name))] = encoding.encode_value(uid)
    return {"id": uid, **data}


@transactional
def get_experiment(tr: Transaction, uid: str) -> dict | None:
    return read_entity(tr, _EXP, uid)


@transactional
def get_experiment_by_name(tr: Transaction, name: str) -> dict | None:
    sd = sys_dir()
    raw = tr[sd.pack((_EXP_BY_NAME, name))]
    if not raw.present():
        return None
    uid = encoding.decode_value(raw)
    return read_entity(tr, _EXP, uid)


@transactional
def list_experiments(tr: Transaction) -> list[dict[str, Any]]:
    return _list_entities(tr, _EXP)


@transactional
def update_experiment(tr: Transaction, uid: str, **fields: Any) -> None:  # noqa: ANN401
    sd = sys_dir()
    if "name" in fields:
        old = read_entity(tr, _EXP, uid)
        if old and old.get("name") != fields["name"]:
            old_name = old["name"]
            new_name = fields["name"]
            del tr[sd.pack((_EXP_BY_NAME, old_name))]
            tr[sd.pack((_EXP_BY_NAME, new_name))] = encoding.encode_value(uid)
            indexes.rename_experiment_index(tr, old_name, new_name)
    _update_entity(tr, _EXP, uid, **fields)


@transactional
def delete_experiment(tr: Transaction, uid: str) -> None:
    sd = sys_dir()
    entity = read_entity(tr, _EXP, uid)
    if entity:
        name = entity.get("name")
        if name:
            del tr[sd.pack((_EXP_BY_NAME, name))]
    _delete_entity(tr, _EXP, uid)
    # Clean up experiment_runs index
    r = sd.range((_EXP_RUNS, uid))
    del tr[r.start : r.stop]


@transactional
def get_runs_for_experiment(tr: Transaction, experiment_uuid: str) -> list[str]:
    sd = sys_dir()
    r = sd.range((_EXP_RUNS, experiment_uuid))
    return [sd.unpack(kv.key)[2] for kv in tr.get_range(r.start, r.stop)]


@transactional
def get_run_experiment_names(
    tr: Transaction,
    run_hashes: list[str],
    run_metas: dict[str, dict] | None = None,
) -> dict[str, str]:
    """Return a ``{run_hash: experiment_name}`` mapping for *run_hashes*.

    Resolves the experiment for each run by reading ``meta["experiment_id"]``
    (either from *run_metas* if supplied, or from the run's FDB meta tree),
    then reading only the experiment ``name`` field.  Experiment names are
    cached within the call to avoid re-reading the same experiment for runs
    that share one.

    Runs with no experiment association are mapped to ``""``.

    :param tr: FDB transaction.
    :param run_hashes: Run hashes to look up.
    :param run_metas: Optional pre-loaded ``{hash: meta_dict}`` to avoid extra
        FDB reads.  When ``None``, meta is read via the ``run_experiment``
        reverse key or the run's meta tree.
    :returns: Dict mapping each run hash to its experiment name (or ``""``).
    """
    sd = sys_dir()
    exp_name_cache: dict[str, str] = {}
    result: dict[str, str] = {}
    for rh in run_hashes:
        exp_uuid: str | None = None

        if run_metas and rh in run_metas:
            exp_uuid = run_metas[rh].get("experiment_id")
        else:
            rev_key = sd.pack((_RUN_EXP, rh))
            raw = tr[rev_key]
            if raw.present():
                exp_uuid = encoding.decode_value(raw)
            else:
                meta = runs.get_run_meta(tr, rh)
                exp_uuid = meta.get("experiment_id") if meta else None

        if not exp_uuid:
            result[rh] = ""
            continue
        if exp_uuid in exp_name_cache:
            result[rh] = exp_name_cache[exp_uuid]
            continue
        name_key = sd.pack((_EXP, exp_uuid, "name"))
        name_raw = tr[name_key]
        name = encoding.decode_value(name_raw) if name_raw.present() else ""
        exp_name_cache[exp_uuid] = name
        result[rh] = name
    return result


@transactional
def set_run_experiment(tr: Transaction, run_hash: str, experiment_uuid: str | None) -> None:
    """Update the experiment_runs index when a run's experiment changes.

    Uses a reverse key ``("run_experiment", run_hash) → exp_uuid`` for
    O(1) lookup of the old experiment instead of scanning all entries.
    """
    sd = sys_dir()

    old_exp_name: str | None = None
    rev_key = sd.pack((_RUN_EXP, run_hash))
    old_raw = tr[rev_key]
    if old_raw.present():
        old_exp_uuid = encoding.decode_value(old_raw)
        old_exp = read_entity(tr, _EXP, old_exp_uuid)
        if old_exp:
            old_exp_name = old_exp.get("name")
            _adjust_run_count(tr, _EXP, old_exp_uuid, -1)
        del tr[sd.pack((_EXP_RUNS, old_exp_uuid, run_hash))]
        del tr[rev_key]

    new_exp_name: str | None = None
    if experiment_uuid is not None:
        tr[sd.pack((_EXP_RUNS, experiment_uuid, run_hash))] = encoding.encode_value(True)
        tr[sd.pack((_RUN_EXP, run_hash))] = encoding.encode_value(experiment_uuid)
        new_exp = read_entity(tr, _EXP, experiment_uuid)
        if new_exp:
            new_exp_name = new_exp.get("name")
            _adjust_run_count(tr, _EXP, experiment_uuid, 1)

    indexes.update_index_field(tr, run_hash, "experiment", old_exp_name, new_exp_name)


# ===================================================================
# Tags
# ===================================================================

_TAG = "tags"
_TAG_BY_NAME = "tags_by_name"
_RUN_TAGS = "run_tags"
_TAG_RUNS = "tag_runs"


@transactional
def create_tag(tr: Transaction, name: str, *, color: str | None = None, description: str | None = None) -> dict:
    sd = sys_dir()

    existing = tr[sd.pack((_TAG_BY_NAME, name))]
    if existing.present():
        msg = f"Tag with name '{name}' already exists"
        raise ValueError(msg)

    uid = _new_uuid()
    now = _now()
    data = {
        "name": name,
        "color": color or "",
        "description": description or "",
        "is_archived": False,
        "run_count": 0,
        "created_at": now,
        "updated_at": now,
    }
    _write_entity(tr, _TAG, uid, data)
    tr[sd.pack((_TAG_BY_NAME, name))] = encoding.encode_value(uid)
    return {"id": uid, **data}


@transactional
def get_tag(tr: Transaction, uid: str) -> dict | None:
    return read_entity(tr, _TAG, uid)


@transactional
def get_tag_by_name(tr: Transaction, name: str) -> dict | None:
    sd = sys_dir()
    raw = tr[sd.pack((_TAG_BY_NAME, name))]
    if not raw.present():
        return None
    uid = encoding.decode_value(raw)
    return read_entity(tr, _TAG, uid)


@transactional
def list_tags(tr: Transaction) -> list[dict[str, Any]]:
    return _list_entities(tr, _TAG)


@transactional
def update_tag(tr: Transaction, uid: str, **fields: Any) -> None:  # noqa: ANN401
    sd = sys_dir()
    if "name" in fields:
        old = read_entity(tr, _TAG, uid)
        if old and old.get("name") != fields["name"]:
            del tr[sd.pack((_TAG_BY_NAME, old["name"]))]
            tr[sd.pack((_TAG_BY_NAME, fields["name"]))] = encoding.encode_value(uid)
    _update_entity(tr, _TAG, uid, **fields)


@transactional
def delete_tag(tr: Transaction, uid: str) -> None:
    sd = sys_dir()
    entity = read_entity(tr, _TAG, uid)
    tag_name: str | None = None
    if entity:
        tag_name = entity.get("name")
        if tag_name:
            del tr[sd.pack((_TAG_BY_NAME, tag_name))]
    _delete_entity(tr, _TAG, uid)
    r = sd.range((_TAG_RUNS, uid))
    for kv in tr.get_range(r.start, r.stop):
        run_hash = sd.unpack(kv.key)[2]
        del tr[sd.pack((_RUN_TAGS, run_hash, uid))]
    del tr[r.start : r.stop]
    if tag_name:
        indexes.remove_all_tag_indexes_for_tag(tr, tag_name)


# --- Run ↔ Tag associations ---


@transactional
def add_tag_to_run(tr: Transaction, run_hash: str, tag_uuid: str) -> None:
    sd = sys_dir()
    tr[sd.pack((_RUN_TAGS, run_hash, tag_uuid))] = encoding.encode_value(True)
    tr[sd.pack((_TAG_RUNS, tag_uuid, run_hash))] = encoding.encode_value(True)
    _adjust_run_count(tr, _TAG, tag_uuid, 1)
    tag = read_entity(tr, _TAG, tag_uuid)
    if tag and tag.get("name"):
        indexes.add_tag_index(tr, run_hash, tag["name"])


@transactional
def remove_tag_from_run(tr: Transaction, run_hash: str, tag_uuid: str) -> None:
    sd = sys_dir()
    tag = read_entity(tr, _TAG, tag_uuid)
    if tag and tag.get("name"):
        indexes.remove_tag_index(tr, run_hash, tag["name"])
    del tr[sd.pack((_RUN_TAGS, run_hash, tag_uuid))]
    del tr[sd.pack((_TAG_RUNS, tag_uuid, run_hash))]
    _adjust_run_count(tr, _TAG, tag_uuid, -1)


@transactional
def get_tags_for_run(tr: Transaction, run_hash: str) -> list[dict]:
    sd = sys_dir()
    r = sd.range((_RUN_TAGS, run_hash))
    tags = []
    for kv in tr.get_range(r.start, r.stop):
        tag_uuid = sd.unpack(kv.key)[2]
        tag = read_entity(tr, _TAG, tag_uuid)
        if tag:
            tags.append(tag)
    return tags


@transactional
def get_runs_for_tag(tr: Transaction, tag_uuid: str) -> list[str]:
    sd = sys_dir()
    r = sd.range((_TAG_RUNS, tag_uuid))
    return [sd.unpack(kv.key)[2] for kv in tr.get_range(r.start, r.stop)]


@transactional
def remove_run_associations(tr: Transaction, run_hash: str) -> None:
    """Remove all experiment and tag associations for a run.

    Uses the ``("run_experiment", run_hash)`` reverse key for O(1)
    experiment lookup.
    """
    sd = sys_dir()

    # Remove experiment association via reverse key (O(1))
    rev_key = sd.pack((_RUN_EXP, run_hash))
    old_raw = tr[rev_key]
    if old_raw.present():
        old_exp_uuid = encoding.decode_value(old_raw)
        _adjust_run_count(tr, _EXP, old_exp_uuid, -1)
        del tr[sd.pack((_EXP_RUNS, old_exp_uuid, run_hash))]
        del tr[rev_key]

    # Remove run_tags entries (run -> tag direction)
    r = sd.range((_RUN_TAGS, run_hash))
    tag_uuids = [sd.unpack(kv.key)[2] for kv in tr.get_range(r.start, r.stop)]
    del tr[r.start : r.stop]

    # Remove tag_runs entries (tag -> run direction) + decrement run_count
    for tag_uuid in tag_uuids:
        del tr[sd.pack((_TAG_RUNS, tag_uuid, run_hash))]
        _adjust_run_count(tr, _TAG, tag_uuid, -1)


# ===================================================================
# Dashboards
# ===================================================================

_DASH = "dashboards"


@transactional
def create_dashboard(
    tr: Transaction,
    name: str,
    *,
    description: str | None = None,
    app_id: str | None = None,
) -> dict:
    uid = _new_uuid()
    now = _now()
    data: dict[str, Any] = {
        "name": name,
        "description": description or "",
        "is_archived": False,
        "created_at": now,
        "updated_at": now,
    }
    if app_id is not None:
        data["app_id"] = app_id
    _write_entity(tr, _DASH, uid, data)
    return {"id": uid, **data}


@transactional
def get_dashboard(tr: Transaction, uid: str) -> dict | None:
    return read_entity(tr, _DASH, uid)


@transactional
def list_dashboards(tr: Transaction) -> list[dict[str, Any]]:
    return _list_entities(tr, _DASH)


@transactional
def update_dashboard(tr: Transaction, uid: str, **fields: Any) -> None:  # noqa: ANN401
    _update_entity(tr, _DASH, uid, **fields)


@transactional
def delete_dashboard(tr: Transaction, uid: str) -> None:
    _delete_entity(tr, _DASH, uid)


# ===================================================================
# Dashboard Apps (ExploreState)
# ===================================================================

_APP = "dashboard_apps"


@transactional
def create_dashboard_app(
    tr: Transaction,
    app_type: str,
    state: dict | None = None,
    *,
    dashboard_id: str | None = None,
) -> dict[str, Any]:
    uid = _new_uuid()
    now = _now()
    data: dict[str, Any] = {
        "type": app_type,
        "state": state or {},
        "created_at": now,
        "updated_at": now,
    }
    if dashboard_id is not None:
        data["dashboard_id"] = dashboard_id
    _write_entity(tr, _APP, uid, data)
    return {"id": uid, **data}


@transactional
def get_dashboard_app(tr: Transaction, uid: str) -> dict | None:
    return read_entity(tr, _APP, uid)


@transactional
def list_dashboard_apps(tr: Transaction) -> list[dict[str, Any]]:
    return _list_entities(tr, _APP)


@transactional
def update_dashboard_app(tr: Transaction, uid: str, **fields: Any) -> None:  # noqa: ANN401
    _update_entity(tr, _APP, uid, **fields)


@transactional
def delete_dashboard_app(tr: Transaction, uid: str) -> None:
    _delete_entity(tr, _APP, uid)


# ===================================================================
# Reports
# ===================================================================

_REPORT = "reports"


@transactional
def create_report(tr: Transaction, name: str, *, code: str | None = None, description: str | None = None) -> dict:
    uid = _new_uuid()
    now = _now()
    data = {
        "name": name,
        "code": code or "",
        "description": description or "",
        "created_at": now,
        "updated_at": now,
    }
    _write_entity(tr, _REPORT, uid, data)
    return {"id": uid, **data}


@transactional
def get_report(tr: Transaction, uid: str) -> dict | None:
    return read_entity(tr, _REPORT, uid)


@transactional
def list_reports(tr: Transaction) -> list[dict[str, Any]]:
    return _list_entities(tr, _REPORT)


@transactional
def update_report(tr: Transaction, uid: str, **fields: Any) -> None:  # noqa: ANN401
    _update_entity(tr, _REPORT, uid, **fields)


@transactional
def delete_report(tr: Transaction, uid: str) -> None:
    _delete_entity(tr, _REPORT, uid)


# ===================================================================
# Notes
# ===================================================================

_NOTE = "notes"


@transactional
def create_note(
    tr: Transaction,
    content: str = "",
    *,
    run_hash: str | None = None,
    experiment_id: str | None = None,
) -> dict:
    uid = _new_uuid()
    now = _now()
    data: dict[str, Any] = {
        "content": content,
        "created_at": now,
        "updated_at": now,
    }
    if run_hash is not None:
        data["run_hash"] = run_hash
    if experiment_id is not None:
        data["experiment_id"] = experiment_id
    _write_entity(tr, _NOTE, uid, data)
    return {"id": uid, **data}


@transactional
def get_note(tr: Transaction, uid: str) -> dict | None:
    return read_entity(tr, _NOTE, uid)


@transactional
def list_notes(tr: Transaction) -> list[dict[str, Any]]:
    return _list_entities(tr, _NOTE)


@transactional
def update_note(tr: Transaction, uid: str, **fields: Any) -> None:  # noqa: ANN401
    _update_entity(tr, _NOTE, uid, **fields)


@transactional
def delete_note(tr: Transaction, uid: str) -> None:
    _delete_entity(tr, _NOTE, uid)


@transactional
def list_notes_for_run(tr: Transaction, run_hash: str) -> list[dict[str, Any]]:
    return [n for n in _list_entities(tr, _NOTE) if n.get("run_hash") == run_hash]


@transactional
def list_notes_for_experiment(tr: Transaction, experiment_id: str) -> list[dict[str, Any]]:
    return [n for n in _list_entities(tr, _NOTE) if n.get("experiment_id") == experiment_id]
