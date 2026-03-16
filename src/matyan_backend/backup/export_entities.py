"""Export structured entities (experiments, tags, dashboards, etc.) from FDB."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from matyan_backend.storage import entities

if TYPE_CHECKING:
    from pathlib import Path

    from matyan_backend.fdb_types import Database

ENTITIES_FILE = "entities.jsonl"

_ENTITY_EXPORTERS: list[tuple[str, Any]] = [
    ("experiment", entities.list_experiments),
    ("tag", entities.list_tags),
    ("dashboard", entities.list_dashboards),
    ("dashboard_app", entities.list_dashboard_apps),
    ("report", entities.list_reports),
    ("note", entities.list_notes),
]


def _json_default(obj: object) -> str:

    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, datetime):
        return obj.isoformat()
    msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)


def export_entities(db: Database, backup_dir: Path) -> dict[str, int]:
    """Export all entities to ``backup_dir/entities.jsonl``.

    Returns a dict of entity_type -> count.
    """
    counts: dict[str, int] = {}
    path = backup_dir / ENTITIES_FILE
    with path.open("w") as f:
        for entity_type, list_fn in _ENTITY_EXPORTERS:
            items = list_fn(db)
            counts[entity_type] = len(items)
            for item in items:
                line = {"entity_type": entity_type, "data": item}
                f.write(json.dumps(line, default=_json_default, ensure_ascii=False) + "\n")
    return counts
