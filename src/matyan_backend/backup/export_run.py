"""Export a single run's data from FDB to the backup archive directory."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from matyan_backend.storage import runs, sequences

if TYPE_CHECKING:
    from pathlib import Path

    from matyan_backend.fdb_types import Database


def _json_default(obj: object) -> str:
    """Handle non-serialisable types in JSON output."""
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, datetime):
        return obj.isoformat()
    msg = f"Object of type {type(obj).__name__} is not JSON serializable"
    raise TypeError(msg)


def _dump_json(data: Any, path: Path) -> None:  # noqa: ANN401
    path.write_text(json.dumps(data, default=_json_default, ensure_ascii=False) + "\n")


def _read_all_sequences(db: Database, run_hash: str, traces: list[dict]) -> list[dict]:
    """Read all sequence data for a run, returning a list of record dicts.

    Each record: {ctx_id, name, step, value, epoch, time}.
    """
    records: list[dict] = []
    for trace in traces:
        ctx_id = trace["context_id"]
        name = trace["name"]
        data = sequences.read_sequence(
            db,
            run_hash,
            ctx_id,
            name,
            columns=("val", "epoch", "time"),
        )
        steps = data.get("steps", [])
        vals = data.get("val", [])
        epochs = data.get("epoch", [])
        times = data.get("time", [])
        for i, step in enumerate(steps):
            record: dict[str, Any] = {
                "ctx_id": ctx_id,
                "name": name,
                "step": step,
                "value": vals[i] if i < len(vals) else None,
            }
            epoch_val = epochs[i] if i < len(epochs) else None
            time_val = times[i] if i < len(times) else None
            if epoch_val is not None:
                record["epoch"] = epoch_val
            if time_val is not None:
                record["time"] = time_val
            records.append(record)
    return records


def export_run(db: Database, run_hash: str, backup_dir: Path) -> int:
    """Export a single run to ``backup_dir/runs/<run_hash>/``.

    Returns the number of sequence records written.
    """
    run_dir = backup_dir / "runs" / run_hash
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = runs.get_run_meta(db, run_hash)
    _dump_json(meta, run_dir / "run.json")

    attrs = runs.get_run_attrs(db, run_hash)
    _dump_json(attrs if attrs is not None else {}, run_dir / "attrs.json")

    traces = runs.get_run_traces_info(db, run_hash)
    _dump_json(traces, run_dir / "traces.json")

    contexts = runs.get_all_contexts(db, run_hash)
    contexts_serializable = {str(k): v for k, v in contexts.items()}
    _dump_json(contexts_serializable, run_dir / "contexts.json")

    records = _read_all_sequences(db, run_hash, traces)
    seq_path = run_dir / "sequences.jsonl"
    with seq_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=_json_default, ensure_ascii=False) + "\n")

    return len(records)
