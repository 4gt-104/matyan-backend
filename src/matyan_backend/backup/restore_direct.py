"""Direct restore: write backup data straight into FDB + S3."""

from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import click
from matyan_api_models.backup import BackupManifest
from tqdm import tqdm

from matyan_backend.config import SETTINGS
from matyan_backend.fdb_types import transactional
from matyan_backend.storage import encoding, entities, indexes, runs, sequences
from matyan_backend.storage.fdb_client import get_directories
from matyan_backend.storage.indexes import clear_run_tombstone

from .export_blobs import _make_gcs_client, _make_s3_client

if TYPE_CHECKING:
    from pathlib import Path

    from google.cloud import storage
    from types_boto3_s3 import S3Client

    from matyan_backend.fdb_types import Database, Transaction

_SEQ_BATCH_SIZE = 500
_S3_UPLOAD_WORKERS = 8


def _load_json(path: Path) -> Any:  # noqa: ANN401
    return json.loads(path.read_text())


def _restore_entities(db: Database, backup_dir: Path) -> None:
    entities_path = backup_dir / "entities.jsonl"
    if not entities_path.exists():
        click.echo("  No entities.jsonl found, skipping.")
        return

    counts: dict[str, int] = {}
    with entities_path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            etype = record["entity_type"]
            data = record["data"]
            counts[etype] = counts.get(etype, 0) + 1
            _restore_entity(db, etype, data)

    for etype, count in counts.items():
        if count:
            click.echo(f"    {etype}: {count}")


def _restore_experiment(db: Database, data: dict) -> None:
    existing = entities.get_experiment_by_name(db, data["name"])
    if existing is None:
        with contextlib.suppress(ValueError):
            entities.create_experiment(db, data["name"], description=data.get("description"))


def _restore_tag(db: Database, data: dict) -> None:
    existing = entities.get_tag_by_name(db, data["name"])
    if existing is None:
        with contextlib.suppress(ValueError):
            entities.create_tag(
                db,
                data["name"],
                color=data.get("color"),
                description=data.get("description"),
            )


def _restore_entity(db: Database, entity_type: str, data: dict) -> None:
    """Restore a single entity, skipping if it already exists."""
    if entity_type == "experiment":
        _restore_experiment(db, data)
    elif entity_type == "tag":
        _restore_tag(db, data)
    elif entity_type == "dashboard":
        entities.create_dashboard(
            db,
            data.get("name", ""),
            description=data.get("description"),
            app_id=data.get("app_id"),
        )
    elif entity_type == "dashboard_app":
        entities.create_dashboard_app(
            db,
            data.get("type", ""),
            data.get("state"),
            dashboard_id=data.get("dashboard_id"),
        )
    elif entity_type == "report":
        entities.create_report(
            db,
            data.get("name", ""),
            code=data.get("code"),
            description=data.get("description"),
        )
    elif entity_type == "note":
        entities.create_note(
            db,
            data.get("content", ""),
            run_hash=data.get("run_hash"),
            experiment_id=data.get("experiment_id"),
        )


@transactional
def _write_contexts_and_traces(
    tr: Transaction,
    run_hash: str,
    contexts: dict[str, dict],
    traces: list[dict],
) -> None:
    """Write all contexts and trace metadata in a single FDB transaction."""
    rd = get_directories().runs
    for ctx_id_str, ctx_dict in contexts.items():
        tr[rd.pack((run_hash, "contexts", int(ctx_id_str)))] = encoding.encode_value(ctx_dict)
    for trace in traces:
        base = (run_hash, "traces", trace["context_id"], trace["name"])
        tr[rd.pack((*base, "dtype"))] = encoding.encode_value(trace.get("dtype", "float"))
        if trace.get("last") is not None:
            tr[rd.pack((*base, "last"))] = encoding.encode_value(trace["last"])
        if trace.get("last_step") is not None:
            tr[rd.pack((*base, "last_step"))] = encoding.encode_value(trace["last_step"])


def _restore_run(db: Database, run_hash: str, run_dir: Path) -> int:
    """Restore a single run from its backup directory. Returns sequence record count."""
    meta = _load_json(run_dir / "run.json")
    attrs = _load_json(run_dir / "attrs.json")
    traces = _load_json(run_dir / "traces.json")
    contexts = _load_json(run_dir / "contexts.json")

    clear_run_tombstone(db, run_hash)

    existing = runs.get_run(db, run_hash)
    if existing is None:
        runs.create_run(
            db,
            run_hash,
            name=meta.get("name"),
            experiment_id=meta.get("experiment_id"),
            description=meta.get("description"),
        )

    meta_fields: dict[str, Any] = {}
    for field in ("is_archived", "active", "finalized_at", "client_start_ts", "duration"):
        if field in meta and meta[field] is not None:
            meta_fields[field] = meta[field]
    if meta_fields:
        runs.update_run_meta(db, run_hash, **meta_fields)

    if attrs:
        runs.set_run_attrs(db, run_hash, (), attrs)

    _write_contexts_and_traces(db, run_hash, contexts, traces)

    seq_count = _restore_sequences(db, run_hash, run_dir)
    _restore_run_associations(db, run_hash, meta)
    return seq_count


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _flush_seq_buffer(
    db: Database,
    run_hash: str,
    buffer: dict[tuple[int, str], list[dict]],
) -> None:
    """Flush buffered sequence records via write_sequence_batch."""
    for (ctx_id, name), steps_data in buffer.items():
        sequences.write_sequence_batch(db, run_hash, ctx_id, name, steps_data)


def _restore_sequences(db: Database, run_hash: str, run_dir: Path) -> int:
    """Write sequence data from sequences.jsonl in batched transactions."""
    seq_path = run_dir / "sequences.jsonl"
    if not seq_path.exists():
        return 0

    total = _count_lines(seq_path)
    count = 0
    buffer: dict[tuple[int, str], list[dict]] = defaultdict(list)
    buffered = 0

    with seq_path.open() as f:
        for raw_line in tqdm(f, total=total, desc="    sequences", unit="rec", leave=False):
            stripped = raw_line.strip()
            if not stripped:
                continue
            rec = json.loads(stripped)
            key = (rec["ctx_id"], rec["name"])
            entry: dict[str, Any] = {"step": rec["step"], "value": rec["value"]}
            if rec.get("epoch") is not None:
                entry["epoch"] = rec["epoch"]
            if rec.get("time") is not None:
                entry["timestamp"] = rec["time"]
            buffer[key].append(entry)
            buffered += 1
            count += 1

            if buffered >= _SEQ_BATCH_SIZE:
                _flush_seq_buffer(db, run_hash, buffer)
                buffer.clear()
                buffered = 0

    if buffered:
        _flush_seq_buffer(db, run_hash, buffer)

    return count


def _restore_run_associations(db: Database, run_hash: str, meta: dict) -> None:
    """Re-establish experiment and tag associations for a restored run."""
    exp_id = meta.get("experiment_id")
    if exp_id:
        exp = entities.get_experiment(db, exp_id)
        if exp:
            entities.set_run_experiment(db, run_hash, exp_id)
            runs.set_run_experiment(db, run_hash, exp_id)


def _upload_blobs(
    run_hash: str,
    run_dir: Path,
    s3_client: S3Client | None = None,
    gcs_client: storage.Client | None = None,
) -> tuple[int, int]:
    """Upload blob files from backup back to S3/GCS in parallel. Returns (count, bytes)."""
    blobs_dir = run_dir / "blobs"
    if not blobs_dir.exists():
        return 0, 0

    blob_files = [p for p in blobs_dir.rglob("*") if p.is_file()]
    if not blob_files:
        return 0, 0

    if SETTINGS.blob_backend_type == "gcs":
        client = gcs_client or _make_gcs_client()
        bucket = client.bucket(SETTINGS.gcs_bucket)

        def _upload_one(blob_path: Path) -> int:
            relative = blob_path.relative_to(blobs_dir)
            s3_key = f"{run_hash}/{relative}"
            data = blob_path.read_bytes()
            bucket.blob(s3_key).upload_from_string(data)
            return len(data)

    else:
        client = s3_client or _make_s3_client()
        bucket_name = SETTINGS.s3_bucket

        def _upload_one(blob_path: Path) -> int:
            relative = blob_path.relative_to(blobs_dir)
            s3_key = f"{run_hash}/{relative}"
            data = blob_path.read_bytes()
            client.put_object(Bucket=bucket_name, Key=s3_key, Body=data)
            return len(data)

    count = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=_S3_UPLOAD_WORKERS) as pool:
        futures = {pool.submit(_upload_one, p): p for p in blob_files}
        for fut in tqdm(as_completed(futures), total=len(blob_files), desc="    blob upload", unit="blob", leave=False):
            total_bytes += fut.result()
            count += 1

    return count, total_bytes


def restore_direct(
    db: Database,
    backup_dir: Path,
    *,
    skip_entities: bool = False,
    skip_blobs: bool = False,
    dry_run: bool = False,
) -> None:
    """Restore a backup archive directly into FDB + S3/GCS."""
    manifest = BackupManifest.read(backup_dir)
    errors = manifest.validate(backup_dir)
    if errors:
        for err in errors:
            click.echo(f"  ERROR: {err}", err=True)
        msg = "Backup validation failed"
        raise click.ClickException(msg)

    if dry_run:
        click.echo(f"Dry run: {manifest.run_count} run(s), {manifest.blob_count} blob(s)")
        click.echo("Validation passed. No data written.")
        return

    if not skip_entities:
        click.echo("Restoring entities...")
        _restore_entities(db, backup_dir)

    s3_client = _make_s3_client() if not skip_blobs and SETTINGS.blob_backend_type == "s3" else None
    gcs_client = _make_gcs_client() if not skip_blobs and SETTINGS.blob_backend_type == "gcs" else None
    total_seq = 0
    total_blobs = 0
    total_blob_bytes = 0

    hashes = manifest.run_hashes
    run_bar = tqdm(hashes, desc="Restoring runs", unit="run")
    for rh in run_bar:
        run_bar.set_postfix_str(rh[:12])
        run_dir = backup_dir / "runs" / rh
        if not run_dir.is_dir():
            tqdm.write(f"  SKIP {rh} (directory missing)")
            continue

        seq_count = _restore_run(db, rh, run_dir)
        total_seq += seq_count

        if not skip_blobs and (s3_client is not None or gcs_client is not None):
            bc, bb = _upload_blobs(rh, run_dir, s3_client=s3_client, gcs_client=gcs_client)
            total_blobs += bc
            total_blob_bytes += bb

    click.echo("Rebuilding indexes...")
    indexed, ghosts = indexes.rebuild_indexes(db)
    click.echo(f"  Indexed {indexed} run(s), cleaned {ghosts} ghost(s)")

    click.echo(
        f"\nRestore complete: {len(hashes)} run(s), "
        f"{total_seq} sequence records, "
        f"{total_blobs} blob(s) ({total_blob_bytes:,} bytes)",
    )
