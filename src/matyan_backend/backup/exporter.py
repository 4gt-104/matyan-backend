"""Backup orchestrator: selects runs, exports data, generates manifest."""

from __future__ import annotations

import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
from matyan_api_models.backup import BackupManifest

from matyan_backend.storage import indexes

from .export_blobs import _make_s3_client, export_blobs_for_run
from .export_entities import export_entities
from .export_run import export_run

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database


def _resolve_run_hashes(
    db: Database,
    *,
    run_hashes: list[str] | None = None,
    experiment: str | None = None,
    since: str | None = None,
) -> list[str]:
    """Determine which runs to back up based on filters."""
    if run_hashes:
        return run_hashes

    if experiment:
        return indexes.lookup_by_experiment(db, experiment)

    if since:
        ts = datetime.fromisoformat(since).timestamp()
        return indexes.lookup_by_created_at(db, start=ts)

    return indexes.lookup_all_run_hashes(db)


def run_backup(
    db: Database,
    output_path: str,
    *,
    run_hashes: list[str] | None = None,
    experiment: str | None = None,
    since: str | None = None,
    include_blobs: bool = True,
    compress: bool = False,
) -> Path:
    """Execute a full backup and return the backup directory (or archive) path."""
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    backup_dir = Path(output_path) / f"matyan-backup-{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    (backup_dir / "runs").mkdir(exist_ok=True)

    hashes = _resolve_run_hashes(db, run_hashes=run_hashes, experiment=experiment, since=since)
    if not hashes:
        click.echo("No runs matched the given filters.")
        manifest = BackupManifest(
            created_at=datetime.now(tz=UTC).isoformat(),
            include_blobs=include_blobs,
            filters=_build_filters(run_hashes, experiment, since),
        )
        manifest.write(backup_dir)
        return _finalise(backup_dir, compress)

    click.echo(f"Backing up {len(hashes)} run(s)...")

    click.echo("Exporting entities...")
    entity_counts = export_entities(db, backup_dir)
    for etype, count in entity_counts.items():
        if count:
            click.echo(f"  {etype}: {count}")

    total_seq_records = 0
    total_blob_count = 0
    total_blob_bytes = 0
    s3_client = _make_s3_client() if include_blobs else None

    for i, rh in enumerate(hashes, 1):
        click.echo(f"  [{i}/{len(hashes)}] Exporting run {rh}...")
        seq_count = export_run(db, rh, backup_dir)
        total_seq_records += seq_count

        if include_blobs and s3_client is not None:
            bc, bb = export_blobs_for_run(rh, backup_dir, s3_client=s3_client)
            total_blob_count += bc
            total_blob_bytes += bb
            if bc:
                click.echo(f"           {bc} blob(s), {bb:,} bytes")

    manifest = BackupManifest(
        created_at=datetime.now(tz=UTC).isoformat(),
        run_count=len(hashes),
        run_hashes=hashes,
        entity_counts=entity_counts,
        blob_count=total_blob_count,
        blob_bytes=total_blob_bytes,
        filters=_build_filters(run_hashes, experiment, since),
        include_blobs=include_blobs,
    )
    manifest.write(backup_dir)

    click.echo(
        f"\nBackup complete: {len(hashes)} run(s), "
        f"{total_seq_records} sequence records, "
        f"{total_blob_count} blob(s) ({total_blob_bytes:,} bytes)",
    )

    return _finalise(backup_dir, compress)


def _build_filters(
    run_hashes: list[str] | None,
    experiment: str | None,
    since: str | None,
) -> dict:
    filters: dict = {}
    if run_hashes:
        filters["runs"] = run_hashes
    if experiment:
        filters["experiment"] = experiment
    if since:
        filters["since"] = since
    return filters


def _finalise(backup_dir: Path, compress: bool) -> Path:
    if not compress:
        click.echo(f"Backup directory: {backup_dir}")
        return backup_dir

    archive_path = backup_dir.with_suffix(".tar.gz")
    click.echo(f"Compressing to {archive_path}...")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(backup_dir, arcname=backup_dir.name)
    shutil.rmtree(backup_dir)
    click.echo(f"Archive: {archive_path}")
    return archive_path
