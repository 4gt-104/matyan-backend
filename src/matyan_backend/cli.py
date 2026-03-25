import click


@click.group()
def main() -> None:
    """Matyan CLI."""


@main.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=53800, help="Server port")
def start(host: str, port: int) -> None:
    """Start the Matyan server."""
    import uvicorn  # noqa: PLC0415

    from matyan_backend.config import SETTINGS  # noqa: PLC0415

    uvicorn.run(
        "matyan_backend.app:app",
        host=host,
        port=port,
        workers=1,
        log_level=SETTINGS.log_level.lower(),
    )


@main.command(name="ingest-worker")
def ingest_worker() -> None:
    """Run the data-ingestion Kafka consumer worker."""
    import asyncio  # noqa: PLC0415

    from matyan_backend.config import SETTINGS  # noqa: PLC0415
    from matyan_backend.logging import configure_logging  # noqa: PLC0415
    from matyan_backend.workers.ingestion import IngestionWorker  # noqa: PLC0415

    configure_logging(SETTINGS.log_level)

    if SETTINGS.metrics_port > 0:
        from prometheus_client import start_http_server  # noqa: PLC0415

        start_http_server(SETTINGS.metrics_port)
        click.echo(f"Prometheus metrics server started on :{SETTINGS.metrics_port}")

    asyncio.run(IngestionWorker().start())


@main.command(name="control-worker")
def control_worker() -> None:
    """Run the control-events Kafka consumer worker."""
    import asyncio  # noqa: PLC0415

    from matyan_backend.config import SETTINGS  # noqa: PLC0415
    from matyan_backend.logging import configure_logging  # noqa: PLC0415
    from matyan_backend.workers.control import ControlWorker  # noqa: PLC0415

    configure_logging(SETTINGS.log_level)

    if SETTINGS.metrics_port > 0:
        from prometheus_client import start_http_server  # noqa: PLC0415

        start_http_server(SETTINGS.metrics_port)
        click.echo(f"Prometheus metrics server started on :{SETTINGS.metrics_port}")

    asyncio.run(ControlWorker().start())


@main.command()
def reindex() -> None:
    """Rebuild all secondary indexes and clean up ghost runs."""
    from matyan_backend.storage.fdb_client import ensure_directories, init_fdb  # noqa: PLC0415
    from matyan_backend.storage.indexes import rebuild_indexes  # noqa: PLC0415

    db = init_fdb()
    ensure_directories(db)
    count, ghost_count = rebuild_indexes(db)
    click.echo(f"Indexed {count} run(s).")
    if ghost_count:
        click.echo(f"Cleaned up {ghost_count} ghost run(s).")


@main.command()
@click.argument("output_path")
@click.option("--runs", default=None, help="Comma-separated run hashes to back up")
@click.option("--experiment", default=None, help="Back up all runs in this experiment")
@click.option("--since", default=None, help="Back up runs created after this ISO datetime")
@click.option("--include-blobs/--no-blobs", default=True, help="Include/skip S3 artifact download")
@click.option("--compress", is_flag=True, default=False, help="Produce .tar.gz archive")
def backup(
    output_path: str,
    runs: str | None,
    experiment: str | None,
    since: str | None,
    include_blobs: bool,
    compress: bool,
) -> None:
    """Back up FDB data and S3 artifacts to a portable archive."""
    from matyan_backend.backup.exporter import run_backup  # noqa: PLC0415
    from matyan_backend.storage.fdb_client import ensure_directories, init_fdb  # noqa: PLC0415

    db = init_fdb()
    ensure_directories(db)

    run_hashes = [h.strip() for h in runs.split(",") if h.strip()] if runs else None
    run_backup(
        db,
        output_path,
        run_hashes=run_hashes,
        experiment=experiment,
        since=since,
        include_blobs=include_blobs,
        compress=compress,
    )


@main.command()
@click.argument("backup_path")
@click.option("--dry-run", is_flag=True, default=False, help="Validate backup without writing")
@click.option("--skip-entities", is_flag=True, default=False, help="Skip restoring experiments/tags/dashboards")
@click.option("--skip-blobs", is_flag=True, default=False, help="Skip uploading blobs to S3")
def restore(backup_path: str, dry_run: bool, skip_entities: bool, skip_blobs: bool) -> None:
    """Restore a backup archive into FDB + S3 (direct mode)."""
    import tarfile  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from matyan_backend.backup.restore_direct import restore_direct  # noqa: PLC0415
    from matyan_backend.storage.fdb_client import ensure_directories, init_fdb  # noqa: PLC0415

    db = init_fdb()
    ensure_directories(db)

    path = Path(backup_path)

    if path.suffix == ".gz" or path.name.endswith(".tar.gz"):
        tmpdir = tempfile.mkdtemp()
        click.echo(f"Extracting archive to {tmpdir}...")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(tmpdir)  # noqa: S202
        extracted = [d for d in Path(tmpdir).iterdir() if d.is_dir()]
        if not extracted:
            msg = "Archive contains no directories"
            raise click.ClickException(msg)
        path = extracted[0]

    restore_direct(db, path, skip_entities=skip_entities, skip_blobs=skip_blobs, dry_run=dry_run)


@main.command(name="finish-stale")
@click.option("--timeout-hours", default=24, type=float, help="Finish runs active longer than this many hours")
def finish_stale(timeout_hours: float) -> None:
    """Mark runs as finished if they have been active longer than a timeout."""
    import time  # noqa: PLC0415

    from matyan_backend.storage.fdb_client import ensure_directories, init_fdb  # noqa: PLC0415
    from matyan_backend.storage.indexes import lookup_by_active  # noqa: PLC0415
    from matyan_backend.storage.runs import get_run_meta, update_run_meta  # noqa: PLC0415

    db = init_fdb()
    ensure_directories(db)

    cutoff = time.time() - timeout_hours * 3600
    active_hashes = lookup_by_active(db, True)
    finished = 0
    for run_hash in active_hashes:
        meta = get_run_meta(db, run_hash)
        created_at = meta.get("created_at", 0)
        if created_at < cutoff:
            update_run_meta(db, run_hash, active=False, finalized_at=time.time())
            finished += 1
            click.echo(f"  Finished stale run: {run_hash} (created {time.time() - created_at:.0f}s ago)")

    click.echo(f"Finished {finished} stale run(s) out of {len(active_hashes)} active.")


@main.command(name="cleanup-orphan-s3")
@click.option("--dry-run", is_flag=True, default=False, help="List tombstones and report; do not delete S3 objects")
@click.option("--limit", default=0, type=int, help="Process at most N run prefixes (0 = all)")
@click.option(
    "--lock-ttl-seconds",
    default=None,
    type=int,
    help="FDB lock TTL in seconds (0 = no lock). Default from config. Schedule via K8s CronJob or cron.",
)
def cleanup_orphan_s3(dry_run: bool, limit: int, lock_ttl_seconds: int | None) -> None:
    """Delete S3 objects for runs that have a deletion tombstone.

    Intended to be scheduled as a Kubernetes CronJob or system cron entry.
    """
    import boto3  # noqa: PLC0415
    from botocore.config import Config as BotoConfig  # noqa: PLC0415

    from matyan_backend.config import SETTINGS  # noqa: PLC0415
    from matyan_backend.jobs.lock import release, try_acquire  # noqa: PLC0415
    from matyan_backend.storage.fdb_client import ensure_directories, init_fdb  # noqa: PLC0415
    from matyan_backend.storage.indexes import list_tombstones  # noqa: PLC0415
    from matyan_backend.workers.control import delete_s3_prefix  # noqa: PLC0415

    ttl = lock_ttl_seconds if lock_ttl_seconds is not None else SETTINGS.cleanup_job_lock_ttl_seconds

    db = init_fdb()
    ensure_directories(db)

    lock_held = False
    if ttl > 0:
        lock_held = try_acquire(db, "cleanup_orphan_s3", ttl)
        if not lock_held:
            click.echo("Could not acquire lock — another instance may be running. Exiting.")
            raise SystemExit(1)

    try:
        tombstones = list_tombstones(db)
        if limit > 0:
            tombstones = tombstones[:limit]

        click.echo(f"Found {len(tombstones)} tombstone(s) to process.")

        if dry_run:
            for run_hash, ts in tombstones:
                click.echo(f"  [dry-run] Would delete S3 prefix: {run_hash}/ (deleted at {ts:.0f})")
            return

        s3 = boto3.client(
            "s3",
            endpoint_url=SETTINGS.s3_endpoint,
            aws_access_key_id=SETTINGS.s3_access_key,
            aws_secret_access_key=SETTINGS.s3_secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name=SETTINGS.s3_region,
        )

        total_deleted = 0
        for i, (run_hash, _ts) in enumerate(tombstones, 1):
            count = delete_s3_prefix(s3, SETTINGS.s3_bucket, f"{run_hash}/")
            total_deleted += count
            if i % 50 == 0 or i == len(tombstones):
                click.echo(f"  Processed {i}/{len(tombstones)} runs ({total_deleted} S3 objects deleted)")

        click.echo(f"Done. Deleted {total_deleted} S3 object(s) across {len(tombstones)} run(s).")
    finally:
        if lock_held:
            release(db, "cleanup_orphan_s3")


@main.command(name="cleanup-tombstones")
@click.option(
    "--older-than-hours",
    default=None,
    type=int,
    help="Only clear tombstones older than H hours. Default from config (168).",
)
@click.option("--dry-run", is_flag=True, default=False, help="List tombstones that would be cleared; do not remove")
@click.option(
    "--lock-ttl-seconds",
    default=None,
    type=int,
    help="FDB lock TTL in seconds (0 = no lock). Default from config. Schedule via K8s CronJob or cron.",
)
def cleanup_tombstones(older_than_hours: int | None, dry_run: bool, lock_ttl_seconds: int | None) -> None:
    """Remove old deletion tombstones from FDB.

    Intended to be scheduled as a Kubernetes CronJob or system cron entry.
    """
    import time  # noqa: PLC0415

    from matyan_backend.config import SETTINGS  # noqa: PLC0415
    from matyan_backend.jobs.lock import release, try_acquire  # noqa: PLC0415
    from matyan_backend.storage.fdb_client import ensure_directories, init_fdb  # noqa: PLC0415
    from matyan_backend.storage.indexes import clear_run_tombstone, list_tombstones  # noqa: PLC0415

    age_hours = older_than_hours if older_than_hours is not None else SETTINGS.tombstone_cleanup_older_than_hours
    ttl = lock_ttl_seconds if lock_ttl_seconds is not None else SETTINGS.cleanup_job_lock_ttl_seconds

    db = init_fdb()
    ensure_directories(db)

    lock_held = False
    if ttl > 0:
        lock_held = try_acquire(db, "cleanup_tombstones", ttl)
        if not lock_held:
            click.echo("Could not acquire lock — another instance may be running. Exiting.")
            raise SystemExit(1)

    try:
        tombstones = list_tombstones(db)
        cutoff = time.time() - age_hours * 3600
        eligible = [(rh, ts) for rh, ts in tombstones if ts < cutoff]

        click.echo(f"Found {len(tombstones)} tombstone(s), {len(eligible)} older than {age_hours}h.")

        if dry_run:
            for run_hash, ts in eligible:
                click.echo(f"  [dry-run] Would clear tombstone: {run_hash} (deleted at {ts:.0f})")
            return

        cleared = 0
        for run_hash, _ts in eligible:
            clear_run_tombstone(db, run_hash)
            cleared += 1

        click.echo(f"Done. Cleared {cleared} tombstone(s).")
    finally:
        if lock_held:
            release(db, "cleanup_tombstones")


if __name__ == "__main__":
    main()
