#!/usr/bin/env python3
"""End-to-end smoke test for the control worker.

1. Seeds a test run with blob refs in FDB.
2. Uploads fake objects to S3 under ``artifacts/{run_id}/``.
3. Deletes the run from FDB.
4. Produces a ``run_deleted`` control event to Kafka.
5. Runs the ControlWorker for a few seconds.
6. Verifies the S3 objects were deleted.
7. Cleans up.

Requirements:
    - FoundationDB running
    - Kafka running on localhost:9092
    - RustFS (S3) running on localhost:9000
    - ``control-events`` topic exists

Usage:
    cd matyan-backend
    python scripts/smoke_control.py
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types_boto3_s3.client import S3Client

    from matyan_backend.fdb_types import Database

# ruff: noqa: PLC0415


def _init_fdb() -> Database:
    from matyan_backend.storage.fdb_client import ensure_directories, init_fdb

    db = init_fdb()
    ensure_directories(db)
    return db


def _get_s3() -> S3Client:
    import boto3
    from botocore.config import Config as BotoConfig

    from matyan_backend.config import SETTINGS

    return boto3.client(
        "s3",
        endpoint_url=SETTINGS.s3_endpoint,
        aws_access_key_id=SETTINGS.s3_access_key,
        aws_secret_access_key=SETTINGS.s3_secret_key,
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _ensure_bucket(s3: S3Client) -> None:
    from botocore.exceptions import ClientError

    from matyan_backend.config import SETTINGS

    try:
        s3.head_bucket(Bucket=SETTINGS.s3_bucket)
    except ClientError:
        s3.create_bucket(Bucket=SETTINGS.s3_bucket)


def seed_run_and_upload(db: Database, s3: S3Client, run_id: str) -> list[str]:
    """Create a run in FDB, upload fake S3 objects, return the S3 keys."""
    from matyan_backend.config import SETTINGS
    from matyan_backend.storage import runs

    runs.create_run(db, run_id)

    s3_keys = [
        f"artifacts/{run_id}/images/img_0.png",
        f"artifacts/{run_id}/images/img_1.png",
        f"artifacts/{run_id}/model.pt",
    ]

    for key in s3_keys:
        s3.put_object(
            Bucket=SETTINGS.s3_bucket,
            Key=key,
            Body=io.BytesIO(b"fake-blob-content"),
        )

    for key in s3_keys:
        runs.set_run_attrs(
            db,
            run_id,
            ("__blobs__", key.split("/", 2)[2]),
            {"s3_key": key, "content_type": "application/octet-stream"},
        )

    print(f"  Seeded run {run_id} with {len(s3_keys)} S3 object(s)")
    return s3_keys


def delete_run_from_fdb(db: Database, run_id: str) -> None:
    from matyan_backend.storage import runs

    runs.delete_run(db, run_id)
    print(f"  Deleted run {run_id} from FDB")


async def produce_control_event(run_id: str) -> None:
    from aiokafka import AIOKafkaProducer

    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: v.encode(),
        key_serializer=lambda k: k.encode() if k else None,
    )
    await producer.start()
    try:
        event = {
            "type": "run_deleted",
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "payload": {"run_id": run_id, "blob_keys": []},
        }
        await producer.send_and_wait(
            "control-events",
            value=json.dumps(event),
            key=run_id,
        )
        print(f"  Published run_deleted event for {run_id}")
    finally:
        await producer.stop()


async def run_worker_briefly(seconds: int = 5) -> None:
    from matyan_backend.workers.control import ControlWorker

    worker = ControlWorker()
    task = asyncio.create_task(worker.start())
    await asyncio.sleep(seconds)
    await worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def verify_s3_cleanup(s3: S3Client, run_id: str, original_keys: list[str]) -> bool:
    from matyan_backend.config import SETTINGS

    prefix = f"artifacts/{run_id}/"
    resp = s3.list_objects_v2(Bucket=SETTINGS.s3_bucket, Prefix=prefix)
    remaining = resp.get("Contents", [])

    if not remaining:
        print(f"  OK: all {len(original_keys)} S3 object(s) deleted under {prefix}")
        return True

    remaining_keys = [obj["Key"] for obj in remaining]
    print(f"  FAIL: {len(remaining_keys)} object(s) still exist: {remaining_keys}")
    return False


def cleanup_fdb(db: Database, run_id: str) -> None:
    """Safety net: delete the run if it still exists."""
    from matyan_backend.storage import runs

    if runs.get_run(db, run_id):
        runs.delete_run(db, run_id)
        print(f"  Cleaned up leftover FDB data for {run_id}")


async def main() -> None:
    run_id = f"smoke-ctrl-{int(time.time())}"

    print("=== Step 1: Initialize FDB + S3 ===")
    db = _init_fdb()
    s3 = _get_s3()
    _ensure_bucket(s3)

    print(f"\n=== Step 2: Seed run {run_id} with S3 blobs ===")
    s3_keys = seed_run_and_upload(db, s3, run_id)

    print("\n=== Step 3: Delete run from FDB ===")
    delete_run_from_fdb(db, run_id)

    print("\n=== Step 4: Produce run_deleted control event ===")
    await produce_control_event(run_id)

    print("\n=== Step 5: Run control worker for 5 seconds ===")
    await run_worker_briefly(5)

    print(f"\n=== Step 6: Verify S3 cleanup for {run_id} ===")
    passed = verify_s3_cleanup(s3, run_id, s3_keys)

    print("\n=== Step 7: Final cleanup ===")
    cleanup_fdb(db, run_id)

    if passed:
        print("\n ALL CHECKS PASSED")
    else:
        print("\n SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
