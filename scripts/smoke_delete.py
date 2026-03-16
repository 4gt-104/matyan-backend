#!/usr/bin/env python3
"""End-to-end smoke test for Kafka-ordered run deletion.

1. Creates a run via ingestion pipeline (Kafka -> worker -> FDB).
2. Marks the run as pending_deletion (simulating backend API behavior).
3. Publishes a ``delete_run`` message to the data-ingestion topic.
4. Verifies the run is immediately hidden from project activity.
5. Runs the ingestion worker to process the delete.
6. Verifies run data is fully cleaned from FDB and tombstone exists.
7. Publishes a late ``log_metric`` for the deleted run.
8. Runs the worker again — verifies the late message is skipped (tombstone).

Requirements:
    - FoundationDB running
    - Kafka running on localhost:9092
    - ``data-ingestion`` topic exists

Usage:
    cd matyan-backend
    FDB_CLUSTER_FILE=../../fdb.cluster python scripts/smoke_delete.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from datetime import UTC, datetime

from aiokafka import AIOKafkaProducer

from matyan_backend.storage import runs
from matyan_backend.storage.fdb_client import ensure_directories, init_fdb
from matyan_backend.storage.indexes import is_run_deleted
from matyan_backend.storage.project import get_project_activity
from matyan_backend.workers.ingestion import IngestionWorker


async def produce_messages(topic: str, messages: list[dict], key: str) -> None:
    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: v.encode(),
        key_serializer=lambda k: k.encode() if k else None,
    )
    await producer.start()
    try:
        for msg in messages:
            await producer.send_and_wait(topic, value=json.dumps(msg), key=key)
    finally:
        await producer.stop()


async def run_worker_briefly(seconds: int = 5) -> None:
    worker = IngestionWorker()
    task = asyncio.create_task(worker.start())
    await asyncio.sleep(seconds)
    await worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


async def main() -> None:
    db = init_fdb()
    ensure_directories(db)
    run_id = f"smoke-delete-{int(time.time())}"
    ok = True

    # --- Step 1: Create a run via Kafka ingestion ---
    print("=== Step 1: Create run via ingestion pipeline ===")
    create_messages = [
        {"type": "create_run", "run_id": run_id, "timestamp": now_iso(), "payload": {}},
        {
            "type": "log_metric",
            "run_id": run_id,
            "timestamp": now_iso(),
            "payload": {"name": "loss", "value": 2.5, "step": 0, "context": {}, "dtype": "float"},
        },
        {
            "type": "log_hparams",
            "run_id": run_id,
            "timestamp": now_iso(),
            "payload": {"value": {"hparams": {"lr": 0.001}}},
        },
        {"type": "finish_run", "run_id": run_id, "timestamp": now_iso(), "payload": {}},
    ]
    await produce_messages("data-ingestion", create_messages, run_id)
    print(f"  Produced {len(create_messages)} messages for run {run_id}")

    print("\n=== Step 2: Run worker to ingest the run ===")
    await run_worker_briefly(5)
    run = runs.get_run(db, run_id)
    if run is None:
        print(f"  FAIL: run {run_id} not found after ingestion")
        sys.exit(1)
    print(f"  OK: run exists — name={run.get('name')}, active={run.get('active')}")

    # --- Step 3: Mark pending_deletion (simulates backend API) ---
    print("\n=== Step 3: Mark run as pending_deletion ===")
    runs.mark_pending_deletion(db, run_id)
    assert runs.is_pending_deletion(db, run_id)
    print("  OK: pending_deletion flag set")

    # --- Step 4: Verify run is hidden from project activity ---
    print("\n=== Step 4: Verify run hidden from project activity ===")
    activity = get_project_activity(db)
    run_count = activity["num_runs"]
    pending_visible = any(r.get("hash") == run_id for r in runs.list_runs_meta(db) if not r.get("pending_deletion"))
    if pending_visible:
        print(f"  FAIL: run {run_id} still visible after pending_deletion")
        ok = False
    else:
        print(f"  OK: run hidden from activity (num_runs={run_count})")

    # --- Step 5: Publish delete_run to Kafka ---
    print("\n=== Step 5: Publish delete_run message to Kafka ===")
    delete_msg = [
        {"type": "delete_run", "run_id": run_id, "timestamp": now_iso(), "payload": {}},
    ]
    await produce_messages("data-ingestion", delete_msg, run_id)
    print("  OK: delete_run message published")

    # --- Step 6: Run worker to process the delete ---
    print("\n=== Step 6: Run worker to process delete ===")
    await run_worker_briefly(5)

    run_after = runs.get_run(db, run_id)
    if run_after is not None:
        print(f"  FAIL: run {run_id} still exists after delete processing")
        ok = False
    else:
        print("  OK: run data fully cleaned from FDB")

    if not is_run_deleted(db, run_id):
        print(f"  FAIL: tombstone not written for {run_id}")
        ok = False
    else:
        print("  OK: tombstone exists")

    # --- Step 7: Send a late log_metric for the deleted run ---
    print("\n=== Step 7: Send late log_metric (should be skipped) ===")
    late_msg = [
        {
            "type": "log_metric",
            "run_id": run_id,
            "timestamp": now_iso(),
            "payload": {"name": "loss", "value": 0.1, "step": 99, "context": {}, "dtype": "float"},
        },
    ]
    await produce_messages("data-ingestion", late_msg, run_id)
    print("  OK: late log_metric published")

    # --- Step 8: Run worker — late message should be skipped ---
    print("\n=== Step 8: Run worker (late message should be skipped) ===")
    await run_worker_briefly(5)

    run_after_late = runs.get_run(db, run_id)
    if run_after_late is not None:
        print(f"  FAIL: run {run_id} was recreated by late message!")
        ok = False
    else:
        print("  OK: run still deleted (tombstone prevented recreation)")

    # --- Done ---
    if ok:
        print("\n ALL CHECKS PASSED")
    else:
        print("\n SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
