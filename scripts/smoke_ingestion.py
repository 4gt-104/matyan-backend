#!/usr/bin/env python3
"""End-to-end smoke test for the ingestion worker.

1. Produces test messages to Kafka ``data-ingestion`` topic.
2. Runs the IngestionWorker for a few seconds.
3. Verifies the run exists in FDB with correct metadata, metrics, and hparams.

Requirements:
    - FoundationDB running
    - Kafka running on localhost:9092
    - ``data-ingestion`` topic exists

Usage:
    cd matyan-backend
    python scripts/smoke_ingestion.py
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import time
from datetime import UTC, datetime

from aiokafka import AIOKafkaProducer
from matyan_api_models.context import context_to_id

from matyan_backend.storage import runs, sequences
from matyan_backend.storage.fdb_client import ensure_directories, get_db, init_fdb
from matyan_backend.workers.ingestion import IngestionWorker

# ruff: noqa: T201


async def produce_test_messages() -> str:
    """Publish a set of test messages to Kafka and return the run ID used."""
    run_id = f"smoke-ingest-{int(time.time())}"
    topic = "data-ingestion"

    producer = AIOKafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: v.encode(),
        key_serializer=lambda k: k.encode() if k else None,
    )
    await producer.start()

    try:
        messages = [
            {
                "type": "create_run",
                "run_id": run_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "payload": {},
            },
            {
                "type": "log_metric",
                "run_id": run_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "payload": {
                    "name": "loss",
                    "value": 2.5,
                    "step": 0,
                    "context": {"subset": "train"},
                    "dtype": "float",
                },
            },
            {
                "type": "log_metric",
                "run_id": run_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "payload": {
                    "name": "loss",
                    "value": 1.8,
                    "step": 1,
                    "context": {"subset": "train"},
                    "dtype": "float",
                },
            },
            {
                "type": "log_metric",
                "run_id": run_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "payload": {
                    "name": "accuracy",
                    "value": 0.72,
                    "step": 0,
                    "context": {"subset": "val"},
                    "dtype": "float",
                },
            },
            {
                "type": "log_hparams",
                "run_id": run_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "payload": {
                    "value": {"lr": 0.001, "batch_size": 32, "optimizer": "adam"},
                },
            },
            {
                "type": "blob_ref",
                "run_id": run_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "payload": {
                    "artifact_path": "images/sample.png",
                    "s3_key": f"artifacts/{run_id}/images/sample.png",
                    "content_type": "image/png",
                },
            },
            {
                "type": "finish_run",
                "run_id": run_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "payload": {},
            },
        ]

        for msg in messages:
            await producer.send_and_wait(
                topic,
                value=json.dumps(msg),
                key=run_id,
            )
        print(f"Produced {len(messages)} messages for run {run_id}")
    finally:
        await producer.stop()

    return run_id


async def run_worker_briefly(seconds: int = 5) -> None:
    """Start the IngestionWorker and stop it after *seconds*."""
    worker = IngestionWorker()
    task = asyncio.create_task(worker.start())
    await asyncio.sleep(seconds)
    await worker.stop()
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def verify_fdb(run_id: str) -> bool:  # noqa: PLR0912
    """Check that the ingested data is present in FDB. Returns True if all checks pass."""
    db = get_db()
    ok = True

    # 1. Run exists
    run = runs.get_run(db, run_id)
    if run is None:
        print(f"  FAIL: run {run_id} not found")
        return False
    print(f"  OK: run exists — name={run.get('name')}, active={run.get('active')}")

    # 2. Run should be finished (active=False)
    if run.get("active") is not False:
        print(f"  FAIL: run should be inactive (finished), got active={run.get('active')}")
        ok = False
    else:
        print("  OK: run is finished (active=False)")

    # 3. Hparams
    hparams = runs.get_run_attrs(db, run_id, ("hparams",))
    if hparams and hparams.get("lr") == 0.001:
        print(f"  OK: hparams present — {hparams}")
    else:
        print(f"  FAIL: hparams missing or wrong — {hparams}")
        ok = False

    # 4. Metrics — loss (train context)
    train_ctx_id = context_to_id({"subset": "train"})
    loss_data = sequences.read_sequence(db, run_id, train_ctx_id, "loss")
    if loss_data["steps"] == [0, 1]:
        print(f"  OK: loss metric has steps [0, 1], values={loss_data.get('val')}")
    else:
        print(f"  FAIL: loss metric steps unexpected — {loss_data}")
        ok = False

    # 5. Metrics — accuracy (val context)
    val_ctx_id = context_to_id({"subset": "val"})
    acc_data = sequences.read_sequence(db, run_id, val_ctx_id, "accuracy")
    if acc_data["steps"] == [0]:
        print(f"  OK: accuracy metric has step [0], values={acc_data.get('val')}")
    else:
        print(f"  FAIL: accuracy metric steps unexpected — {acc_data}")
        ok = False

    # 6. Blob ref
    blob = runs.get_run_attrs(db, run_id, ("__blobs__", "images/sample.png"))
    if blob and blob.get("s3_key"):
        print(f"  OK: blob ref present — s3_key={blob['s3_key']}")
    else:
        print(f"  FAIL: blob ref missing — {blob}")
        ok = False

    # 7. Trace info
    traces = runs.get_run_traces_info(db, run_id)
    trace_names = {t["name"] for t in traces}
    if {"loss", "accuracy"} <= trace_names:
        print(f"  OK: trace info present — names={trace_names}")
    else:
        print(f"  FAIL: trace info incomplete — names={trace_names}")
        ok = False

    return ok


def cleanup(run_id: str) -> None:

    db = get_db()
    runs.delete_run(db, run_id)
    print(f"Cleaned up run {run_id}")


async def main() -> None:

    db = init_fdb()
    ensure_directories(db)

    print("=== Step 1: Produce test messages to Kafka ===")
    run_id = await produce_test_messages()

    print("\n=== Step 2: Run ingestion worker for 5 seconds ===")
    await run_worker_briefly(5)

    print(f"\n=== Step 3: Verify FDB state for run {run_id} ===")
    passed = verify_fdb(run_id)

    print("\n=== Step 4: Cleanup ===")
    cleanup(run_id)

    if passed:
        print("\n ALL CHECKS PASSED")
    else:
        print("\n SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
