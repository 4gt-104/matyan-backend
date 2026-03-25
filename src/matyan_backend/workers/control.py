"""Control worker: consumes from the Kafka ``control-events`` topic and
executes async side effects (S3 blob cleanup, cascade operations, etc.).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

import boto3
from aiokafka import AIOKafkaConsumer
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from loguru import logger
from matyan_api_models.kafka import ControlEvent
from pydantic import ValidationError

from matyan_backend.config import SETTINGS
from matyan_backend.fdb_types import FDBError
from matyan_backend.kafka.security import kafka_security_kwargs
from matyan_backend.storage.fdb_client import ensure_directories, init_fdb
from matyan_backend.workers.metrics import (
    CONTROL_EVENTS_CONSUMED,
    CONTROL_EVENTS_PROCESSED,
    CONTROL_PROCESSING_DURATION,
    CONTROL_PROCESSING_ERRORS,
)

if TYPE_CHECKING:
    from types_boto3_s3.client import S3Client

    from matyan_backend.fdb_types import Database


class ControlWorker:
    def __init__(self) -> None:
        self._consumer: AIOKafkaConsumer | None = None
        self._db: Database | None = None
        self._s3: S3Client | None = None
        self._running = False

    async def start(self) -> None:
        self._db = init_fdb()
        ensure_directories(self._db)
        logger.info("FDB initialized")

        self._s3 = boto3.client(
            "s3",
            endpoint_url=SETTINGS.s3_endpoint,
            aws_access_key_id=SETTINGS.s3_access_key,
            aws_secret_access_key=SETTINGS.s3_secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name=SETTINGS.s3_region,
        )
        logger.info("S3 client initialized (endpoint={})", SETTINGS.s3_endpoint)

        security_kwargs = kafka_security_kwargs()
        self._consumer = AIOKafkaConsumer(
            SETTINGS.kafka_control_events_topic,
            bootstrap_servers=SETTINGS.kafka_bootstrap_servers,
            group_id="control-workers",
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda v: v.decode(),
            **security_kwargs,
        )
        await self._consumer.start()
        logger.info(
            "Control worker started (topic={}, group=control-workers)",
            SETTINGS.kafka_control_events_topic,
        )

        self._running = True
        try:
            await self._consume_loop()
        finally:
            await self.stop()

    async def stop(self) -> None:
        self._running = False
        if self._consumer is not None:
            await self._consumer.stop()
            logger.info("Control worker stopped")

    async def _consume_loop(self) -> None:
        assert self._consumer is not None
        async for record in self._consumer:
            CONTROL_EVENTS_CONSUMED.inc()
            try:
                event = ControlEvent.model_validate_json(record.value)
                with CONTROL_PROCESSING_DURATION.time():
                    await asyncio.to_thread(self._handle_event, event)
                await self._consumer.commit()
                CONTROL_EVENTS_PROCESSED.labels(event_type=event.type).inc()
            except ValidationError:
                logger.warning("Invalid control event, skipping", offset=record.offset, partition=record.partition)
                CONTROL_PROCESSING_ERRORS.labels(error_type="validation").inc()
            except FDBError:
                logger.exception("FDB error processing control event", offset=record.offset, partition=record.partition)
                CONTROL_PROCESSING_ERRORS.labels(error_type="fdb").inc()
            except ClientError:
                logger.exception("S3 error processing control event", offset=record.offset, partition=record.partition)
                CONTROL_PROCESSING_ERRORS.labels(error_type="s3").inc()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Unexpected error processing control event",
                    offset=record.offset,
                    partition=record.partition,
                )
                CONTROL_PROCESSING_ERRORS.labels(error_type="unknown").inc()

    def _handle_event(self, event: ControlEvent) -> None:
        db = self._db
        s3 = self._s3
        assert db is not None
        assert s3 is not None
        handler = _HANDLERS.get(event.type)
        if handler is None:
            logger.warning("Unknown control event type: {}", event.type)
            return
        handler(db, s3, event)


# ------------------------------------------------------------------
# Per-type handlers (all run synchronously inside asyncio.to_thread)
# ------------------------------------------------------------------


def delete_s3_prefix(s3: S3Client, bucket: str, prefix: str) -> int:
    """Delete all S3 objects under *prefix*. Returns count of deleted objects."""
    deleted_count = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        if not contents:
            continue
        objects = [{"Key": obj["Key"]} for obj in contents]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": objects, "Quiet": True})
        deleted_count += len(objects)
    return deleted_count


def _handle_run_deleted(db: Database, s3: S3Client, event: ControlEvent) -> None:  # noqa: ARG001
    p = event.payload
    run_id: str = p.get("run_id", "")
    if not run_id:
        logger.warning("run_deleted event missing run_id")
        return

    blob_keys: list[str] = p.get("blob_keys", [])
    bucket = SETTINGS.s3_bucket

    if blob_keys:
        objects = [{"Key": k} for k in blob_keys]
        resp = s3.delete_objects(Bucket=bucket, Delete={"Objects": objects, "Quiet": True})
        deleted_count = len(blob_keys) - len(resp.get("Errors", []))
    else:
        deleted_count = delete_s3_prefix(s3, bucket, f"{run_id}/")

    logger.info("run_deleted: cleaned up {} S3 object(s) for run {}", deleted_count, run_id)


def _handle_experiment_deleted(db: Database, s3: S3Client, event: ControlEvent) -> None:  # noqa: ARG001
    logger.info(
        "experiment_deleted: no-op (hook for future side effects), payload={}",
        event.payload,
    )


def _handle_tag_deleted(db: Database, s3: S3Client, event: ControlEvent) -> None:  # noqa: ARG001
    logger.info(
        "tag_deleted: no-op (hook for future side effects), payload={}",
        event.payload,
    )


def _handle_run_archived(db: Database, s3: S3Client, event: ControlEvent) -> None:  # noqa: ARG001
    logger.info(
        "run_archived: no-op (hook for future side effects), payload={}",
        event.payload,
    )


_HandlerFn = Callable[["Database", "S3Client", ControlEvent], None]

_HANDLERS: dict[str, _HandlerFn] = {
    "run_deleted": _handle_run_deleted,
    "experiment_deleted": _handle_experiment_deleted,
    "tag_deleted": _handle_tag_deleted,
    "run_archived": _handle_run_archived,
    "run_unarchived": _handle_run_archived,
}
