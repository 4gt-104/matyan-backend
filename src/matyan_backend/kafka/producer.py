from __future__ import annotations

from datetime import UTC, datetime

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
from loguru import logger
from matyan_api_models.kafka import ControlEvent, IngestionMessage

from matyan_backend.config import SETTINGS

from .security import kafka_security_kwargs


class ControlEventProducer:
    """Async Kafka producer for the ``control-events`` topic."""

    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        security_kwargs = kafka_security_kwargs()
        self._producer = AIOKafkaProducer(
            bootstrap_servers=SETTINGS.kafka_bootstrap_servers,
            value_serializer=lambda v: v.encode() if isinstance(v, str) else v,
            key_serializer=lambda k: k.encode() if isinstance(k, str) else k,
            **security_kwargs,
        )
        await self._producer.start()
        logger.info("Kafka control-event producer started ({})", SETTINGS.kafka_bootstrap_servers)

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            logger.info("Kafka control-event producer stopped")

    async def publish(self, event: ControlEvent) -> None:
        if self._producer is None:
            msg = "Kafka producer not started"
            raise RuntimeError(msg)
        await self._producer.send_and_wait(
            topic=SETTINGS.kafka_control_events_topic,
            value=event.model_dump_json(),
        )


class DataIngestionProducer:
    """Async Kafka producer for the ``data-ingestion`` topic.

    Used by the backend API to publish deletion messages that must be ordered
    relative to ingestion messages for the same run.
    """

    def __init__(self) -> None:
        self._producer: AIOKafkaProducer | None = None

    async def start(self) -> None:
        security_kwargs = kafka_security_kwargs()
        self._producer = AIOKafkaProducer(
            bootstrap_servers=SETTINGS.kafka_bootstrap_servers,
            value_serializer=lambda v: v.encode() if isinstance(v, str) else v,
            key_serializer=lambda k: k.encode() if isinstance(k, str) else k,
            **security_kwargs,
        )
        await self._producer.start()
        logger.info("Kafka data-ingestion producer started ({})", SETTINGS.kafka_bootstrap_servers)

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
            logger.info("Kafka data-ingestion producer stopped")

    async def publish(self, message: IngestionMessage) -> None:
        if self._producer is None:
            msg = "Kafka data-ingestion producer not started"
            raise RuntimeError(msg)
        await self._producer.send_and_wait(
            topic=SETTINGS.kafka_data_ingestion_topic,
            key=message.run_id,
            value=message.model_dump_json(),
        )


_producer: ControlEventProducer | None = None


def get_producer() -> ControlEventProducer:
    global _producer  # noqa: PLW0603
    if _producer is None:
        _producer = ControlEventProducer()
    return _producer


_ingestion_producer: DataIngestionProducer | None = None


def get_ingestion_producer() -> DataIngestionProducer:
    global _ingestion_producer  # noqa: PLW0603
    if _ingestion_producer is None:
        _ingestion_producer = DataIngestionProducer()
    return _ingestion_producer


async def emit_delete_run(producer: DataIngestionProducer, run_id: str) -> None:
    """Publish a ``delete_run`` message to the data-ingestion topic.

    Keyed by *run_id* so it is ordered after all prior ingestion messages
    for the same run within the Kafka partition.
    """
    try:
        await producer.publish(
            IngestionMessage(
                type="delete_run",
                run_id=run_id,
                timestamp=datetime.now(UTC),
                payload={},
            ),
        )
    except KafkaError:
        logger.exception("Failed to publish delete_run for run {}", run_id)


async def emit_control_event(producer: ControlEventProducer, event_type: str, **payload: object) -> None:
    """Convenience helper used by API endpoints to publish a control event.

    Logs and swallows Kafka failures so that the HTTP response is not
    affected — the FDB write already succeeded and side effects are
    eventually consistent.
    """  # noqa: D401
    try:
        await producer.publish(
            ControlEvent(
                type=event_type,
                timestamp=datetime.now(UTC),
                payload=dict(payload),
            ),
        )
    except KafkaError:
        logger.exception("Failed to publish control event", event_type=event_type)
