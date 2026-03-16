"""Tests for kafka/producer.py — ControlEventProducer and emit_control_event."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from aiokafka.errors import KafkaError
from matyan_api_models.kafka import ControlEvent

from matyan_backend.kafka.producer import ControlEventProducer, emit_control_event, get_producer


class TestControlEventProducer:
    def test_publish_raises_when_not_started(self) -> None:
        producer = ControlEventProducer()
        event = ControlEvent(type="test", timestamp=datetime.now(UTC), payload={})
        with pytest.raises(RuntimeError, match="Kafka producer not started"):
            asyncio.run(producer.publish(event))


class TestGetProducer:
    def test_returns_singleton(self) -> None:
        with patch("matyan_backend.kafka.producer._producer", None):
            p1 = get_producer()
            p2 = get_producer()
            assert p1 is p2


class TestEmitControlEvent:
    def test_swallows_kafka_error(self) -> None:
        producer = AsyncMock(spec=ControlEventProducer)
        producer.publish.side_effect = KafkaError("connection lost")

        asyncio.run(emit_control_event(producer, "run_deleted", run_id="r1"))
        producer.publish.assert_called_once()

    def test_success(self) -> None:
        producer = AsyncMock(spec=ControlEventProducer)
        asyncio.run(emit_control_event(producer, "run_deleted", run_id="r1"))
        producer.publish.assert_called_once()
