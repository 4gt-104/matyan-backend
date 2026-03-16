"""Tests for worker Prometheus metric definitions.

These tests verify that the metric objects are properly defined and can be
incremented/observed without errors. They do not require a running Kafka or
FDB instance.
"""

from __future__ import annotations

from prometheus_client import REGISTRY

from matyan_backend.workers.metrics import (
    CONTROL_EVENTS_CONSUMED,
    CONTROL_EVENTS_PROCESSED,
    CONTROL_PROCESSING_DURATION,
    CONTROL_PROCESSING_ERRORS,
    INGESTION_BATCH_DURATION,
    INGESTION_BATCH_SIZE,
    INGESTION_MESSAGES_CONSUMED,
    INGESTION_MESSAGES_PROCESSED,
    INGESTION_PROCESSING_ERRORS,
)


class TestIngestionWorkerMetrics:
    def test_metrics_registered(self) -> None:

        assert INGESTION_MESSAGES_CONSUMED is not None
        assert INGESTION_MESSAGES_PROCESSED is not None
        assert INGESTION_PROCESSING_ERRORS is not None
        assert INGESTION_BATCH_SIZE is not None
        assert INGESTION_BATCH_DURATION is not None

    def test_counter_increment(self) -> None:

        INGESTION_MESSAGES_CONSUMED.inc()
        INGESTION_MESSAGES_PROCESSED.labels(message_type="log_metric").inc()

    def test_histogram_observe(self) -> None:

        INGESTION_BATCH_SIZE.observe(42)
        INGESTION_BATCH_DURATION.observe(0.123)

    def test_metrics_appear_in_registry(self) -> None:
        names = {m.name for m in REGISTRY.collect()}
        assert "matyan_ingestion_messages_consumed" in names
        assert "matyan_ingestion_messages_processed" in names
        assert "matyan_ingestion_processing_errors" in names
        assert "matyan_ingestion_batch_size" in names
        assert "matyan_ingestion_batch_processing_seconds" in names


class TestControlWorkerMetrics:
    def test_metrics_registered(self) -> None:

        assert CONTROL_EVENTS_CONSUMED is not None
        assert CONTROL_EVENTS_PROCESSED is not None
        assert CONTROL_PROCESSING_ERRORS is not None
        assert CONTROL_PROCESSING_DURATION is not None

    def test_counter_increment(self) -> None:

        CONTROL_EVENTS_CONSUMED.inc()
        CONTROL_EVENTS_PROCESSED.labels(event_type="run_deleted").inc()

    def test_histogram_observe(self) -> None:

        CONTROL_PROCESSING_DURATION.observe(0.05)

    def test_metrics_appear_in_registry(self) -> None:
        names = {m.name for m in REGISTRY.collect()}
        assert "matyan_control_events_consumed" in names
        assert "matyan_control_events_processed" in names
        assert "matyan_control_processing_errors" in names
        assert "matyan_control_processing_duration_seconds" in names
