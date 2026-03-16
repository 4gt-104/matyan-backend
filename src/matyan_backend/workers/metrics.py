"""Prometheus metrics for ingestion and control workers.

Workers do not run an HTTP server by default; the CLI commands start
``prometheus_client.start_http_server`` on a configurable port so Prometheus
can scrape them.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# ---------------------------------------------------------------------------
# Ingestion worker metrics
# ---------------------------------------------------------------------------

INGESTION_MESSAGES_CONSUMED = Counter(
    "matyan_ingestion_messages_consumed_total",
    "Total Kafka messages fetched by the ingestion worker",
)

INGESTION_MESSAGES_PROCESSED = Counter(
    "matyan_ingestion_messages_processed_total",
    "Messages successfully handled by the ingestion worker",
    ["message_type"],
)

INGESTION_PROCESSING_ERRORS = Counter(
    "matyan_ingestion_processing_errors_total",
    "Errors encountered while processing ingestion messages",
    ["error_type"],
)

INGESTION_BATCH_SIZE = Histogram(
    "matyan_ingestion_batch_size",
    "Number of records in each ingestion batch",
    buckets=(1, 5, 10, 25, 50, 100, 200, 500),
)

INGESTION_BATCH_DURATION = Histogram(
    "matyan_ingestion_batch_processing_seconds",
    "Time to process each ingestion batch",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

INGESTION_TXN_ESTIMATED_BYTES = Histogram(
    "matyan_ingestion_txn_estimated_bytes",
    "Estimated FDB transaction size in bytes per committed run-group chunk",
    buckets=(100_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 10_000_000),
)

INGESTION_FDB_RETRIES = Counter(
    "matyan_ingestion_fdb_retries_total",
    "Number of retryable FDB errors encountered by the ingestion worker",
)

# ---------------------------------------------------------------------------
# Control worker metrics
# ---------------------------------------------------------------------------

CONTROL_EVENTS_CONSUMED = Counter(
    "matyan_control_events_consumed_total",
    "Total control events read from Kafka",
)

CONTROL_EVENTS_PROCESSED = Counter(
    "matyan_control_events_processed_total",
    "Control events successfully handled",
    ["event_type"],
)

CONTROL_PROCESSING_ERRORS = Counter(
    "matyan_control_processing_errors_total",
    "Errors encountered while processing control events",
    ["error_type"],
)

CONTROL_PROCESSING_DURATION = Histogram(
    "matyan_control_processing_duration_seconds",
    "Time to process each control event",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
