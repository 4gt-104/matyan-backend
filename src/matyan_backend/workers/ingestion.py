"""Ingestion worker: consumes from the Kafka ``data-ingestion`` topic and
writes run data into FoundationDB via the storage layer.

Performance notes:
- Messages are fetched in batches via ``getmany()`` to reduce Kafka overhead.
- Within a batch, messages are grouped by ``run_id`` and each group shares a
  single FDB transaction (capped at ``ingest_max_messages_per_txn``).
- ``_next_step`` uses a reverse range scan — O(1) regardless of sequence size.
  FDB's read-your-own-writes ensures correctness within multi-message txns.
- ``set_context`` calls are cached in-memory so repeated writes for the same
  ``(run_hash, ctx_id)`` pair are skipped after the first one.
- Kafka offsets are committed once per batch, not per message.
- A bounded in-memory cache avoids repeated ``is_run_deleted`` FDB reads.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from loguru import logger
from matyan_api_models.context import context_to_id
from matyan_api_models.kafka import ControlEvent, IngestionMessage
from pydantic import ValidationError

from matyan_backend.config import SETTINGS
from matyan_backend.fdb_types import FDB_TRANSACTION_TOO_LARGE, FDBError, run_with_retry
from matyan_backend.kafka.security import kafka_security_kwargs
from matyan_backend.storage import entities, runs, sequences
from matyan_backend.storage.fdb_client import ensure_directories, init_fdb
from matyan_backend.storage.indexes import is_run_deleted
from matyan_backend.workers.metrics import (
    INGESTION_BATCH_DURATION,
    INGESTION_BATCH_SIZE,
    INGESTION_FDB_RETRIES,
    INGESTION_MESSAGES_CONSUMED,
    INGESTION_MESSAGES_PROCESSED,
    INGESTION_PROCESSING_ERRORS,
    INGESTION_TXN_ESTIMATED_BYTES,
)

if TYPE_CHECKING:
    from matyan_backend.fdb_types import Database, Transaction

    type FdbArg = Database | Transaction

_DELETED_CACHE_MAX = 10_000
_CONTEXT_CACHE_MAX = 50_000


class _ContextCache:
    """Bounded LRU cache of ``(run_hash, ctx_id)`` pairs already written.

    Avoids redundant ``set_context`` FDB writes — context dicts rarely change
    after the first write for a given ``(run, ctx)`` pair.
    """

    def __init__(self, maxsize: int = _CONTEXT_CACHE_MAX) -> None:
        self._data: OrderedDict[tuple[str, int], None] = OrderedDict()
        self._maxsize = maxsize

    def contains(self, run_hash: str, ctx_id: int) -> bool:
        key = (run_hash, ctx_id)
        if key in self._data:
            self._data.move_to_end(key)
            return True
        return False

    def add(self, run_hash: str, ctx_id: int) -> None:
        key = (run_hash, ctx_id)
        if key in self._data:
            self._data.move_to_end(key)
        elif len(self._data) >= self._maxsize:
            self._data.popitem(last=False)
        self._data[key] = None

    def evict_run(self, run_hash: str) -> None:
        to_remove = [k for k in self._data if k[0] == run_hash]
        for k in to_remove:
            del self._data[k]


class _DeletedRunCache:
    """Bounded LRU cache for ``is_run_deleted`` results.

    Avoids a FDB read per message for runs we've already checked.
    False-negatives (run deleted by another worker after we cached ``False``)
    are harmless -- the handler will attempt a write that is later overwritten
    or the run's data is cleaned up by the next ``delete_run`` event.
    """

    def __init__(self, maxsize: int = _DELETED_CACHE_MAX) -> None:
        self._data: OrderedDict[str, bool] = OrderedDict()
        self._maxsize = maxsize

    def get(self, run_id: str) -> bool | None:
        if run_id in self._data:
            self._data.move_to_end(run_id)
            return self._data[run_id]
        return None

    def put(self, run_id: str, deleted: bool) -> None:
        if run_id in self._data:
            self._data.move_to_end(run_id)
        elif len(self._data) >= self._maxsize:
            self._data.popitem(last=False)
        self._data[run_id] = deleted

    def mark_deleted(self, run_id: str) -> None:
        self.put(run_id, deleted=True)


class IngestionWorker:
    def __init__(self) -> None:
        self._consumer: AIOKafkaConsumer | None = None
        self._control_producer: AIOKafkaProducer | None = None
        self._db: Database | None = None
        self._running = False
        self._deleted_cache = _DeletedRunCache()
        self._context_cache = _ContextCache()

    async def start(self) -> None:
        self._db = init_fdb()
        ensure_directories(self._db)
        logger.info("FDB initialized")

        security_kwargs = kafka_security_kwargs()
        self._consumer = AIOKafkaConsumer(
            SETTINGS.kafka_data_ingestion_topic,
            bootstrap_servers=SETTINGS.kafka_bootstrap_servers,
            group_id="ingestion-workers",
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda v: v.decode(),
            **security_kwargs,
        )
        self._control_producer = AIOKafkaProducer(
            bootstrap_servers=SETTINGS.kafka_bootstrap_servers,
            value_serializer=lambda v: v.encode() if isinstance(v, str) else v,
            **security_kwargs,
        )
        await self._consumer.start()
        await self._control_producer.start()
        logger.info(
            "Ingestion worker started (topic={}, group=ingestion-workers)",
            SETTINGS.kafka_data_ingestion_topic,
        )

        self._running = True
        try:
            await self._consume_loop()
        finally:
            await self.stop()

    async def stop(self) -> None:
        self._running = False
        if self._control_producer is not None:
            await self._control_producer.stop()
        if self._consumer is not None:
            await self._consumer.stop()
            logger.info("Ingestion worker stopped")

    async def _emit_control_event(self, event_type: str, **payload: object) -> None:
        if self._control_producer is None:
            return
        event = ControlEvent(
            type=event_type,
            timestamp=datetime.now(tz=UTC),
            payload=dict(payload),
        )
        await self._control_producer.send_and_wait(
            topic=SETTINGS.kafka_control_events_topic,
            value=event.model_dump_json(),
        )

    async def _consume_loop(self) -> None:
        assert self._consumer is not None
        while self._running:
            batch = await self._consumer.getmany(
                timeout_ms=SETTINGS.ingest_batch_timeout_ms,
                max_records=SETTINGS.ingest_batch_size,
            )
            if not batch:
                continue
            records = [rec for tp_records in batch.values() for rec in tp_records]
            INGESTION_MESSAGES_CONSUMED.inc(len(records))
            INGESTION_BATCH_SIZE.observe(len(records))
            with INGESTION_BATCH_DURATION.time():
                delete_run_ids = await asyncio.to_thread(self._handle_batch, records)
            await self._consumer.commit()
            for run_id in delete_run_ids:
                await self._emit_control_event("run_deleted", run_id=run_id)

    @staticmethod
    def _parse_and_group(
        records: list,
    ) -> tuple[dict[str, list[IngestionMessage]], list[IngestionMessage]]:
        """Parse records and split into per-run groups + delete messages."""
        groups: dict[str, list[IngestionMessage]] = {}
        deletes: list[IngestionMessage] = []
        for record in records:
            try:
                msg = IngestionMessage.model_validate_json(record.value)
            except ValidationError:
                logger.warning(
                    "Invalid ingestion message, skipping",
                    offset=record.offset,
                    partition=record.partition,
                )
                continue
            if msg.type == "delete_run":
                deletes.append(msg)
            else:
                groups.setdefault(msg.run_id, []).append(msg)
        return groups, deletes

    def _handle_batch(self, records: list) -> list[str]:
        """Process a batch of Kafka records. Returns run_ids for delete_run events."""
        groups, deletes = self._parse_and_group(records)

        max_per_txn = SETTINGS.ingest_max_messages_per_txn
        max_txn_bytes = SETTINGS.ingest_max_txn_bytes
        for run_id, msgs in groups.items():
            pending: list[list[IngestionMessage]] = _chunk_messages(msgs, max_per_txn, max_txn_bytes)
            while pending:
                chunk = pending.pop(0)
                self._process_chunk(run_id, chunk, pending)

        delete_run_ids: list[str] = []
        for msg in deletes:
            try:
                self._handle_delete(msg)
                delete_run_ids.append(msg.run_id)
                INGESTION_MESSAGES_PROCESSED.labels(message_type="delete_run").inc()
            except FDBError:
                logger.exception("FDB error deleting run {}", msg.run_id)
                INGESTION_PROCESSING_ERRORS.labels(error_type="fdb").inc()
            except Exception:  # noqa: BLE001
                logger.exception("Unexpected error deleting run {}", msg.run_id)
                INGESTION_PROCESSING_ERRORS.labels(error_type="unknown").inc()
        return delete_run_ids

    def _process_chunk(
        self,
        run_id: str,
        chunk: list[IngestionMessage],
        pending: list[list[IngestionMessage]],
    ) -> None:
        """Try to commit *chunk*; on transaction-too-large, split and re-queue."""
        try:
            self._handle_run_group(run_id, chunk)
            INGESTION_TXN_ESTIMATED_BYTES.observe(sum(_estimate_message_size(m) for m in chunk))
            for m in chunk:
                INGESTION_MESSAGES_PROCESSED.labels(message_type=m.type).inc()
        except FDBError as exc:
            if exc.code == FDB_TRANSACTION_TOO_LARGE and len(chunk) > 1:
                mid = len(chunk) // 2
                logger.warning(
                    "Transaction too large for run {} ({} msgs), splitting into {} + {}",
                    run_id,
                    len(chunk),
                    mid,
                    len(chunk) - mid,
                )
                pending.insert(0, chunk[mid:])
                pending.insert(0, chunk[:mid])
                INGESTION_PROCESSING_ERRORS.labels(error_type="txn_too_large_split").inc()
            else:
                logger.exception("FDB error processing run group (run={})", run_id)
                INGESTION_PROCESSING_ERRORS.labels(error_type="fdb").inc()
        except ValidationError:
            logger.exception("Validation error processing run group (run={})", run_id)
            INGESTION_PROCESSING_ERRORS.labels(error_type="validation").inc()
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error processing run group (run={})", run_id)
            INGESTION_PROCESSING_ERRORS.labels(error_type="unknown").inc()

    def _handle_run_group(self, run_id: str, messages: list[IngestionMessage]) -> None:
        """Process all messages for a single run in one FDB transaction."""
        db = self._db
        assert db is not None

        def _txn() -> None:
            tr = db.create_transaction()

            if self._is_run_deleted(tr, run_id):
                logger.debug("Skipping {} message(s) for deleted run {}", len(messages), run_id)
                return

            for msg in messages:
                handler = _HANDLERS.get(msg.type)
                if handler is None:
                    logger.warning("Unknown message type: {}", msg.type)
                    continue
                handler(tr, msg, context_cache=self._context_cache)

            tr.commit().wait()

        run_with_retry(
            _txn,
            max_attempts=SETTINGS.fdb_retry_max_attempts,
            initial_delay=SETTINGS.fdb_retry_initial_delay_sec,
            max_delay=SETTINGS.fdb_retry_max_delay_sec,
            on_retry=INGESTION_FDB_RETRIES.inc,
        )

    def _handle_delete(self, msg: IngestionMessage) -> None:
        """Process a delete_run message in its own transaction."""
        db = self._db
        assert db is not None
        handler = _HANDLERS.get("delete_run")
        if not handler:
            return

        def _txn() -> None:
            tr = db.create_transaction()
            handler(tr, msg)
            tr.commit().wait()

        run_with_retry(
            _txn,
            max_attempts=SETTINGS.fdb_retry_max_attempts,
            initial_delay=SETTINGS.fdb_retry_initial_delay_sec,
            max_delay=SETTINGS.fdb_retry_max_delay_sec,
            on_retry=INGESTION_FDB_RETRIES.inc,
        )
        self._deleted_cache.mark_deleted(msg.run_id)
        self._context_cache.evict_run(msg.run_id)

    def _is_run_deleted(self, tr: Transaction, run_id: str) -> bool:
        """Check deletion with cache to avoid redundant FDB reads."""
        cached = self._deleted_cache.get(run_id)
        if cached is not None:
            return cached
        deleted = is_run_deleted(tr, run_id)
        self._deleted_cache.put(run_id, deleted)
        return deleted

    # ------------------------------------------------------------------
    # Per-type handlers (all run synchronously inside asyncio.to_thread)
    # ------------------------------------------------------------------


def _set_context_if_needed(
    db: FdbArg,
    run_hash: str,
    ctx_id: int,
    ctx: dict,
    cache: _ContextCache | None,
) -> None:
    """Write the context to FDB only if not already cached."""
    if cache is not None and cache.contains(run_hash, ctx_id):
        return
    runs.set_context(db, run_hash, ctx_id, ctx)
    if cache is not None:
        cache.add(run_hash, ctx_id)


def _parse_client_ts(payload: dict) -> float | None:
    raw = payload.get("client_datetime")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).timestamp()
    except (ValueError, TypeError):
        return None


def _handle_create_run(
    db: FdbArg,
    msg: IngestionMessage,
    *,
    context_cache: _ContextCache | None = None,  # noqa: ARG001
) -> None:
    force_resume = msg.payload.get("force_resume", False)
    client_ts = _parse_client_ts(msg.payload)
    if force_resume:
        existing = runs.resume_run(db, msg.run_id)
        if existing is not None:
            if client_ts is not None:
                runs.update_run_meta(db, msg.run_id, client_start_ts=client_ts)
            logger.debug("Resumed run {}", msg.run_id)
            return
    runs.create_run(db, msg.run_id)
    if client_ts is not None:
        runs.update_run_meta(db, msg.run_id, client_start_ts=client_ts)
    logger.debug("Created run {}", msg.run_id)


def _handle_log_metric(
    db: FdbArg,
    msg: IngestionMessage,
    *,
    context_cache: _ContextCache | None = None,
) -> None:
    p = msg.payload
    ctx = p.get("context") or {}
    ctx_id = context_to_id(ctx)
    name: str = p["name"]
    value = p["value"]
    step: int | None = p.get("step")
    epoch: int | None = p.get("epoch")
    dtype: str = p.get("dtype") or "float"

    if step is None:
        step = _next_step(db, msg.run_id, ctx_id, name)

    _set_context_if_needed(db, msg.run_id, ctx_id, ctx, context_cache)

    sequences.write_sequence_step(
        db,
        msg.run_id,
        ctx_id,
        name,
        step,
        value,
        epoch=epoch,
        timestamp=time.time(),
    )

    runs.set_trace_info(
        db,
        msg.run_id,
        ctx_id,
        name,
        dtype=dtype,
        last=value,
        last_step=step,
    )

    logger.debug("Logged metric {}.{} step={} for run {}", name, ctx_id, step, msg.run_id)


def _handle_log_hparams(
    db: FdbArg,
    msg: IngestionMessage,
    *,
    context_cache: _ContextCache | None = None,  # noqa: ARG001
) -> None:
    value = msg.payload.get("value", {})
    if not value or not isinstance(value, dict):
        return
    path: list[str] = []
    current = cast("dict[str, Any]", value)
    while isinstance(current, dict) and len(current) == 1:
        key = next(iter(current))
        path.append(key)
        current = current[key]
    runs.set_run_attrs(db, msg.run_id, tuple(path), current)
    logger.debug("Set attrs at {} for run {}", path, msg.run_id)


def _handle_finish_run(
    db: FdbArg,
    msg: IngestionMessage,
    *,
    context_cache: _ContextCache | None = None,  # noqa: ARG001
) -> None:
    client_finish = _parse_client_ts(msg.payload)
    meta = runs.get_run_meta(db, msg.run_id)
    client_start = meta.get("client_start_ts")

    duration: float | None = None
    if client_finish is not None and client_start is not None:
        duration = max(0.0, client_finish - client_start)

    created_at = meta.get("created_at", 0.0)
    finalized_at = (created_at + duration) if duration is not None else time.time()

    runs.update_run_meta(
        db,
        msg.run_id,
        active=False,
        finalized_at=finalized_at,
        duration=duration,
    )
    logger.debug("Finished run {} (duration={}s)", msg.run_id, duration)


def _handle_log_custom_object(
    db: FdbArg,
    msg: IngestionMessage,
    *,
    context_cache: _ContextCache | None = None,
) -> None:
    p = msg.payload
    ctx = p.get("context") or {}
    ctx_id = context_to_id(ctx)
    name: str = p["name"]
    value: dict = p["value"]
    step: int | None = p.get("step")
    epoch: int | None = p.get("epoch")
    dtype: str = p.get("dtype") or "custom"

    if step is None:
        step = _next_step(db, msg.run_id, ctx_id, name)

    _set_context_if_needed(db, msg.run_id, ctx_id, ctx, context_cache)

    sequences.write_sequence_step(
        db,
        msg.run_id,
        ctx_id,
        name,
        step,
        value,
        epoch=epoch,
        timestamp=time.time(),
    )

    runs.set_trace_info(
        db,
        msg.run_id,
        ctx_id,
        name,
        dtype=dtype,
        last=0.0,
        last_step=step,
    )

    logger.debug("Logged custom object {}.{} step={} for run {}", name, ctx_id, step, msg.run_id)


def _handle_blob_ref(
    db: FdbArg,
    msg: IngestionMessage,
    *,
    context_cache: _ContextCache | None = None,  # noqa: ARG001
) -> None:
    p = msg.payload
    artifact_path = p.get("artifact_path", "unknown")
    runs.set_run_attrs(
        db,
        msg.run_id,
        ("__blobs__", artifact_path),
        {
            "s3_key": p.get("s3_key", ""),
            "content_type": p.get("content_type", "application/octet-stream"),
        },
    )
    logger.debug("Stored blob ref {} for run {}", artifact_path, msg.run_id)


def _handle_set_run_property(
    db: FdbArg,
    msg: IngestionMessage,
    *,
    context_cache: _ContextCache | None = None,  # noqa: ARG001
) -> None:
    p = msg.payload
    run = runs.get_run(db, msg.run_id)
    if run is None:
        runs.create_run(db, msg.run_id)

    meta_updates: dict[str, object] = {}
    if "name" in p:
        meta_updates["name"] = p["name"]
    if "description" in p:
        meta_updates["description"] = p["description"]
    if "archived" in p:
        meta_updates["is_archived"] = p["archived"]
    if meta_updates:
        runs.update_run_meta(db, msg.run_id, **meta_updates)

    if "experiment" in p:
        exp_name = p["experiment"]
        exp = entities.get_experiment_by_name(db, exp_name)
        if exp is None:
            exp = entities.create_experiment(db, exp_name)
        entities.set_run_experiment(db, msg.run_id, exp["id"])
        runs.set_run_experiment(db, msg.run_id, exp["id"])

    logger.debug("Set properties {} for run {}", list(p.keys()), msg.run_id)


def _handle_add_tag(
    db: FdbArg,
    msg: IngestionMessage,
    *,
    context_cache: _ContextCache | None = None,  # noqa: ARG001
) -> None:
    tag_name = msg.payload["tag_name"]
    run = runs.get_run(db, msg.run_id)
    if run is None:
        runs.create_run(db, msg.run_id)

    tag = entities.get_tag_by_name(db, tag_name)
    if tag is None:
        tag = entities.create_tag(db, tag_name)
    entities.add_tag_to_run(db, msg.run_id, tag["id"])
    runs.add_tag_to_run(db, msg.run_id, tag["id"])
    logger.debug("Added tag {!r} to run {}", tag_name, msg.run_id)


def _handle_remove_tag(
    db: FdbArg,
    msg: IngestionMessage,
    *,
    context_cache: _ContextCache | None = None,  # noqa: ARG001
) -> None:
    tag_name = msg.payload["tag_name"]
    tag = entities.get_tag_by_name(db, tag_name)
    if tag is None:
        logger.warning("Tag {!r} not found, skipping remove for run {}", tag_name, msg.run_id)
        return
    entities.remove_tag_from_run(db, msg.run_id, tag["id"])
    runs.remove_tag_from_run(db, msg.run_id, tag["id"])
    logger.debug("Removed tag {!r} from run {}", tag_name, msg.run_id)


def _handle_log_terminal_line(
    db: FdbArg,
    msg: IngestionMessage,
    *,
    context_cache: _ContextCache | None = None,
) -> None:
    p = msg.payload
    line: str = p["line"]
    step: int = p["step"]

    _set_context_if_needed(db, msg.run_id, 0, {}, context_cache)

    sequences.write_sequence_step(
        db,
        msg.run_id,
        0,
        "logs",
        step,
        line,
        timestamp=time.time(),
    )

    runs.set_trace_info(db, msg.run_id, 0, "logs", dtype="logs", last=0.0, last_step=step)
    logger.debug("Logged terminal line step={} for run {}", step, msg.run_id)


def _handle_log_record(
    db: FdbArg,
    msg: IngestionMessage,
    *,
    context_cache: _ContextCache | None = None,
) -> None:
    p = msg.payload
    record_dict = {
        "message": p["message"],
        "log_level": p["level"],
        "timestamp": p["timestamp"],
        "args": p.get("extra_args"),
        "logger_info": p.get("logger_info"),
    }

    step = _next_step(db, msg.run_id, 0, "__log_records")

    _set_context_if_needed(db, msg.run_id, 0, {}, context_cache)

    sequences.write_sequence_step(
        db,
        msg.run_id,
        0,
        "__log_records",
        step,
        record_dict,
        timestamp=time.time(),
    )

    runs.set_trace_info(db, msg.run_id, 0, "__log_records", dtype="log_records", last=0.0, last_step=step)
    logger.debug("Logged record step={} for run {}", step, msg.run_id)


def _handle_delete_run(
    db: FdbArg,
    msg: IngestionMessage,
    *,
    context_cache: _ContextCache | None = None,  # noqa: ARG001
) -> None:
    runs.delete_run(db, msg.run_id)
    logger.info("Deleted run {} (data + indexes cleared, tombstone written)", msg.run_id)


def _next_step(db: FdbArg, run_hash: str, ctx_id: int, name: str) -> int:
    """Auto-assign the next step if the client didn't provide one.

    Uses a reverse range scan — O(1) vs the previous O(N) full scan.
    """
    last = sequences.get_sequence_last_step(db, run_hash, ctx_id, name)
    return 0 if last is None else last + 1


def _chunk_messages(
    msgs: list[IngestionMessage],
    max_count: int,
    max_bytes: int,
) -> list[list[IngestionMessage]]:
    """Split *msgs* into chunks respecting both count and estimated byte limits."""
    chunks: list[list[IngestionMessage]] = []
    current: list[IngestionMessage] = []
    current_bytes = 0

    for msg in msgs:
        msg_size = _estimate_message_size(msg)
        if current and (len(current) >= max_count or current_bytes + msg_size > max_bytes):
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(msg)
        current_bytes += msg_size

    if current:
        chunks.append(current)
    return chunks


_FIXED_OVERHEAD_BYTES = 500
_METRIC_OVERHEAD_BYTES = 300

_SMALL_MSG_TYPES = frozenset(
    {
        "create_run",
        "finish_run",
        "blob_ref",
        "add_tag",
        "remove_tag",
        "log_terminal_line",
        "log_record",
        "delete_run",
    },
)


def _estimate_message_size(msg: IngestionMessage) -> int:
    """Approximate FDB write bytes for *msg*. Conservative (overestimates)."""
    if msg.type in _SMALL_MSG_TYPES:
        return _estimate_small_message(msg)

    if msg.type == "log_metric":
        return _estimate_metric_message(msg)

    if msg.type == "log_custom_object":
        data = msg.payload.get("data")
        if isinstance(data, (bytes, str)):
            return _FIXED_OVERHEAD_BYTES + len(data)
        return _FIXED_OVERHEAD_BYTES + len(str(data or ""))

    if msg.type in ("log_hparams", "set_run_property"):
        return _FIXED_OVERHEAD_BYTES + len(str(msg.payload.get("value", {})))

    return _FIXED_OVERHEAD_BYTES


def _estimate_small_message(msg: IngestionMessage) -> int:
    if msg.type == "log_terminal_line":
        return _FIXED_OVERHEAD_BYTES + len(msg.payload.get("line", "") or "")
    if msg.type == "log_record":
        return _FIXED_OVERHEAD_BYTES + len(str(msg.payload.get("message", "")))
    return _FIXED_OVERHEAD_BYTES


def _estimate_metric_message(msg: IngestionMessage) -> int:
    val = msg.payload.get("value")
    if isinstance(val, (list, tuple)):
        val_size = len(val) * 8
    elif isinstance(val, bytes):
        val_size = len(val)
    else:
        val_size = 16
    return _METRIC_OVERHEAD_BYTES + val_size


_HandlerFn = Callable[..., None]

_HANDLERS: dict[str, _HandlerFn] = {
    "create_run": _handle_create_run,
    "log_metric": _handle_log_metric,
    "log_custom_object": _handle_log_custom_object,
    "log_hparams": _handle_log_hparams,
    "finish_run": _handle_finish_run,
    "blob_ref": _handle_blob_ref,
    "set_run_property": _handle_set_run_property,
    "add_tag": _handle_add_tag,
    "remove_tag": _handle_remove_tag,
    "log_terminal_line": _handle_log_terminal_line,
    "log_record": _handle_log_record,
    "delete_run": _handle_delete_run,
}
