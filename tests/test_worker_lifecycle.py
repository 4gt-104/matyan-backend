"""Tests for worker start/stop/consume_loop lifecycle methods and kafka producer start/stop.

All Kafka interactions are mocked.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

from botocore.exceptions import ClientError
from matyan_api_models.kafka import ControlEvent, IngestionMessage

from matyan_backend.fdb_types import FDBError
from matyan_backend.kafka.producer import ControlEventProducer
from matyan_backend.workers.control import ControlWorker
from matyan_backend.workers.ingestion import IngestionWorker, _ContextCache, _DeletedRunCache


class _AsyncIter:
    """Helper to make a mock Kafka consumer iterable (used by ControlWorker)."""

    def __init__(self, records: list) -> None:
        self._records = records
        self._index = 0

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> MagicMock:
        if self._index >= len(self._records):
            raise StopAsyncIteration
        record = self._records[self._index]
        self._index += 1
        return record


def _make_getmany_consumer(records: list, worker: IngestionWorker) -> MagicMock:
    """Build a mock consumer whose ``getmany()`` returns *records* once then stops the worker."""
    call_count = 0

    async def _getmany(**_kwargs: object) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count == 1 and records:
            return {"tp0": records}
        worker._running = False  # noqa: SLF001
        return {}

    mock_consumer = MagicMock()
    mock_consumer.getmany = _getmany
    mock_consumer.commit = AsyncMock()
    return mock_consumer


class TestIngestionWorkerLifecycle:
    def test_stop_when_consumer_none(self) -> None:
        worker = IngestionWorker()
        worker._consumer = None  # noqa: SLF001
        asyncio.run(worker.stop())
        assert not worker._running  # noqa: SLF001

    def test_stop_with_consumer(self) -> None:
        worker = IngestionWorker()
        worker._consumer = AsyncMock()  # noqa: SLF001
        asyncio.run(worker.stop())
        worker._consumer.stop.assert_called_once()  # noqa: SLF001

    def test_consume_loop_processes_batch(self) -> None:
        worker = IngestionWorker()
        worker._db = MagicMock()  # noqa: SLF001
        worker._running = True  # noqa: SLF001

        msg = IngestionMessage(
            type="create_run",
            run_id="lifecycle1",
            timestamp=datetime.now(UTC),
            payload={},
        )

        mock_record = MagicMock()
        mock_record.value = msg.model_dump_json()

        mock_consumer = _make_getmany_consumer([mock_record], worker)
        worker._consumer = mock_consumer  # noqa: SLF001

        with patch.object(worker, "_handle_batch", return_value=[]):
            asyncio.run(worker._consume_loop())  # noqa: SLF001
            mock_consumer.commit.assert_called_once()

    def test_consume_loop_skips_empty_batch(self) -> None:
        worker = IngestionWorker()
        worker._db = MagicMock()  # noqa: SLF001
        worker._running = True  # noqa: SLF001

        mock_consumer = _make_getmany_consumer([], worker)
        worker._consumer = mock_consumer  # noqa: SLF001

        asyncio.run(worker._consume_loop())  # noqa: SLF001
        mock_consumer.commit.assert_not_called()

    def test_consume_loop_emits_control_events_for_deletes(self) -> None:
        worker = IngestionWorker()
        worker._db = MagicMock()  # noqa: SLF001
        worker._running = True  # noqa: SLF001

        msg = IngestionMessage(
            type="delete_run",
            run_id="del1",
            timestamp=datetime.now(UTC),
            payload={},
        )

        mock_record = MagicMock()
        mock_record.value = msg.model_dump_json()

        mock_consumer = _make_getmany_consumer([mock_record], worker)
        worker._consumer = mock_consumer  # noqa: SLF001

        with (
            patch.object(worker, "_handle_batch", return_value=["del1"]),
            patch.object(worker, "_emit_control_event", new_callable=AsyncMock) as mock_emit,
        ):
            asyncio.run(worker._consume_loop())  # noqa: SLF001
            mock_emit.assert_called_once_with("run_deleted", run_id="del1")

    def test_handle_batch_groups_by_run_id(self) -> None:
        worker = IngestionWorker()
        worker._db = MagicMock()  # noqa: SLF001

        msgs = [
            IngestionMessage(type="create_run", run_id="b1", timestamp=datetime.now(UTC), payload={}),
            IngestionMessage(type="create_run", run_id="b2", timestamp=datetime.now(UTC), payload={}),
            IngestionMessage(
                type="log_hparams",
                run_id="b1",
                timestamp=datetime.now(UTC),
                payload={"value": {"lr": 0.01}},
            ),
        ]
        records = []
        for m in msgs:
            r = MagicMock()
            r.value = m.model_dump_json()
            r.offset = 0
            r.partition = 0
            records.append(r)

        with patch.object(worker, "_handle_run_group") as mock_group:
            delete_ids = worker._handle_batch(records)  # noqa: SLF001
            assert mock_group.call_count == 2
            call_run_ids = {call.args[0] for call in mock_group.call_args_list}
            assert call_run_ids == {"b1", "b2"}
            assert delete_ids == []

    def test_handle_batch_separates_deletes(self) -> None:
        worker = IngestionWorker()
        worker._db = MagicMock()  # noqa: SLF001

        msg = IngestionMessage(type="delete_run", run_id="del2", timestamp=datetime.now(UTC), payload={})
        record = MagicMock()
        record.value = msg.model_dump_json()
        record.offset = 0
        record.partition = 0

        with patch.object(worker, "_handle_delete") as mock_del:
            delete_ids = worker._handle_batch([record])  # noqa: SLF001
            mock_del.assert_called_once()
            assert delete_ids == ["del2"]

    def test_handle_batch_tolerates_validation_error(self) -> None:
        worker = IngestionWorker()
        worker._db = MagicMock()  # noqa: SLF001

        bad_record = MagicMock()
        bad_record.value = "not-valid-json"
        bad_record.offset = 0
        bad_record.partition = 0

        delete_ids = worker._handle_batch([bad_record])  # noqa: SLF001
        assert delete_ids == []

    def test_handle_batch_tolerates_run_group_fdb_error(self) -> None:
        worker = IngestionWorker()
        worker._db = MagicMock()  # noqa: SLF001

        msg = IngestionMessage(type="create_run", run_id="fdb_err", timestamp=datetime.now(UTC), payload={})
        record = MagicMock()
        record.value = msg.model_dump_json()
        record.offset = 0
        record.partition = 0

        with patch.object(worker, "_handle_run_group", side_effect=FDBError(1000)):
            delete_ids = worker._handle_batch([record])  # noqa: SLF001
            assert delete_ids == []

    def test_handle_batch_splits_on_transaction_too_large(self) -> None:
        """When _handle_run_group raises FDBError(2101), the chunk is split and retried."""
        worker = IngestionWorker()
        worker._db = MagicMock()  # noqa: SLF001

        msgs = [
            IngestionMessage(type="create_run", run_id="split_r", timestamp=datetime.now(UTC), payload={})
            for _ in range(4)
        ]
        records = []
        for i, m in enumerate(msgs):
            r = MagicMock()
            r.value = m.model_dump_json()
            r.offset = i
            r.partition = 0
            records.append(r)

        call_count = 0

        def _side_effect(_run_id: str, chunk: list) -> None:
            nonlocal call_count
            call_count += 1
            if len(chunk) > 2:
                raise FDBError(2101)

        with patch.object(worker, "_handle_run_group", side_effect=_side_effect):
            worker._handle_batch(records)  # noqa: SLF001

        # Original chunk of 4 → split into 2+2, each ≤ 2 → succeeds
        assert call_count >= 3  # 1 failed + 2 successful (at minimum)

    def test_handle_batch_single_msg_txn_too_large_not_split(self) -> None:
        """A single message that is too large cannot be split further — treated as FDB error."""
        worker = IngestionWorker()
        worker._db = MagicMock()  # noqa: SLF001

        msg = IngestionMessage(type="create_run", run_id="single_r", timestamp=datetime.now(UTC), payload={})
        record = MagicMock()
        record.value = msg.model_dump_json()
        record.offset = 0
        record.partition = 0

        with patch.object(worker, "_handle_run_group", side_effect=FDBError(2101)):
            delete_ids = worker._handle_batch([record])  # noqa: SLF001
            assert delete_ids == []

    def test_handle_batch_tolerates_run_group_unexpected_error(self) -> None:
        worker = IngestionWorker()
        worker._db = MagicMock()  # noqa: SLF001

        msg = IngestionMessage(type="create_run", run_id="unexp_err", timestamp=datetime.now(UTC), payload={})
        record = MagicMock()
        record.value = msg.model_dump_json()
        record.offset = 0
        record.partition = 0

        with patch.object(worker, "_handle_run_group", side_effect=RuntimeError("boom")):
            delete_ids = worker._handle_batch([record])  # noqa: SLF001
            assert delete_ids == []

    def test_handle_batch_tolerates_delete_error(self) -> None:
        worker = IngestionWorker()
        worker._db = MagicMock()  # noqa: SLF001

        msg = IngestionMessage(type="delete_run", run_id="del_err", timestamp=datetime.now(UTC), payload={})
        record = MagicMock()
        record.value = msg.model_dump_json()
        record.offset = 0
        record.partition = 0

        with patch.object(worker, "_handle_delete", side_effect=FDBError(1000)):
            delete_ids = worker._handle_batch([record])  # noqa: SLF001
            assert delete_ids == []

    def test_handle_run_group_calls_handlers(self) -> None:
        worker = IngestionWorker()
        mock_db = MagicMock()
        mock_tr = MagicMock()
        mock_db.create_transaction.return_value = mock_tr
        worker._db = mock_db  # noqa: SLF001

        messages = [
            IngestionMessage(type="create_run", run_id="rg1", timestamp=datetime.now(UTC), payload={}),
        ]

        with (
            patch.object(worker, "_is_run_deleted", return_value=False),
            patch("matyan_backend.workers.ingestion._HANDLERS", {"create_run": MagicMock()}) as mock_handlers,
        ):
            worker._handle_run_group("rg1", messages)  # noqa: SLF001
            mock_handlers["create_run"].assert_called_once()
            mock_tr.commit.return_value.wait.assert_called_once()

    def test_handle_run_group_skips_deleted(self) -> None:
        worker = IngestionWorker()
        mock_db = MagicMock()
        mock_tr = MagicMock()
        mock_db.create_transaction.return_value = mock_tr
        worker._db = mock_db  # noqa: SLF001

        messages = [
            IngestionMessage(type="create_run", run_id="rg_del", timestamp=datetime.now(UTC), payload={}),
        ]

        with patch.object(worker, "_is_run_deleted", return_value=True):
            worker._handle_run_group("rg_del", messages)  # noqa: SLF001
            mock_tr.commit.assert_not_called()


class TestControlWorkerLifecycle:
    def test_stop_when_consumer_none(self) -> None:
        worker = ControlWorker()
        worker._consumer = None  # noqa: SLF001
        asyncio.run(worker.stop())
        assert not worker._running  # noqa: SLF001

    def test_stop_with_consumer(self) -> None:
        worker = ControlWorker()
        worker._consumer = AsyncMock()  # noqa: SLF001
        asyncio.run(worker.stop())
        worker._consumer.stop.assert_called_once()  # noqa: SLF001

    def test_consume_loop_processes_event(self) -> None:
        worker = ControlWorker()
        worker._db = MagicMock()  # noqa: SLF001
        worker._s3 = MagicMock()  # noqa: SLF001

        event = ControlEvent(
            type="run_deleted",
            timestamp=datetime.now(UTC),
            payload={"run_id": "cl1"},
        )

        mock_record = MagicMock()
        mock_record.value = event.model_dump_json()

        mock_consumer = _AsyncIter([mock_record])
        mock_consumer.commit = AsyncMock()  # ty:ignore[unresolved-attribute]
        worker._consumer = mock_consumer  # type: ignore[assignment]  # noqa: SLF001

        with patch.object(worker, "_handle_event"):
            asyncio.run(worker._consume_loop())  # noqa: SLF001
            mock_consumer.commit.assert_called_once()  # ty:ignore[unresolved-attribute]

    def test_consume_loop_validation_error(self) -> None:
        worker = ControlWorker()
        worker._db = MagicMock()  # noqa: SLF001
        worker._s3 = MagicMock()  # noqa: SLF001

        mock_record = MagicMock()
        mock_record.value = "bad-json"
        mock_record.offset = 0
        mock_record.partition = 0

        mock_consumer = _AsyncIter([mock_record])
        mock_consumer.commit = AsyncMock()  # ty:ignore[unresolved-attribute]
        worker._consumer = mock_consumer  # type: ignore[assignment]  # noqa: SLF001

        asyncio.run(worker._consume_loop())  # noqa: SLF001
        mock_consumer.commit.assert_not_called()  # ty:ignore[unresolved-attribute]

    def test_consume_loop_fdb_error(self) -> None:
        worker = ControlWorker()
        worker._db = MagicMock()  # noqa: SLF001
        worker._s3 = MagicMock()  # noqa: SLF001

        event = ControlEvent(
            type="run_deleted",
            timestamp=datetime.now(UTC),
            payload={"run_id": "fdb_err"},
        )
        mock_record = MagicMock()
        mock_record.value = event.model_dump_json()
        mock_record.offset = 0
        mock_record.partition = 0

        mock_consumer = _AsyncIter([mock_record])
        mock_consumer.commit = AsyncMock()  # ty:ignore[unresolved-attribute]
        worker._consumer = mock_consumer  # type: ignore[assignment]  # noqa: SLF001

        with patch.object(worker, "_handle_event", side_effect=FDBError(1000)):
            asyncio.run(worker._consume_loop())  # noqa: SLF001

    def test_consume_loop_client_error(self) -> None:
        worker = ControlWorker()
        worker._db = MagicMock()  # noqa: SLF001
        worker._s3 = MagicMock()  # noqa: SLF001

        event = ControlEvent(
            type="run_deleted",
            timestamp=datetime.now(UTC),
            payload={"run_id": "s3_err"},
        )
        mock_record = MagicMock()
        mock_record.value = event.model_dump_json()
        mock_record.offset = 0
        mock_record.partition = 0

        mock_consumer = _AsyncIter([mock_record])
        mock_consumer.commit = AsyncMock()  # ty:ignore[unresolved-attribute]
        worker._consumer = mock_consumer  # type: ignore[assignment]  # noqa: SLF001

        with patch.object(worker, "_handle_event", side_effect=ClientError({"Error": {"Code": "500"}}, "op")):
            asyncio.run(worker._consume_loop())  # noqa: SLF001

    def test_consume_loop_unexpected_error(self) -> None:
        worker = ControlWorker()
        worker._db = MagicMock()  # noqa: SLF001
        worker._s3 = MagicMock()  # noqa: SLF001

        event = ControlEvent(
            type="run_deleted",
            timestamp=datetime.now(UTC),
            payload={"run_id": "unexp"},
        )
        mock_record = MagicMock()
        mock_record.value = event.model_dump_json()
        mock_record.offset = 0
        mock_record.partition = 0

        mock_consumer = _AsyncIter([mock_record])
        mock_consumer.commit = AsyncMock()  # ty:ignore[unresolved-attribute]
        worker._consumer = mock_consumer  # type: ignore[assignment]  # noqa: SLF001

        with patch.object(worker, "_handle_event", side_effect=RuntimeError("boom")):
            asyncio.run(worker._consume_loop())  # noqa: SLF001


class TestDeletedRunCache:
    def test_cache_miss_returns_none(self) -> None:
        cache = _DeletedRunCache(maxsize=5)
        assert cache.get("missing") is None

    def test_put_and_get(self) -> None:
        cache = _DeletedRunCache(maxsize=5)
        cache.put("r1", deleted=False)
        assert cache.get("r1") is False

    def test_mark_deleted(self) -> None:
        cache = _DeletedRunCache(maxsize=5)
        cache.put("r1", deleted=False)
        cache.mark_deleted("r1")
        assert cache.get("r1") is True

    def test_eviction_when_full(self) -> None:
        cache = _DeletedRunCache(maxsize=3)
        cache.put("a", deleted=False)
        cache.put("b", deleted=False)
        cache.put("c", deleted=False)
        cache.put("d", deleted=False)
        assert cache.get("a") is None
        assert cache.get("b") is not None

    def test_lru_order_preserved_on_get(self) -> None:
        cache = _DeletedRunCache(maxsize=3)
        cache.put("a", deleted=False)
        cache.put("b", deleted=False)
        cache.put("c", deleted=False)
        cache.get("a")
        cache.put("d", deleted=False)
        assert cache.get("a") is not None
        assert cache.get("b") is None


class TestContextCache:
    def test_miss_returns_false(self) -> None:
        cache = _ContextCache(maxsize=5)
        assert cache.contains("r1", 0) is False

    def test_add_and_contains(self) -> None:
        cache = _ContextCache(maxsize=5)
        cache.add("r1", 0)
        assert cache.contains("r1", 0) is True

    def test_different_ctx_id(self) -> None:
        cache = _ContextCache(maxsize=5)
        cache.add("r1", 0)
        assert cache.contains("r1", 1) is False

    def test_evict_run(self) -> None:
        cache = _ContextCache(maxsize=10)
        cache.add("r1", 0)
        cache.add("r1", 1)
        cache.add("r2", 0)
        cache.evict_run("r1")
        assert cache.contains("r1", 0) is False
        assert cache.contains("r1", 1) is False
        assert cache.contains("r2", 0) is True

    def test_lru_eviction(self) -> None:
        cache = _ContextCache(maxsize=3)
        cache.add("r1", 0)
        cache.add("r2", 0)
        cache.add("r3", 0)
        cache.add("r4", 0)
        assert cache.contains("r1", 0) is False
        assert cache.contains("r2", 0) is True

    def test_lru_order_preserved_on_contains(self) -> None:
        cache = _ContextCache(maxsize=3)
        cache.add("r1", 0)
        cache.add("r2", 0)
        cache.add("r3", 0)
        cache.contains("r1", 0)
        cache.add("r4", 0)
        assert cache.contains("r1", 0) is True
        assert cache.contains("r2", 0) is False


class TestIngestionWorkerStart:
    def test_start_initializes_and_consumes(self) -> None:
        worker = IngestionWorker()

        mock_consumer_instance = MagicMock()
        mock_consumer_instance.start = AsyncMock()
        mock_consumer_instance.stop = AsyncMock()
        mock_consumer_instance.commit = AsyncMock()

        async def _empty_getmany(**_kwargs: object) -> dict:
            worker._running = False  # noqa: SLF001
            return {}

        mock_consumer_instance.getmany = _empty_getmany

        mock_producer_instance = AsyncMock()

        async def _test() -> None:
            with (
                patch("matyan_backend.workers.ingestion.init_fdb", return_value=MagicMock()),
                patch("matyan_backend.workers.ingestion.ensure_directories"),
                patch("matyan_backend.workers.ingestion.AIOKafkaConsumer", return_value=mock_consumer_instance),
                patch("matyan_backend.workers.ingestion.AIOKafkaProducer", return_value=mock_producer_instance),
                patch("matyan_backend.workers.ingestion.kafka_security_kwargs", return_value={}),
            ):
                await worker.start()

        asyncio.run(_test())
        mock_consumer_instance.start.assert_called_once()
        mock_consumer_instance.stop.assert_called_once()


class TestControlWorkerStart:
    def test_start_initializes_and_consumes(self) -> None:
        worker = ControlWorker()

        mock_consumer_instance = _AsyncIter([])
        mock_consumer_instance.start = AsyncMock()  # ty:ignore[unresolved-attribute]
        mock_consumer_instance.stop = AsyncMock()  # ty:ignore[unresolved-attribute]
        mock_consumer_instance.commit = AsyncMock()  # ty:ignore[unresolved-attribute]

        async def _test() -> None:
            with (
                patch("matyan_backend.workers.control.init_fdb", return_value=MagicMock()),
                patch("matyan_backend.workers.control.ensure_directories"),
                patch("matyan_backend.workers.control.boto3") as mock_boto3,
                patch("matyan_backend.workers.control.AIOKafkaConsumer", return_value=mock_consumer_instance),
            ):
                mock_boto3.client.return_value = MagicMock()
                await worker.start()

        asyncio.run(_test())
        mock_consumer_instance.start.assert_called_once()  # ty:ignore[unresolved-attribute]
        mock_consumer_instance.stop.assert_called_once()  # ty:ignore[unresolved-attribute]


class TestKafkaProducerStartStop:
    def test_start_stop(self) -> None:
        producer = ControlEventProducer()

        mock_aioproducer = AsyncMock()

        async def _test() -> None:
            with patch("matyan_backend.kafka.producer.AIOKafkaProducer", return_value=mock_aioproducer):
                await producer.start()
                assert producer._producer is mock_aioproducer  # noqa: SLF001
                mock_aioproducer.start.assert_called_once()

                await producer.stop()
                mock_aioproducer.stop.assert_called_once()

        asyncio.run(_test())

    def test_stop_when_none(self) -> None:
        producer = ControlEventProducer()
        asyncio.run(producer.stop())

    def test_publish_sends(self) -> None:
        producer = ControlEventProducer()
        producer._producer = AsyncMock()  # noqa: SLF001

        event = ControlEvent(type="test", timestamp=datetime.now(UTC), payload={})

        asyncio.run(producer.publish(event))
        producer._producer.send_and_wait.assert_called_once()  # noqa: SLF001
