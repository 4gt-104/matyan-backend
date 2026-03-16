"""Tests for FDB transaction size estimation, chunking, and related metrics."""

from __future__ import annotations

from matyan_api_models.kafka import IngestionMessage
from prometheus_client import REGISTRY

from matyan_backend.workers.ingestion import _chunk_messages, _estimate_message_size
from matyan_backend.workers.metrics import INGESTION_FDB_RETRIES, INGESTION_TXN_ESTIMATED_BYTES


def _msg(msg_type: str, payload: dict | None = None) -> IngestionMessage:
    return IngestionMessage(
        type=msg_type,
        run_id="run-1",
        timestamp="2025-01-01T00:00:00Z",
        payload=payload or {},
    )


class TestEstimateMessageSize:
    def test_create_run_returns_small_constant(self) -> None:
        est = _estimate_message_size(_msg("create_run"))
        assert 0 < est <= 1000

    def test_finish_run_returns_small_constant(self) -> None:
        est = _estimate_message_size(_msg("finish_run"))
        assert 0 < est <= 1000

    def test_log_metric_scalar_small(self) -> None:
        est = _estimate_message_size(_msg("log_metric", {"name": "loss", "value": 0.5}))
        assert est > 0
        assert est < 2000

    def test_log_metric_list_scales_with_length(self) -> None:
        small = _estimate_message_size(_msg("log_metric", {"name": "m", "value": [1.0] * 10}))
        large = _estimate_message_size(_msg("log_metric", {"name": "m", "value": [1.0] * 1000}))
        assert large > small

    def test_log_hparams_larger_value_larger_estimate(self) -> None:
        small = _estimate_message_size(_msg("log_hparams", {"value": {"lr": 0.01}}))
        large = _estimate_message_size(_msg("log_hparams", {"value": {f"k{i}": i for i in range(100)}}))
        assert large > small

    def test_log_terminal_line_includes_line_length(self) -> None:
        short = _estimate_message_size(_msg("log_terminal_line", {"line": "x", "step": 0}))
        long_ = _estimate_message_size(_msg("log_terminal_line", {"line": "x" * 10_000, "step": 0}))
        assert long_ > short

    def test_unknown_type_returns_fixed_overhead(self) -> None:
        est = _estimate_message_size(_msg("unknown_type_xyz"))
        assert est > 0

    def test_all_small_types_positive(self) -> None:
        for t in ("create_run", "finish_run", "blob_ref", "add_tag", "remove_tag", "delete_run"):
            est = _estimate_message_size(_msg(t))
            assert est > 0, f"{t} returned non-positive"


class TestChunkMessages:
    def test_single_chunk_when_under_limits(self) -> None:
        msgs = [_msg("create_run") for _ in range(5)]
        chunks = _chunk_messages(msgs, max_count=100, max_bytes=10_000_000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 5

    def test_splits_on_count_limit(self) -> None:
        msgs = [_msg("create_run") for _ in range(10)]
        chunks = _chunk_messages(msgs, max_count=3, max_bytes=10_000_000)
        assert len(chunks) == 4  # 3 + 3 + 3 + 1
        assert sum(len(c) for c in chunks) == 10

    def test_splits_on_byte_limit(self) -> None:
        msgs = [_msg("log_metric", {"name": "m", "value": [1.0] * 1000}) for _ in range(5)]
        per_msg = _estimate_message_size(msgs[0])
        # Set byte limit to allow exactly 2 messages
        limit = per_msg * 2 + 1
        chunks = _chunk_messages(msgs, max_count=1000, max_bytes=limit)
        assert len(chunks) == 3  # 2 + 2 + 1
        assert sum(len(c) for c in chunks) == 5

    def test_single_large_message_gets_own_chunk(self) -> None:
        big = _msg("log_metric", {"name": "m", "value": [1.0] * 10_000})
        small = _msg("create_run")
        msgs = [small, big, small]
        # Byte limit smaller than one big message — big msg still forms a solo chunk
        big_size = _estimate_message_size(big)
        limit = big_size - 1
        chunks = _chunk_messages(msgs, max_count=1000, max_bytes=limit)
        assert any(len(c) == 1 and c[0].type == "log_metric" for c in chunks)
        assert sum(len(c) for c in chunks) == 3

    def test_empty_input(self) -> None:
        chunks = _chunk_messages([], max_count=10, max_bytes=10_000)
        assert chunks == []

    def test_respects_both_limits(self) -> None:
        msgs = [_msg("log_metric", {"name": "m", "value": [1.0] * 100}) for _ in range(20)]
        per_msg = _estimate_message_size(msgs[0])
        chunks = _chunk_messages(msgs, max_count=5, max_bytes=per_msg * 3 + 1)
        # Count limit = 5, byte limit ≈ 3 messages → byte limit dominates
        for chunk in chunks:
            assert len(chunk) <= 5
            assert sum(_estimate_message_size(m) for m in chunk) <= per_msg * 3 + 1 + per_msg


class TestIngestionTxnMetrics:
    def test_txn_estimated_bytes_histogram_registered(self) -> None:
        names = {m.name for m in REGISTRY.collect()}
        assert "matyan_ingestion_txn_estimated_bytes" in names

    def test_txn_estimated_bytes_histogram_observable(self) -> None:
        INGESTION_TXN_ESTIMATED_BYTES.observe(1_000_000)
        found = False
        for metric in REGISTRY.collect():
            if metric.name == "matyan_ingestion_txn_estimated_bytes":
                for s in metric.samples:
                    if s.name == "matyan_ingestion_txn_estimated_bytes_count":
                        found = True
                        assert s.value >= 1
                        break
                break
        assert found

    def test_fdb_retries_counter_registered(self) -> None:
        names = {m.name for m in REGISTRY.collect()}
        assert "matyan_ingestion_fdb_retries" in names

    def test_fdb_retries_counter_increments(self) -> None:
        before = INGESTION_FDB_RETRIES._value.get()  # noqa: SLF001
        INGESTION_FDB_RETRIES.inc()
        after = INGESTION_FDB_RETRIES._value.get()  # noqa: SLF001
        assert after == before + 1
