"""Tests that streaming/search queue sizes are driven by config settings."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import patch

from matyan_backend.config import Settings
from matyan_backend.storage import runs, sequences

from .conftest import BASE

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

    from matyan_backend.fdb_types import Database


def _seed_run(db: Database, run_hash: str) -> None:
    runs.create_run(db, run_hash)
    runs.set_context(db, run_hash, 0, {})
    runs.set_run_attrs(db, run_hash, (), {"hparams": {"lr": 0.01}})
    runs.set_trace_info(db, run_hash, 0, "loss", dtype="float", last=0.5, last_step=2)
    for i in range(3):
        sequences.write_sequence_step(db, run_hash, 0, "loss", i, 1.0 - i * 0.3, epoch=0, timestamp=1000.0 + i)


def _seed_text_run(db: Database, run_hash: str) -> None:
    runs.create_run(db, run_hash)
    runs.set_context(db, run_hash, 0, {})
    runs.set_trace_info(db, run_hash, 0, "texts", dtype="text", last=0.0, last_step=1)
    for i in range(2):
        sequences.write_sequence_step(db, run_hash, 0, "texts", i, {"data": f"txt_{i}"})


def _make_queue_spy(captured: list[int]) -> type:
    """Return a Queue subclass that records *maxsize* on construction."""
    _real = asyncio.Queue

    class _SpyQueue(_real):  # type: ignore[misc]
        def __init__(self, maxsize: int = 0) -> None:
            captured.append(maxsize)
            super().__init__(maxsize=maxsize)

    return _SpyQueue


class TestConfigDefaults:
    def test_default_run_search_queue_maxsize(self) -> None:
        assert Settings().run_search_queue_maxsize == 256

    def test_default_lazy_metric_queue_maxsize(self) -> None:
        assert Settings().lazy_metric_queue_maxsize == 256

    def test_default_custom_search_queue_maxsize(self) -> None:
        assert Settings().custom_search_queue_maxsize == 128


class TestRunSearchQueueConfig:
    def test_uses_configured_maxsize(self, db: Database, client: TestClient) -> None:
        _seed_run(db, "qrs1")
        captured: list[int] = []
        spy = _make_queue_spy(captured)

        with (
            patch("matyan_backend.api.runs._streaming.SETTINGS.run_search_queue_maxsize", 100),
            patch("matyan_backend.api.runs._streaming.asyncio.Queue", spy),
        ):
            resp = client.get(f"{BASE}/runs/search/run/", params={"report_progress": "false"})

        assert resp.status_code == 200
        assert 100 in captured


class TestLazyMetricQueueConfig:
    def test_uses_configured_maxsize(self, db: Database, client: TestClient) -> None:
        _seed_run(db, "qlm1")
        captured: list[int] = []
        spy = _make_queue_spy(captured)

        with (
            patch("matyan_backend.api.runs._streaming.SETTINGS.lazy_metric_queue_maxsize", 200),
            patch("matyan_backend.api.runs._streaming.asyncio.Queue", spy),
            patch(
                "matyan_backend.api.runs._streaming.plan_query",
                return_value=type("R", (), {"candidates": None, "exact": False})(),
            ),
        ):
            resp = client.get(
                f"{BASE}/runs/search/metric/",
                params={"q": 'metric.name == "loss"', "report_progress": "false"},
            )

        assert resp.status_code == 200
        assert 200 in captured


class TestCustomSearchQueueConfig:
    def test_uses_configured_maxsize(self, db: Database, client: TestClient) -> None:
        _seed_text_run(db, "qco1")
        captured: list[int] = []
        spy = _make_queue_spy(captured)

        with (
            patch("matyan_backend.api.runs._custom_objects.SETTINGS.custom_search_queue_maxsize", 64),
            patch("matyan_backend.api.runs._custom_objects.asyncio.Queue", spy),
        ):
            resp = client.get(
                f"{BASE}/runs/search/texts/",
                params={"report_progress": "false"},
            )

        assert resp.status_code == 200
        assert 64 in captured
