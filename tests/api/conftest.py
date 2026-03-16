"""Shared fixtures for API-level tests.

Provides a ``client`` fixture backed by FastAPI's ``TestClient``.  Each test
gets its own isolated FDB directory prefix (same pattern as storage tests)
so tests don't interfere with each other.

Requires a running FDB instance (``docker compose up -d``).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from fdb.directory_impl import directory as _fdb_directory

from matyan_backend.app import app
from matyan_backend.deps import kafka_ingestion_producer, kafka_producer
from matyan_backend.kafka.producer import ControlEventProducer, DataIngestionProducer
from matyan_backend.storage import fdb_client

if TYPE_CHECKING:
    from collections.abc import Generator

    from matyan_api_models.kafka import ControlEvent, IngestionMessage

    from matyan_backend.fdb_types import Database
    from matyan_backend.storage.fdb_client import Directories


class _NoOpProducer(ControlEventProducer):
    """Kafka producer stub that silently discards events during tests."""

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def publish(self, event: ControlEvent) -> None:
        pass


class _NoOpIngestionProducer(DataIngestionProducer):
    """Kafka data-ingestion producer stub that silently discards messages during tests."""

    def __init__(self) -> None:
        super().__init__()
        self.published: list[IngestionMessage] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def publish(self, message: IngestionMessage) -> None:
        self.published.append(message)


@pytest.fixture(scope="session", autouse=True)
def _init_fdb() -> None:
    fdb_client.init_fdb()


@pytest.fixture(autouse=True)
def _isolated_directories(monkeypatch: pytest.MonkeyPatch) -> Generator[Directories, None, None]:
    db = fdb_client.get_db()
    test_id = uuid.uuid4().hex[:12]
    prefix = ("_test_api", test_id)

    runs = _fdb_directory.create_or_open(db, (*prefix, "data", "runs"))
    indexes = _fdb_directory.create_or_open(db, (*prefix, "data", "indexes"))
    system = _fdb_directory.create_or_open(db, (*prefix, "system"))
    dirs = fdb_client.Directories(runs=runs, indexes=indexes, system=system)

    monkeypatch.setattr(fdb_client, "_directories", dirs)

    yield dirs

    if _fdb_directory.exists(db, prefix):
        _fdb_directory.remove(db, prefix)


@pytest.fixture
def db() -> Database:
    """Return the FDB database handle."""
    return fdb_client.get_db()


@pytest.fixture(autouse=True)
def _override_kafka_producers() -> Generator[None, None, None]:
    """Replace both Kafka producer dependencies with no-op stubs."""
    app.dependency_overrides[kafka_producer] = _NoOpProducer
    app.dependency_overrides[kafka_ingestion_producer] = _NoOpIngestionProducer
    yield
    app.dependency_overrides.pop(kafka_producer, None)
    app.dependency_overrides.pop(kafka_ingestion_producer, None)


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient for the FastAPI app.

    The lifespan is *not* triggered (FDB is already initialized by
    ``_init_fdb`` and directories are patched by ``_isolated_directories``).
    """
    return TestClient(app, raise_server_exceptions=True)


BASE = "/api/v1/rest"
