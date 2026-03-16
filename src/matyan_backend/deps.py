"""FastAPI dependency injection functions for route handlers.

Route handlers use these via ``Depends()`` or the ``Annotated`` aliases
(``FdbDb``, ``FdbDirs``) to receive the FDB database handle and directory
references.  No manager/service layer — routes call ``storage.*`` functions
directly.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from .fdb_types import Database
from .kafka.producer import ControlEventProducer, DataIngestionProducer, get_ingestion_producer, get_producer
from .storage.fdb_client import Directories, get_db, get_directories


def fdb_db() -> Database:
    """Return the FDB Database handle (initialized at app startup)."""
    return get_db()


def fdb_dirs() -> Directories:
    """Return the cached FDB Directories namedtuple."""
    return get_directories()


def kafka_producer() -> ControlEventProducer:
    """Return the Kafka control-event producer (started at app startup)."""
    return get_producer()


def kafka_ingestion_producer() -> DataIngestionProducer:
    """Return the Kafka data-ingestion producer (started at app startup)."""
    return get_ingestion_producer()


# Annotated aliases — use as parameter types in route signatures so FastAPI
# automatically injects the dependency (e.g. ``db: FdbDb`` in a route).
FdbDb = Annotated[Database, Depends(fdb_db)]
FdbDirs = Annotated[Directories, Depends(fdb_dirs)]
KafkaProducerDep = Annotated[ControlEventProducer, Depends(kafka_producer)]
IngestionProducerDep = Annotated[DataIngestionProducer, Depends(kafka_ingestion_producer)]
