"""Health and readiness probe endpoints.

``/health/live`` — lightweight liveness check (process is running).
``/health/ready`` — readiness check: verifies FDB connectivity and that
Kafka producers are started.  Returns 503 if any dependency is unhealthy.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from matyan_backend.kafka.producer import get_ingestion_producer, get_producer
from matyan_backend.storage.fdb_client import ping

health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get("/live/")
async def liveness() -> dict:
    """Lightweight liveness probe: returns 200 if the process is running."""
    return {"status": "ok"}


def _check_fdb() -> bool:
    return ping()


def _check_kafka() -> dict[str, bool]:
    control = get_producer()
    ingestion = get_ingestion_producer()
    return {
        "control_producer": control._producer is not None,  # noqa: SLF001
        "ingestion_producer": ingestion._producer is not None,  # noqa: SLF001
    }


@health_router.get("/ready/")
async def readiness() -> JSONResponse:
    """Readiness probe: checks FDB and Kafka; returns 503 if any dependency is unhealthy."""
    checks: dict[str, object] = {}
    healthy = True

    try:
        await asyncio.to_thread(_check_fdb)
        checks["fdb"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["fdb"] = str(exc)
        healthy = False

    try:
        kafka_status = _check_kafka()
        checks["kafka"] = kafka_status
        if not all(kafka_status.values()):
            healthy = False
    except Exception as exc:  # noqa: BLE001
        checks["kafka"] = str(exc)
        healthy = False

    status_code = 200 if healthy else 503
    return JSONResponse({"status": "ok" if healthy else "degraded", "checks": checks}, status_code=status_code)
