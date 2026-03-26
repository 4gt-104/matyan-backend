import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import generate_latest
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from .api import main_router
from .api.errors import ErrorEnvelope
from .api.health import health_router
from .config import SETTINGS
from .kafka.producer import get_ingestion_producer, get_producer
from .logging import configure_logging
from .metrics import HTTP_REQUEST_DURATION, HTTP_REQUESTS_TOTAL, normalize_path
from .storage.fdb_client import ensure_directories, get_db, init_fdb
from .storage.project import get_project_params_cached, init_params_cache

configure_logging(SETTINGS.log_level)


class TrailingSlashMiddleware(BaseHTTPMiddleware):
    """Normalize request paths so that ``/foo`` matches routes defined as ``/foo/``.

    If the incoming path does not end with ``/``, append one before
    dispatching.  This avoids 307 redirects that break CORS preflight.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not request.url.path.endswith("/"):
            request.scope["path"] = request.url.path + "/"
        return await call_next(request)


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status and elapsed time; record Prometheus metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        logger.trace(
            "{method} {path} -> {status} ({elapsed:.1f}ms)",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed=elapsed * 1000,
        )
        if SETTINGS.metrics_enabled:
            path_template = normalize_path(request.url.path)
            status_class = f"{response.status_code // 100}xx"
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                path_template=path_template,
                status_class=status_class,
            ).inc()
            HTTP_REQUEST_DURATION.labels(
                method=request.method,
                path_template=path_template,
                status_class=status_class,
            ).observe(elapsed)
        return response


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_fdb()
    ensure_directories()
    control_producer = get_producer()
    ingestion_producer = get_ingestion_producer()
    await control_producer.start()
    await ingestion_producer.start()

    init_params_cache(
        maxsize=SETTINGS.project_params_cache_maxsize,
        ttl=SETTINGS.project_params_cache_ttl,
    )
    try:
        get_project_params_cached(
            get_db(),
            sequence_types=("metric", "images", "figures", "texts", "audios", "distributions"),
            exclude_params=True,
        )
        logger.info("Project params cache warmed")
    except Exception:  # noqa: BLE001
        logger.warning("Failed to warm project params cache (non-fatal)")

    try:
        yield
    finally:
        await ingestion_producer.stop()
        await control_producer.stop()


app = FastAPI(title="Matyan Server", lifespan=lifespan)


async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Convert HTTPException to standard error envelope."""
    envelope = ErrorEnvelope.from_http_exception(exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=envelope.model_dump(),
    )


async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Convert 422 validation errors to standard error envelope."""
    envelope = ErrorEnvelope.from_http_exception(422, exc.errors())
    return JSONResponse(status_code=422, content=envelope.model_dump())


async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Return 500 with generic message and log traceback."""
    logger.exception("Unhandled exception: %s", exc)
    envelope = ErrorEnvelope.from_http_exception(500, None)
    return JSONResponse(status_code=500, content=envelope.model_dump())


app.add_exception_handler(HTTPException, http_exception_handler)  # ty:ignore[invalid-argument-type]
app.add_exception_handler(RequestValidationError, validation_exception_handler)  # ty:ignore[invalid-argument-type]
app.add_exception_handler(Exception, unhandled_exception_handler)

app.add_middleware(RequestTimingMiddleware)  # ty:ignore[invalid-argument-type]

app.add_middleware(TrailingSlashMiddleware)  # ty:ignore[invalid-argument-type]

app.add_middleware(
    CORSMiddleware,  # ty:ignore[invalid-argument-type]
    allow_origins=SETTINGS.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_router = APIRouter(tags=["metrics"])


@metrics_router.get("/metrics/")
async def prometheus_metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


app.include_router(main_router, prefix="/api/v1")
app.include_router(health_router)
app.include_router(metrics_router)
