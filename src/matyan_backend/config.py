from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dev defaults used only for production guard checks (do not use these in production).
_DEV_BLOB_URI_SECRET = "Juw5-cLlemQI2jAWvceOUB3_CrVfBmI99YIzkpGUXR4="  # noqa: S105
_DEV_S3_CRED = "rustfsadmin"
_DEV_S3_ENDPOINT = "http://localhost:9000"
_DEV_KAFKA_BOOTSTRAP = "localhost:9092"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Set to "production" (env: MATYAN_ENVIRONMENT) to enable strict checks: sensitive/critical
    # settings must be overridden (no dev defaults). Default "development" keeps current behavior.
    environment: str = Field(
        default="development",
        validation_alias=AliasChoices("MATYAN_ENVIRONMENT", "ENVIRONMENT"),
    )

    log_level: str = "INFO"

    server_url: str = "http://localhost:53800"

    # FoundationDB
    fdb_cluster_file: str = "fdb.cluster"
    fdb_api_version: int = 730
    fdb_retry_max_attempts: int = 5
    fdb_retry_initial_delay_sec: float = 0.05
    fdb_retry_max_delay_sec: float = 2.0

    # S3 (RustFS in dev, AWS S3 in prod)
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "rustfsadmin"
    s3_secret_key: str = "rustfsadmin"  # noqa: S105
    s3_bucket: str = "matyan-artifacts"
    s3_region: str = "us-east-1"

    # Blob URI encryption key (Fernet URL-safe base64-encoded 32 bytes)
    blob_uri_secret: str = "Juw5-cLlemQI2jAWvceOUB3_CrVfBmI99YIzkpGUXR4="  # noqa: S105

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_data_ingestion_topic: str = "data-ingestion"
    kafka_control_events_topic: str = "control-events"
    kafka_security_protocol: str = ""
    kafka_sasl_mechanism: str = ""
    kafka_sasl_username: str = ""
    kafka_sasl_password: str = ""

    # Ingestion worker batching
    ingest_batch_size: int = 200
    ingest_batch_timeout_ms: int = 100
    ingest_max_messages_per_txn: int = 100
    ingest_max_txn_bytes: int = 8 * 1024 * 1024  # 8 MB target; FDB hard limit is 10 MB

    # Prometheus metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090

    # Project params cache
    project_params_cache_ttl: int = 30
    project_params_cache_maxsize: int = 32

    # Streaming / search queue sizes
    run_search_queue_maxsize: int = 256
    lazy_metric_queue_maxsize: int = 256
    custom_search_queue_maxsize: int = 128
    metric_candidate_run_concurrency: int = 2
    metric_trace_chunk_concurrency: int = 2
    fdb_thread_pool_size: int = 16

    # Query timing (verbose per-step logs for superset/lazy paths)
    query_timing_enabled: bool = False

    # Periodic cleanup jobs
    tombstone_cleanup_older_than_hours: int = 168
    cleanup_job_lock_ttl_seconds: int = 0

    # CORS
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8000",
        "http://localhost:53800",
        "http://127.0.0.1:53800",
    )


def validate_production_settings(settings: Settings) -> None:  # noqa: C901
    """When environment is production, require that sensitive/critical settings are not dev defaults.

    Raises ValueError with a clear message on first violation. Call after Settings() is built.
    """
    if settings.environment != "production":
        return
    if settings.blob_uri_secret == _DEV_BLOB_URI_SECRET:
        msg = (
            "In production, BLOB_URI_SECRET must be set explicitly and must not be the default dev value. "
            "Set from env or a secrets backend."
        )
        raise ValueError(msg)
    if not settings.blob_uri_secret.strip():
        msg = "In production, BLOB_URI_SECRET must be set (non-empty). Supply via env or a secrets backend."
        raise ValueError(msg)
    if settings.s3_access_key == _DEV_S3_CRED:
        msg = (
            "In production, S3_ACCESS_KEY must be set explicitly and must not be the default dev value. "
            "Set from env or a secrets backend."
        )
        raise ValueError(msg)
    if settings.s3_secret_key == _DEV_S3_CRED:
        msg = (
            "In production, S3_SECRET_KEY must be set explicitly and must not be the default dev value. "
            "Set from env or a secrets backend."
        )
        raise ValueError(msg)
    if settings.s3_endpoint == _DEV_S3_ENDPOINT:
        msg = "In production, S3_ENDPOINT must be set explicitly and must not be the default dev value. Set from env."
        raise ValueError(msg)
    if not settings.s3_endpoint.strip():
        msg = "In production, S3_ENDPOINT must be set (non-empty)."
        raise ValueError(msg)
    if settings.kafka_bootstrap_servers == _DEV_KAFKA_BOOTSTRAP:
        msg = (
            "In production, KAFKA_BOOTSTRAP_SERVERS must be set explicitly and must not be the default dev value. "
            "Set from env."
        )
        raise ValueError(msg)
    if not settings.kafka_bootstrap_servers.strip():
        msg = "In production, KAFKA_BOOTSTRAP_SERVERS must be set (non-empty)."
        raise ValueError(msg)
    if not settings.fdb_cluster_file.strip():
        msg = "In production, FDB_CLUSTER_FILE must be set (non-empty)."
        raise ValueError(msg)


SETTINGS = Settings()
validate_production_settings(SETTINGS)
